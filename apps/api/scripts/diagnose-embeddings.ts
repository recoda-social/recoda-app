/**
 * 仮説 H1（現行の埋め込み空間がアーティスト・リリース年代・文字数にどれくらい引っ張られているか）を
 * D1（2048 次元 x）または Turso（現行 256 次元 z）に対して実測する診断スクリプト。
 *
 * 接続モード（自動判別）：
 * - D1 モード：CLOUDFLARE_ACCOUNT_ID と WORKERS_AI_API_TOKEN がある場合。D1 の x（2048 次元）から
 *   非評価データのみで PCA 256 次元を学習し、2048 次元と PCA 256 次元の両方を診断する。
 * - Turso モード：上記が無く TURSO_DATABASE_URL がある場合。Turso の post_embeddings から
 *   現行モデル（EMBEDDING_MODEL）の 256 次元 z を直接読んで診断 1・2・3 を実行する。
 *
 * 計測内容：
 * 1. 現行 PCA 256 次元（および D1 モード時は 2048 次元そのまま）の「不要因子の漏れ率」
 *    - 同一アーティスト率（全体 / aspect != artist / 本文にアーティスト名あり・なし別）
 *    - 同一年代率（全体 / aspect != era かつ mood != nostalgic の投稿のみ）
 *    - 上位 3 件との文字数相関
 * 2. PCA 256 次元空間からの 5-fold CV リッジ回帰 R²
 *    - aspect / mood / リリース年 / リリース年（aspect・mood 残差後）/ 頻出アーティスト / 文字数
 * 3. 学習データ（eval 700 件を除く）から推定した不要方向（アーティスト重心部分空間・年代方向）を
 *    ソフト減衰（λ = 0..1）したときの感度チェック
 */
import { readFile } from "node:fs/promises";
import { createClient } from "@libsql/client";
import { postEmbeddings, relations } from "db";
import { EMBEDDING_DIMENSIONS, EMBEDDING_INPUT_DIMENSIONS, EMBEDDING_MODEL } from "db/constants";
import { officialPostId } from "db/official-reviews";
import { eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/libsql";
import { restVectorStore } from "./lib/cloudflare.ts";
import {
  attenuateDirection,
  attenuateSubspace,
  decadeOfYear,
  estimateArtistSubspace,
  evaluateLeakage,
  parseIsrcYear,
  randomLeakageBaseline,
  residualizeByFeatures,
  ridgeCvR2,
  ridgeFit,
} from "./lib/diagnose.ts";
import type { DiagnosticItem } from "./lib/diagnose.ts";
import { fitTransforms } from "./lib/pca.ts";
import type { Transform } from "./lib/pca.ts";

type OfficialData = {
  songs: Array<{
    isrc: string;
    title?: string | null;
    artistName?: string | null;
    artistIds?: Array<string> | null;
  }>;
  posts: Array<{
    key: string;
    isrc: string;
    content: string;
  }>;
};

type EvalData = {
  reviews: Array<{
    key: string;
    isrc: string;
    aspectPrimary: string;
    aspects: Array<string>;
    moodPrimary: string;
    moods: Array<string>;
  }>;
};

const D = EMBEDDING_INPUT_DIMENSIONS;
const K = EMBEDDING_DIMENSIONS;
const RIDGE_ALPHAS = [0.1, 1, 10, 50, 200, 1000];
const ARTIST_SUBSPACE_RANK = 32;
const LAMBDAS = [0, 0.25, 0.5, 0.75, 1.0];

function fail(error: Error): never {
  console.error(error);
  process.exit(1);
}

function isOfficialData(value: unknown): value is OfficialData {
  return (
    typeof value === "object" &&
    value !== null &&
    "songs" in value &&
    Array.isArray(value.songs) &&
    "posts" in value &&
    Array.isArray(value.posts)
  );
}

function isEvalData(value: unknown): value is EvalData {
  return (
    typeof value === "object" &&
    value !== null &&
    "reviews" in value &&
    Array.isArray(value.reviews)
  );
}

const [rawOfficial, rawEval]: [unknown, unknown] = await Promise.all([
  readFile(
    new URL("../../../packages/db/data/official-reviews.json", import.meta.url).pathname,
    "utf8",
  ).then((text) => JSON.parse(text) as unknown),
  readFile(
    new URL("../../../packages/db/data/embedding-eval.json", import.meta.url).pathname,
    "utf8",
  ).then((text) => JSON.parse(text) as unknown),
]);

if (!isOfficialData(rawOfficial)) {
  fail(new Error("official-reviews.json の形式が不正です"));
}
if (!isEvalData(rawEval)) {
  fail(new Error("embedding-eval.json の形式が不正です"));
}

const songByIsrc = new Map(rawOfficial.songs.map((song) => [song.isrc, song]));
const postByKey = new Map(rawOfficial.posts.map((post) => [post.key, post]));
const evalKeySet = new Set(rawEval.reviews.map((review) => review.key));

const allPostIds = rawOfficial.posts.map((post) => officialPostId(post.key));
const hasD1Env = Boolean(process.env.CLOUDFLARE_ACCOUNT_ID && process.env.WORKERS_AI_API_TOKEN);
const hasTursoEnv = Boolean(process.env.TURSO_DATABASE_URL);
if (!hasD1Env && !hasTursoEnv) {
  fail(
    new Error(
      "CLOUDFLARE_ACCOUNT_ID + WORKERS_AI_API_TOKEN（D1 モード）または TURSO_DATABASE_URL（Turso モード）を設定してください",
    ),
  );
}

const artistKeyOf = (isrc: string) => {
  const song = songByIsrc.get(isrc);
  return song?.artistIds?.[0] ?? song?.artistName ?? `isrc:${isrc}`;
};

let sourceLabel: string;
let explainedPca: number | null = null;
let trainMeta: Array<{
  key: string;
  isrc: string;
  artistKey: string;
  year: number | null;
  z: Float64Array;
}> = [];
let labeled: Array<DiagnosticItem & { year: number | null; x?: Float32Array; z: Float64Array }> =
  [];

if (hasD1Env) {
  const storeName = process.env.X_STORE_NAME ?? "x-store-staging-01";
  sourceLabel = `D1:${storeName}`;
  const store = await restVectorStore(storeName);
  if (store instanceof Error) {
    fail(store);
  }
  const xs = await store.get(allPostIds);
  if (xs instanceof Error) {
    fail(xs);
  }

  const rawTrain = rawOfficial.posts.flatMap((post) => {
    if (evalKeySet.has(post.key)) {
      return [];
    }
    const x = xs.get(officialPostId(post.key));
    if (!x) {
      return [];
    }
    const year = parseIsrcYear(post.isrc);
    return [{ key: post.key, isrc: post.isrc, artistKey: artistKeyOf(post.isrc), year, x }];
  });

  const rawLabeled = rawEval.reviews.flatMap((review) => {
    const post = postByKey.get(review.key);
    const x = xs.get(officialPostId(review.key));
    if (!post || !x) {
      return [];
    }
    const song = songByIsrc.get(review.isrc);
    const year = parseIsrcYear(review.isrc);
    const title = song?.title ?? "";
    const artistName = song?.artistName ?? "";
    return [
      {
        artistKey: artistKeyOf(review.isrc),
        decade: year !== null ? decadeOfYear(year) : null,
        aspectPrimary: review.aspectPrimary,
        aspects: review.aspects,
        moodPrimary: review.moodPrimary,
        moods: review.moods,
        contentLength: post.content.length,
        mentionsArtist: artistName.length > 0 && post.content.includes(artistName),
        mentionsTitle: title.length > 0 && post.content.includes(title),
        year,
        x,
      },
    ];
  });

  if (rawTrain.length === 0 || rawLabeled.length === 0) {
    fail(
      new Error(
        `D1（${storeName}）に公式レビューの x が不足しています（train=${rawTrain.length}, eval=${rawLabeled.length}）`,
      ),
    );
  }

  const fitted = fitTransforms(
    rawTrain.map((item) => item.x),
    D,
    K,
  );
  explainedPca = fitted.explained;
  const projectWith = (t: Transform) => (x: Float32Array) => {
    const z = new Float64Array(K);
    for (let r = 0; r < K; r++) {
      let acc = t.b[r] ?? 0;
      const base = r * D;
      for (let j = 0; j < D; j++) {
        acc += (t.a[base + j] ?? 0) * (x[j] ?? 0);
      }
      z[r] = acc;
    }
    return z;
  };
  const projectPca = projectWith(fitted.pca);

  trainMeta = rawTrain.map((item) => ({ ...item, z: projectPca(item.x) }));
  labeled = rawLabeled.map((item) => ({ ...item, z: projectPca(item.x) }));
} else {
  const model = process.env.EMBEDDING_MODEL ?? EMBEDDING_MODEL;
  sourceLabel = `Turso:${model}`;
  const db = drizzle({
    client: createClient({
      url: process.env.TURSO_DATABASE_URL || "http://127.0.0.1:8080",
      authToken: process.env.TURSO_AUTH_TOKEN || undefined,
    }),
    relations,
  });
  const rows = await db
    .select({ postId: postEmbeddings.postId, embedding: postEmbeddings.embedding })
    .from(postEmbeddings)
    .where(eq(postEmbeddings.model, model));
  const zByPostId = new Map(rows.map((row) => [row.postId, Float64Array.from(row.embedding)]));

  trainMeta = rawOfficial.posts.flatMap((post) => {
    if (evalKeySet.has(post.key)) {
      return [];
    }
    const z = zByPostId.get(officialPostId(post.key));
    if (!z) {
      return [];
    }
    const year = parseIsrcYear(post.isrc);
    return [{ key: post.key, isrc: post.isrc, artistKey: artistKeyOf(post.isrc), year, z }];
  });

  labeled = rawEval.reviews.flatMap((review) => {
    const post = postByKey.get(review.key);
    const z = zByPostId.get(officialPostId(review.key));
    if (!post || !z) {
      return [];
    }
    const song = songByIsrc.get(review.isrc);
    const year = parseIsrcYear(review.isrc);
    const title = song?.title ?? "";
    const artistName = song?.artistName ?? "";
    return [
      {
        artistKey: artistKeyOf(review.isrc),
        decade: year !== null ? decadeOfYear(year) : null,
        aspectPrimary: review.aspectPrimary,
        aspects: review.aspects,
        moodPrimary: review.moodPrimary,
        moods: review.moods,
        contentLength: post.content.length,
        mentionsArtist: artistName.length > 0 && post.content.includes(artistName),
        mentionsTitle: title.length > 0 && post.content.includes(title),
        year,
        z,
      },
    ];
  });

  if (trainMeta.length === 0 || labeled.length === 0) {
    fail(
      new Error(
        `Turso（model=${model}）に公式レビューの z が不足しています（train=${trainMeta.length}, eval=${labeled.length}）`,
      ),
    );
  }
}

console.log(
  `source=${sourceLabel} / 学習=${trainMeta.length} 件 / 評価=${labeled.length} 件（欠損=${rawEval.reviews.length - labeled.length} 件）`,
);

const trainZs = trainMeta.map((item) => item.z);
const evalZs = labeled.map((item) => item.z);

// -----------------------------------------------------------------------------
// 診断 1: 現行 PCA 256 次元（および D1 モード時は 2048 次元）の不要因子の漏れ率
// -----------------------------------------------------------------------------
const baseline = randomLeakageBaseline(labeled);
const leak2048 = labeled.every((item) => item.x !== undefined)
  ? evaluateLeakage(
      labeled,
      labeled.map((item) => item.x ?? new Float32Array()),
    )
  : null;
const leakPca = evaluateLeakage(labeled, evalZs);

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const formatLeakRow = (leak: typeof leakPca) => ({
  "aspect@3": pct(leak.aspectAt3),
  "mood@3": pct(leak.moodAt3),
  "同一Artist@10": pct(leak.sameArtistAt10),
  "同一Artist(非artist観点)@10": pct(leak.sameArtistNonArtistAspectAt10),
  "同一年代(非era/nostalgic)@10": pct(leak.sameDecadeNonEraAt10),
  "文字数相関@3": leak.lengthCorrAt3.toFixed(2),
});
const leakageRows: Array<{
  方式: string;
  "aspect@3": string;
  "mood@3": string;
  "同一Artist@10": string;
  "同一Artist(非artist観点)@10": string;
  "同一年代(非era/nostalgic)@10": string;
  "文字数相関@3": string;
}> = [
  {
    方式: "ランダム期待値",
    "aspect@3": pct(baseline.aspect),
    "mood@3": pct(baseline.mood),
    "同一Artist@10": pct(baseline.sameArtistEligible),
    "同一Artist(非artist観点)@10": pct(baseline.sameArtistEligible),
    "同一年代(非era/nostalgic)@10": pct(baseline.sameDecadeNonEra),
    "文字数相関@3": "0.00",
  },
];
if (leak2048) {
  leakageRows.push({ 方式: "2048次元(そのまま)", ...formatLeakRow(leak2048) });
}
leakageRows.push({ 方式: "PCA 256次元(現行)", ...formatLeakRow(leakPca) });
console.log("\n=== 診断 1: 不要因子の漏れ率（評価 700 件） ===");
console.table(leakageRows);
console.log(
  `[PCA 256] アーティスト言及ありでの同一Artist@10: ${pct(leakPca.sameArtistWhenMentionedAt10)} / 言及なし: ${pct(leakPca.sameArtistWhenNotMentionedAt10)}`,
);

// -----------------------------------------------------------------------------
// 診断 2: PCA 256 次元空間における各因子の線形エンコード量（5-fold CV R²）
// -----------------------------------------------------------------------------
const aspectVocab = [...new Set(labeled.flatMap((r) => r.aspects))].sort();
const moodVocab = [...new Set(labeled.flatMap((r) => r.moods))].sort();

const contentFeatures = labeled.map((r) => {
  const vec = new Float64Array(aspectVocab.length + moodVocab.length);
  for (let i = 0; i < aspectVocab.length; i++) {
    vec[i] = r.aspects.includes(aspectVocab[i] ?? "") ? 1 : 0;
  }
  for (let i = 0; i < moodVocab.length; i++) {
    vec[aspectVocab.length + i] = r.moods.includes(moodVocab[i] ?? "") ? 1 : 0;
  }
  return vec;
});

const aspectR2s = aspectVocab.map((label) => {
  const y = Float64Array.from(labeled, (r) => (r.aspects.includes(label) ? 1 : 0));
  return { label, ...ridgeCvR2(evalZs, y, RIDGE_ALPHAS) };
});
const moodR2s = moodVocab.map((label) => {
  const y = Float64Array.from(labeled, (r) => (r.moods.includes(label) ? 1 : 0));
  return { label, ...ridgeCvR2(evalZs, y, RIDGE_ALPHAS) };
});

const yearValidIdx = labeled.flatMap((r, i) => (r.year !== null ? [i] : []));
const yearZs = yearValidIdx.flatMap((i) => {
  const z = evalZs[i];
  return z ? [z] : [];
});
const yearY = Float64Array.from(yearValidIdx, (i) => labeled[i]?.year ?? 0);
const yearContentFeatures = yearValidIdx.flatMap((i) => {
  const f = contentFeatures[i];
  return f ? [f] : [];
});
const yearCv = ridgeCvR2(yearZs, yearY, RIDGE_ALPHAS);
const yearResidualY = residualizeByFeatures(yearY, yearContentFeatures, 1);
const yearResidualCv = ridgeCvR2(yearZs, yearResidualY, RIDGE_ALPHAS);

const lengthY = Float64Array.from(labeled, (r) => r.contentLength);
const lengthCv = ridgeCvR2(evalZs, lengthY, RIDGE_ALPHAS);

// 評価セット内で 5 件以上ある上位アーティストの 1-vs-rest R²（生 & 内容残差後）
const evalArtistCounts = new Map<string, number>();
for (const r of labeled) {
  evalArtistCounts.set(r.artistKey, (evalArtistCounts.get(r.artistKey) ?? 0) + 1);
}
const frequentArtists = [...evalArtistCounts.entries()]
  .filter(([, c]) => c >= 5)
  .sort((a, b) => b[1] - a[1]);

const artistR2s = frequentArtists.map(([artistKey, count]) => {
  const y = Float64Array.from(labeled, (r) => (r.artistKey === artistKey ? 1 : 0));
  const raw = ridgeCvR2(evalZs, y, RIDGE_ALPHAS);
  const resY = residualizeByFeatures(y, contentFeatures, 1);
  const residual = ridgeCvR2(evalZs, resY, RIDGE_ALPHAS);
  return { artistKey, count, rawR2: raw.r2, residualR2: residual.r2 };
});

const mean = (arr: Array<number>) =>
  arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

const probeSummary = {
  "aspect (10種平均)": { cvR2: mean(aspectR2s.map((x) => x.r2)).toFixed(3) },
  "mood (11種平均)": { cvR2: mean(moodR2s.map((x) => x.r2)).toFixed(3) },
  "リリース年 (全体)": { cvR2: yearCv.r2.toFixed(3) },
  "リリース年 (内容残差後)": { cvR2: yearResidualCv.r2.toFixed(3) },
  [`頻出アーティスト (${artistR2s.length}組平均)`]: {
    cvR2: mean(artistR2s.map((x) => x.rawR2)).toFixed(3),
  },
  [`頻出アーティスト (内容残差後)`]: {
    cvR2: mean(artistR2s.map((x) => x.residualR2)).toFixed(3),
  },
  文字数: { cvR2: lengthCv.r2.toFixed(3) },
};
console.log("\n=== 診断 2: PCA 256 次元からの 5-fold CV 線形予測 R² ===");
console.table(probeSummary);

// -----------------------------------------------------------------------------
// 診断 3: 学習データから推定した不要方向のソフト減衰（λ スイープ）
// -----------------------------------------------------------------------------
const artistBasis = estimateArtistSubspace(
  trainMeta.map((item, i) => ({
    artistKey: item.artistKey,
    z: trainZs[i] ?? new Float64Array(K),
  })),
  K,
  ARTIST_SUBSPACE_RANK,
);
const artistRank = Math.floor(artistBasis.length / K);

const trainYearIdx = trainMeta.flatMap((item, i) => (item.year !== null ? [i] : []));
const trainYearZs = trainYearIdx.flatMap((i) => {
  const z = trainZs[i];
  return z ? [z] : [];
});
const trainYearY = Float64Array.from(trainYearIdx, (i) => trainMeta[i]?.year ?? 0);
const yearModel = ridgeFit(trainYearZs, trainYearY, 50);
let yearNormSq = 0;
for (let j = 0; j < K; j++) {
  const v = yearModel.w[j] ?? 0;
  yearNormSq += v * v;
}
const yearNorm = Math.sqrt(yearNormSq) || 1;
const yearUnitDir = Float64Array.from(yearModel.w, (v) => v / yearNorm);

const artistSweep = LAMBDAS.map((lambda) => {
  const attenuated = evalZs.map((z) => attenuateSubspace(z, artistBasis, artistRank, lambda));
  const res = evaluateLeakage(labeled, attenuated);
  return {
    lambda,
    "aspect@3": pct(res.aspectAt3),
    "mood@3": pct(res.moodAt3),
    "aspect@10": pct(res.aspectAt10),
    "mood@10": pct(res.moodAt10),
    "同一Artist(非artist)@10": pct(res.sameArtistNonArtistAspectAt10),
    "同一年代(非era)@10": pct(res.sameDecadeNonEraAt10),
    raw: res,
  };
});

const yearSweep = LAMBDAS.map((lambda) => {
  const attenuated = evalZs.map((z) => attenuateDirection(z, yearUnitDir, lambda));
  const res = evaluateLeakage(labeled, attenuated);
  return {
    lambda,
    "aspect@3": pct(res.aspectAt3),
    "mood@3": pct(res.moodAt3),
    "aspect@10": pct(res.aspectAt10),
    "mood@10": pct(res.moodAt10),
    "同一Artist(非artist)@10": pct(res.sameArtistNonArtistAspectAt10),
    "同一年代(非era)@10": pct(res.sameDecadeNonEraAt10),
    raw: res,
  };
});

console.log(`\n=== 診断 3a: アーティスト重心部分空間（上位 ${artistRank} 方向）のソフト減衰 ===`);
console.table(artistSweep.map(({ raw: _, ...rest }) => rest));

console.log("\n=== 診断 3b: リリース年方向（1 方向・無条件）のソフト減衰（交絡確認用） ===");
console.table(yearSweep.map(({ raw: _, ...rest }) => rest));

console.log(
  JSON.stringify({
    source: sourceLabel,
    trainCount: trainMeta.length,
    evalCount: labeled.length,
    explainedPca,
    baseline,
    leak2048,
    leakPca,
    probes: {
      aspectMeanR2: mean(aspectR2s.map((x) => x.r2)),
      moodMeanR2: mean(moodR2s.map((x) => x.r2)),
      yearR2: yearCv.r2,
      yearResidualR2: yearResidualCv.r2,
      artistMeanRawR2: mean(artistR2s.map((x) => x.rawR2)),
      artistMeanResidualR2: mean(artistR2s.map((x) => x.residualR2)),
      lengthR2: lengthCv.r2,
      aspectR2s,
      moodR2s,
      artistR2s,
    },
    artistSweep: artistSweep.map(({ lambda, raw }) => ({ lambda, ...raw })),
    yearSweep: yearSweep.map(({ lambda, raw }) => ({ lambda, ...raw })),
  }),
);
