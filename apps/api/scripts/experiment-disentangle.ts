/**
 * 仮説 H2（Step 1：文字数方向の除去 ＋ Step 2：教師あり直交ブロック射影）を
 * D1 の 2048 次元ベクトル x に対して厳密比較する実験スクリプト。
 *
 * 比較する内容（すべて非評価 2,978 件 + 評価 700 件の 5-fold CV でリークゼロ検証）：
 * 1. Step 1 単体比較（文字数相関 0.64 の除去）：
 *    - Baseline：現行 PCA 256 次元
 *    - 共分散直交射影（α = ∞、len / log(len)）
 *    - Shrinkage-LEACE（α ∈ {10, 100, 1000}、len / log(len)）
 * 2. Step 2 比較（[z_aspect (10) | z_mood (11) | z_residual (235)] = 256 次元）：
 *    - A. 多出力 Ridge 予測スコア（独立連結）
 *    - B. 特異値重み付き PLS（クロス共分散 SVD）
 *    - C. Löwdin 対称直交化＋予測力スケーリング
 *    それぞれについて、ブロック重み比率 (w_asp², w_mood², w_res²) と
 *    単独ブロック検索（Disentanglement の直接確認）を評価する。
 * 3. ガードレール検証（経路②：短い言葉検索 53 クエリ @10）：
 *    - Workers AI で 53 クエリを埋め込み、提案手法が言葉検索を劣化させないか確認する。
 *
 * 必要な環境変数：CLOUDFLARE_ACCOUNT_ID、WORKERS_AI_API_TOKEN、X_STORE_NAME（既定: x-store-staging-01）
 */
import { readFile } from "node:fs/promises";
import { EMBEDDING_DIMENSIONS, EMBEDDING_INPUT_DIMENSIONS } from "db/constants";
import { officialPostId } from "db/official-reviews";
import { restEmbedder, restVectorStore } from "./lib/cloudflare.ts";
import { decadeOfYear, evaluateLeakage, parseIsrcYear } from "./lib/diagnose.ts";
import type { DiagnosticItem } from "./lib/diagnose.ts";
import {
  applyEraser,
  blockRmsNorm,
  crossCovarianceScalar,
  dualRidgeMultiFit,
  fitCovOrthoEraser,
  fitPlsBlock,
  fitResidualPca,
  fitShrinkageLeaceEraser,
  projectRows,
  symmetricOrthogonalizeScaled,
} from "./lib/disentangle.ts";
import type { Rank1Eraser } from "./lib/disentangle.ts";
import { covariance, topEigen } from "./lib/pca.ts";

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
  queries: Array<{ text: string; aspect: string | null; mood: string | null }>;
};

const D = EMBEDDING_INPUT_DIMENSIONS;
const K = EMBEDDING_DIMENSIONS;
const FOLDS = 5;

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
    Array.isArray(value.reviews) &&
    "queries" in value &&
    Array.isArray(value.queries)
  );
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function normalizeVec(v: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(v);
  let sq = 0;
  for (let i = 0; i < out.length; i++) {
    const val = out[i] ?? 0;
    sq += val * val;
  }
  const norm = Math.sqrt(sq) || 1;
  for (let i = 0; i < out.length; i++) {
    out[i] = (out[i] ?? 0) / norm;
  }
  return out;
}

/** 検証 fold のクエリ群から全候補（または未見 fold 内）を検索して aspect/mood @3/@10 と文字数相関を測る。 */
function evalSubsetRetrieval(
  items: Array<DiagnosticItem>,
  vectors: Array<Float64Array>,
  queryIndices: Array<number>,
  candidateIndices: Array<number>,
) {
  const vs = vectors.map(normalizeVec);
  let a3 = 0;
  let m3 = 0;
  let a10 = 0;
  let m10 = 0;
  const qLens: Array<number> = [];
  const top3Lens: Array<number> = [];

  for (const qi of queryIndices) {
    const qItem = items[qi];
    const qv = vs[qi];
    if (!qItem || !qv) {
      continue;
    }
    const scored: Array<[number, number]> = [];
    for (const cj of candidateIndices) {
      if (cj === qi) {
        continue;
      }
      const cv = vs[cj];
      if (!cv) {
        continue;
      }
      let dot = 0;
      for (let t = 0; t < qv.length; t++) {
        dot += (qv[t] ?? 0) * (cv[t] ?? 0);
      }
      scored.push([dot, cj]);
    }
    scored.sort((x, y) => y[0] - x[0]);
    const top3 = scored.slice(0, 3).flatMap(([, j]) => {
      const it = items[j];
      return it ? [it] : [];
    });
    const top10 = scored.slice(0, 10).flatMap(([, j]) => {
      const it = items[j];
      return it ? [it] : [];
    });
    const k3 = top3.length || 1;
    const k10 = top10.length || 1;
    a3 += top3.filter((r) => r.aspectPrimary === qItem.aspectPrimary).length / k3;
    m3 += top3.filter((r) => r.moodPrimary === qItem.moodPrimary).length / k3;
    a10 += top10.filter((r) => r.aspectPrimary === qItem.aspectPrimary).length / k10;
    m10 += top10.filter((r) => r.moodPrimary === qItem.moodPrimary).length / k10;
    qLens.push(qItem.contentLength);
    top3Lens.push(top3.reduce((s, r) => s + r.contentLength, 0) / k3);
  }
  const n = queryIndices.length || 1;
  // Pearson correlation for length
  let mx = 0;
  let my = 0;
  for (let i = 0; i < qLens.length; i++) {
    mx += (qLens[i] ?? 0) / n;
    my += (top3Lens[i] ?? 0) / n;
  }
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < qLens.length; i++) {
    const a = (qLens[i] ?? 0) - mx;
    const b = (top3Lens[i] ?? 0) - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const denom = Math.sqrt(dx * dy);
  return {
    aspectAt3: a3 / n,
    moodAt3: m3 / n,
    aspectAt10: a10 / n,
    moodAt10: m10 / n,
    lengthCorrAt3: denom > 0 ? num / denom : 0,
  };
}

function evalTextQueries(
  items: Array<DiagnosticItem>,
  docVectors: Array<Float64Array>,
  queries: EvalData["queries"],
  queryVectors: Array<Float64Array>,
) {
  const docs = docVectors.map(normalizeVec);
  const qs = queryVectors.map(normalizeVec);
  let hits = 0;
  for (let qi = 0; qi < queries.length; qi++) {
    const q = queries[qi];
    const qv = qs[qi];
    if (!q || !qv) {
      continue;
    }
    const scored = docs.map((dv, j): [number, number] => {
      let dot = 0;
      for (let t = 0; t < qv.length; t++) {
        dot += (qv[t] ?? 0) * (dv[t] ?? 0);
      }
      return [dot, j];
    });
    scored.sort((a, b) => b[0] - a[0]);
    const top10 = scored.slice(0, 10).flatMap(([, j]) => {
      const it = items[j];
      return it ? [it] : [];
    });
    hits +=
      top10.filter(
        (r) => (!q.aspect || r.aspects.includes(q.aspect)) && (!q.mood || r.moods.includes(q.mood)),
      ).length / 10;
  }
  return queries.length > 0 ? hits / queries.length : 0;
}

// -----------------------------------------------------------------------------
// データ読み込み
// -----------------------------------------------------------------------------
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
const officialData = rawOfficial;
const evalData = rawEval;

const songByIsrc = new Map(officialData.songs.map((song) => [song.isrc, song]));
const postByKey = new Map(officialData.posts.map((post) => [post.key, post]));
const evalKeySet = new Set(evalData.reviews.map((review) => review.key));

const storeName = process.env.X_STORE_NAME ?? "x-store-staging-01";
const store = await restVectorStore(storeName);
if (store instanceof Error) {
  fail(store);
}
const allPostIds = rawOfficial.posts.map((post) => officialPostId(post.key));
const [xs, queryXs] = await Promise.all([
  store.get(allPostIds),
  restEmbedder(rawEval.queries.map((q) => q.text)),
]);
if (xs instanceof Error) {
  fail(xs);
}
if (queryXs instanceof Error) {
  fail(queryXs);
}

const artistKeyOf = (isrc: string) => {
  const song = songByIsrc.get(isrc);
  return song?.artistIds?.[0] ?? song?.artistName ?? `isrc:${isrc}`;
};

const trainItems = rawOfficial.posts.flatMap((post) => {
  if (evalKeySet.has(post.key)) {
    return [];
  }
  const x = xs.get(officialPostId(post.key));
  if (!x) {
    return [];
  }
  return [{ key: post.key, contentLength: post.content.length, x }];
});

const labeled = rawEval.reviews.flatMap((review) => {
  const post = postByKey.get(review.key);
  const x = xs.get(officialPostId(review.key));
  if (!post || !x) {
    return [];
  }
  const song = songByIsrc.get(review.isrc);
  const year = parseIsrcYear(review.isrc);
  const title = song?.title ?? "";
  const artistName = song?.artistName ?? "";
  const item: DiagnosticItem = {
    artistKey: artistKeyOf(review.isrc),
    decade: year !== null ? decadeOfYear(year) : null,
    aspectPrimary: review.aspectPrimary,
    aspects: review.aspects,
    moodPrimary: review.moodPrimary,
    moods: review.moods,
    contentLength: post.content.length,
    mentionsArtist: artistName.length > 0 && post.content.includes(artistName),
    mentionsTitle: title.length > 0 && post.content.includes(title),
  };
  return [{ ...item, x }];
});

console.log(
  `D1=${storeName} / 非評価学習=${trainItems.length} 件 / 評価=${labeled.length} 件 / 言葉クエリ=${rawEval.queries.length} 件`,
);

// 非評価 2,978 件の平均・共分散
const { mean: meanX, c: covX } = covariance(
  trainItems.map((item) => item.x),
  D,
);
const centerX = (x: Float32Array) => {
  const out = new Float64Array(D);
  for (let j = 0; j < D; j++) {
    out[j] = (x[j] ?? 0) - (meanX[j] ?? 0);
  }
  return out;
};
const trainCentered = trainItems.map((item) => centerX(item.x));
const evalCentered = labeled.map((item) => centerX(item.x));
const queryCentered = queryXs.map((x) => centerX(x));

// -----------------------------------------------------------------------------
// 実験 1: Step 1 単体比較（文字数方向の除去：共分散直交射影 vs Shrinkage-LEACE）
// -----------------------------------------------------------------------------
const trainLens = Float64Array.from(trainItems, (item) => item.contentLength);
const trainLogLens = Float64Array.from(trainItems, (item) =>
  Math.log(Math.max(1, item.contentLength)),
);
const vLen = crossCovarianceScalar(
  trainItems.map((item) => item.x),
  meanX,
  trainLens,
  D,
);
const vLogLen = crossCovarianceScalar(
  trainItems.map((item) => item.x),
  meanX,
  trainLogLens,
  D,
);

const step1Candidates: Array<{ name: string; eraser: Rank1Eraser | null; lambda: number }> = [
  { name: "Baseline (PCA 256)", eraser: null, lambda: 0 },
  { name: "Cov-Ortho (len, λ=0.5)", eraser: fitCovOrthoEraser(vLen), lambda: 0.5 },
  { name: "Cov-Ortho (len, λ=1.0)", eraser: fitCovOrthoEraser(vLen), lambda: 1.0 },
  { name: "Cov-Ortho (logLen, λ=1.0)", eraser: fitCovOrthoEraser(vLogLen), lambda: 1.0 },
  {
    name: "LEACE (len, α=10, λ=1.0)",
    eraser: fitShrinkageLeaceEraser(covX, vLen, D, 10),
    lambda: 1.0,
  },
  {
    name: "LEACE (len, α=100, λ=1.0)",
    eraser: fitShrinkageLeaceEraser(covX, vLen, D, 100),
    lambda: 1.0,
  },
  {
    name: "LEACE (len, α=1000, λ=1.0)",
    eraser: fitShrinkageLeaceEraser(covX, vLen, D, 1000),
    lambda: 1.0,
  },
];

const step1Results: Array<{
  条件: string;
  "aspect@3": string;
  "mood@3": string;
  "aspect@10": string;
  "mood@10": string;
  "言葉検索@10": string;
  "文字数相関@3": string;
  raw: {
    aspectAt3: number;
    moodAt3: number;
    aspectAt10: number;
    moodAt10: number;
    textAt10: number;
    lengthCorrAt3: number;
  };
}> = [];

type Step1Selection = {
  name: string;
  eraser: Rank1Eraser | null;
  lambda: number;
  score: number;
};

let bestStep1: Step1Selection = {
  name: "Baseline (PCA 256)",
  eraser: null,
  lambda: 0,
  score: -Infinity,
};

for (const cand of step1Candidates) {
  const eraser = cand.eraser;
  const trClean = eraser
    ? trainCentered.map((x) => applyEraser(x, eraser, cand.lambda))
    : trainCentered;
  const evClean = eraser
    ? evalCentered.map((x) => applyEraser(x, eraser, cand.lambda))
    : evalCentered;
  const qClean = eraser
    ? queryCentered.map((x) => applyEraser(x, eraser, cand.lambda))
    : queryCentered;

  const { c: covClean } = covariance(
    trClean.map((x) => Float32Array.from(x)),
    D,
  );
  const { vectors: pcaBasis } = topEigen(covClean, D, K);
  const evZs = evClean.map((x) => projectRows(x, pcaBasis, K, D));
  const qZs = qClean.map((x) => projectRows(x, pcaBasis, K, D));

  const leak = evaluateLeakage(labeled, evZs);
  const textAt10 = evalTextQueries(labeled, evZs, rawEval.queries, qZs);
  const combined = leak.aspectAt3 + leak.moodAt3;
  if (combined > bestStep1.score) {
    bestStep1 = { name: cand.name, eraser: cand.eraser, lambda: cand.lambda, score: combined };
  }
  step1Results.push({
    条件: cand.name,
    "aspect@3": pct(leak.aspectAt3),
    "mood@3": pct(leak.moodAt3),
    "aspect@10": pct(leak.aspectAt10),
    "mood@10": pct(leak.moodAt10),
    "言葉検索@10": pct(textAt10),
    "文字数相関@3": leak.lengthCorrAt3.toFixed(2),
    raw: {
      aspectAt3: leak.aspectAt3,
      moodAt3: leak.moodAt3,
      aspectAt10: leak.aspectAt10,
      moodAt10: leak.moodAt10,
      textAt10,
      lengthCorrAt3: leak.lengthCorrAt3,
    },
  });
}

console.log("\n=== 実験 1: Step 1 単体比較（文字数方向の除去 → PCA 256） ===");
console.table(step1Results.map(({ raw: _, ...rest }) => rest));
console.log(`Step 1 最良条件: ${bestStep1.name}`);

// -----------------------------------------------------------------------------
// 実験 2: Step 2 比較（5-fold CV での教師あり直交ブロック射影）
// -----------------------------------------------------------------------------
const aspectVocab = [...new Set(labeled.flatMap((r) => r.aspects))].sort();
const moodVocab = [...new Set(labeled.flatMap((r) => r.moods))].sort();
const aspDim = aspectVocab.length;
const moodDim = moodVocab.length;
const supDim = aspDim + moodDim;
const resDim = K - supDim;

const yAspectAll = labeled.map((r) =>
  Float64Array.from(aspectVocab, (label) => (r.aspects.includes(label) ? 1 : 0)),
);
const yMoodAll = labeled.map((r) =>
  Float64Array.from(moodVocab, (label) => (r.moods.includes(label) ? 1 : 0)),
);

const allIdx = Array.from({ length: labeled.length }, (_, i) => i);

// Step 2 を「Step 1 なし」と「Step 1 最良」の両方で比較するため、共通評価関数を定義
type Step2Method = "ridge-concat" | "pls-svd" | "lowdin-scaled";
type WeightPreset = { label: string; wAsp: number; wMood: number; wRes: number };

const weightPresets: Array<WeightPreset> = [
  {
    label: "asp35/mood35/res30",
    wAsp: Math.sqrt(0.35),
    wMood: Math.sqrt(0.35),
    wRes: Math.sqrt(0.3),
  },
  {
    label: "asp25/mood25/res50",
    wAsp: Math.sqrt(0.25),
    wMood: Math.sqrt(0.25),
    wRes: Math.sqrt(0.5),
  },
  {
    label: "asp45/mood45/res10",
    wAsp: Math.sqrt(0.45),
    wMood: Math.sqrt(0.45),
    wRes: Math.sqrt(0.1),
  },
  { label: "aspect単独(100/0/0)", wAsp: 1, wMood: 0, wRes: 0 },
  { label: "mood単独(0/100/0)", wAsp: 0, wMood: 1, wRes: 0 },
  { label: "residual単独(0/0/100)", wAsp: 0, wMood: 0, wRes: 1 },
];

function runStep2Cv(
  trClean: Array<Float64Array>,
  evClean: Array<Float64Array>,
  qClean: Array<Float64Array>,
  method: Step2Method,
  ridgeAlpha: number,
) {
  const { c: covClean } = covariance(
    trClean.map((x) => Float32Array.from(x)),
    D,
  );

  // 各 fold でモデルを 1 回だけ学習し、全重みプリセットを一括評価する
  const accFull = new Map<
    string,
    { a3: number; m3: number; a10: number; m10: number; len: number; text: number }
  >();
  const accValOnly = new Map<string, { a3: number; m3: number }>();
  for (const preset of weightPresets) {
    accFull.set(preset.label, { a3: 0, m3: 0, a10: 0, m10: 0, len: 0, text: 0 });
    accValOnly.set(preset.label, { a3: 0, m3: 0 });
  }

  for (let fold = 0; fold < FOLDS; fold++) {
    const trainIdx: Array<number> = [];
    const valIdx: Array<number> = [];
    for (let i = 0; i < labeled.length; i++) {
      if (i % FOLDS === fold) {
        valIdx.push(i);
      } else {
        trainIdx.push(i);
      }
    }

    const foldX = trainIdx.flatMap((i) => {
      const x = evClean[i];
      return x ? [x] : [];
    });
    const foldYAsp = trainIdx.flatMap((i) => {
      const y = yAspectAll[i];
      return y ? [y] : [];
    });
    const foldYMood = trainIdx.flatMap((i) => {
      const y = yMoodAll[i];
      return y ? [y] : [];
    });

    let uAsp: Float64Array;
    let uMood: Float64Array;
    if (method === "pls-svd") {
      uAsp = fitPlsBlock(foldX, foldYAsp, D, 1);
      uMood = fitPlsBlock(foldX, foldYMood, D, 1);
    } else {
      const rawAsp = dualRidgeMultiFit(foldX, foldYAsp, D, ridgeAlpha);
      const rawMood = dualRidgeMultiFit(foldX, foldYMood, D, ridgeAlpha);
      if (method === "lowdin-scaled") {
        const combined = new Float64Array(supDim * D);
        combined.set(rawAsp, 0);
        combined.set(rawMood, aspDim * D);
        const ortho = symmetricOrthogonalizeScaled(combined, supDim, D);
        uAsp = ortho.subarray(0, aspDim * D);
        uMood = ortho.subarray(aspDim * D, supDim * D);
      } else {
        uAsp = rawAsp;
        uMood = rawMood;
      }
    }

    const supCombined = new Float64Array(supDim * D);
    supCombined.set(uAsp, 0);
    supCombined.set(uMood, aspDim * D);
    const uRes = fitResidualPca(covClean, supCombined, supDim, D, resDim);

    const sAsp = blockRmsNorm(trClean, uAsp, aspDim, D);
    const sMood = blockRmsNorm(trClean, uMood, moodDim, D);
    const sRes = blockRmsNorm(trClean, uRes, resDim, D);

    const evAsp = evClean.map((x) => projectRows(x, uAsp, aspDim, D));
    const evMood = evClean.map((x) => projectRows(x, uMood, moodDim, D));
    const evRes = evClean.map((x) => projectRows(x, uRes, resDim, D));

    const qAsp = qClean.map((x) => projectRows(x, uAsp, aspDim, D));
    const qMood = qClean.map((x) => projectRows(x, uMood, moodDim, D));
    const qRes = qClean.map((x) => projectRows(x, uRes, resDim, D));

    const compose = (a: Float64Array, m: Float64Array, r: Float64Array, preset: WeightPreset) => {
      const z = new Float64Array(K);
      const ca = preset.wAsp / sAsp;
      const cm = preset.wMood / sMood;
      const cr = preset.wRes / sRes;
      for (let j = 0; j < aspDim; j++) {
        z[j] = (a[j] ?? 0) * ca;
      }
      for (let j = 0; j < moodDim; j++) {
        z[aspDim + j] = (m[j] ?? 0) * cm;
      }
      for (let j = 0; j < resDim; j++) {
        z[supDim + j] = (r[j] ?? 0) * cr;
      }
      return z;
    };

    for (const preset of weightPresets) {
      const evZs = evClean.map((_, i) =>
        compose(
          evAsp[i] ?? new Float64Array(aspDim),
          evMood[i] ?? new Float64Array(moodDim),
          evRes[i] ?? new Float64Array(resDim),
          preset,
        ),
      );
      const qZs = qClean.map((_, i) =>
        compose(
          qAsp[i] ?? new Float64Array(aspDim),
          qMood[i] ?? new Float64Array(moodDim),
          qRes[i] ?? new Float64Array(resDim),
          preset,
        ),
      );

      const full = evalSubsetRetrieval(labeled, evZs, valIdx, allIdx);
      const valOnly = evalSubsetRetrieval(labeled, evZs, valIdx, valIdx);
      const text = evalTextQueries(labeled, evZs, evalData.queries, qZs);

      const fAcc = accFull.get(preset.label);
      const vAcc = accValOnly.get(preset.label);
      if (fAcc && vAcc) {
        fAcc.a3 += full.aspectAt3 / FOLDS;
        fAcc.m3 += full.moodAt3 / FOLDS;
        fAcc.a10 += full.aspectAt10 / FOLDS;
        fAcc.m10 += full.moodAt10 / FOLDS;
        fAcc.len += full.lengthCorrAt3 / FOLDS;
        fAcc.text += text / FOLDS;
        vAcc.a3 += valOnly.aspectAt3 / FOLDS;
        vAcc.m3 += valOnly.moodAt3 / FOLDS;
      }
    }
  }

  return weightPresets.map((preset) => {
    const f = accFull.get(preset.label) ?? { a3: 0, m3: 0, a10: 0, m10: 0, len: 0, text: 0 };
    const v = accValOnly.get(preset.label) ?? { a3: 0, m3: 0 };
    return {
      preset: preset.label,
      full: f,
      valOnly: v,
    };
  });
}

// Baseline の val-only（140 vs 139）スコアも比較用に計算
const { vectors: basePcaBasis } = topEigen(covX, D, K);
const baseEvalZs = evalCentered.map((x) => projectRows(x, basePcaBasis, K, D));
let baseValA3 = 0;
let baseValM3 = 0;
for (let fold = 0; fold < FOLDS; fold++) {
  const valIdx = allIdx.filter((i) => i % FOLDS === fold);
  const res = evalSubsetRetrieval(labeled, baseEvalZs, valIdx, valIdx);
  baseValA3 += res.aspectAt3 / FOLDS;
  baseValM3 += res.moodAt3 / FOLDS;
}

const bestEraser = bestStep1.eraser;
const trBest = bestEraser
  ? trainCentered.map((x) => applyEraser(x, bestEraser, bestStep1.lambda))
  : trainCentered;
const evBest = bestEraser
  ? evalCentered.map((x) => applyEraser(x, bestEraser, bestStep1.lambda))
  : evalCentered;
const qBest = bestEraser
  ? queryCentered.map((x) => applyEraser(x, bestEraser, bestStep1.lambda))
  : queryCentered;

const step2Configs: Array<{
  label: string;
  tr: Array<Float64Array>;
  ev: Array<Float64Array>;
  q: Array<Float64Array>;
  method: Step2Method;
  alpha: number;
}> = [
  {
    label: "Ridge連結 (α=500, Step1なし)",
    tr: trainCentered,
    ev: evalCentered,
    q: queryCentered,
    method: "ridge-concat",
    alpha: 500,
  },
  {
    label: "Ridge連結 (α=2000, Step1なし)",
    tr: trainCentered,
    ev: evalCentered,
    q: queryCentered,
    method: "ridge-concat",
    alpha: 2000,
  },
  {
    label: "PLS-SVD (Step1なし)",
    tr: trainCentered,
    ev: evalCentered,
    q: queryCentered,
    method: "pls-svd",
    alpha: 0,
  },
  {
    label: "Löwdin直交+スケーリング (α=2000, Step1なし)",
    tr: trainCentered,
    ev: evalCentered,
    q: queryCentered,
    method: "lowdin-scaled",
    alpha: 2000,
  },
  {
    label: "Step1最良 + Ridge連結 (α=2000)",
    tr: trBest,
    ev: evBest,
    q: qBest,
    method: "ridge-concat",
    alpha: 2000,
  },
  {
    label: "Step1最良 + Löwdin直交 (α=2000)",
    tr: trBest,
    ev: evBest,
    q: qBest,
    method: "lowdin-scaled",
    alpha: 2000,
  },
];

const step2AllResults: Array<{
  config: string;
  rows: ReturnType<typeof runStep2Cv>;
}> = [];

console.log(
  `\n=== 実験 2: Step 2 比較（5-fold CV、Baseline val-only: aspect@3=${pct(baseValA3)}, mood@3=${pct(baseValM3)}） ===`,
);

for (const cfg of step2Configs) {
  const rows = runStep2Cv(cfg.tr, cfg.ev, cfg.q, cfg.method, cfg.alpha);
  step2AllResults.push({ config: cfg.label, rows });
  console.log(`\n--- ${cfg.label} ---`);
  console.table(
    rows.map((r) => ({
      重み比率: r.preset,
      "aspect@3(全)": pct(r.full.a3),
      "mood@3(全)": pct(r.full.m3),
      "aspect@3(未見)": pct(r.valOnly.a3),
      "mood@3(未見)": pct(r.valOnly.m3),
      "言葉検索@10": pct(r.full.text),
      "文字数相関@3": r.full.len.toFixed(2),
    })),
  );
}

console.log(
  JSON.stringify({
    storeName,
    trainCount: trainItems.length,
    evalCount: labeled.length,
    step1: step1Results.map(({ 条件, raw }) => ({ condition: 条件, ...raw })),
    bestStep1: bestStep1.name,
    baselineValOnly: { aspectAt3: baseValA3, moodAt3: baseValM3 },
    step2: step2AllResults,
  }),
);
