import { cholesky } from "../../src/embeddings/linear-algebra.ts";
import { topEigen } from "./pca.ts";

/** ISRC の 6〜7 桁目（録音・登録年 2 桁）を西暦 4 桁に変換する。 */
export function parseIsrcYear(isrc: string): number | null {
  if (isrc.length < 7) {
    return null;
  }
  const yy = Number(isrc.slice(5, 7));
  if (!Number.isInteger(yy) || yy < 0 || yy > 99) {
    return null;
  }
  return yy >= 50 ? 1900 + yy : 2000 + yy;
}

/** 西暦から年代ラベル（例: "1990s"）を作る。 */
export function decadeOfYear(year: number): string {
  return `${Math.floor(year / 10) * 10}s`;
}

/** コレスキー因子 L（下三角、n × n）を使って L Lᵀ x = b を解く。 */
function solveCholesky(l: Float64Array, b: Float64Array, n: number): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i] ?? 0;
    const row = i * n;
    for (let j = 0; j < i; j++) {
      sum -= (l[row + j] ?? 0) * (y[j] ?? 0);
    }
    y[i] = sum / (l[row + i] ?? 1);
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i] ?? 0;
    for (let j = i + 1; j < n; j++) {
      sum -= (l[j * n + i] ?? 0) * (x[j] ?? 0);
    }
    x[i] = sum / (l[i * n + i] ?? 1);
  }
  return x;
}

/**
 * 中心化付きリッジ回帰 min_w ‖(X - μ_x)w - (y - μ_y)‖² + alpha ‖w‖²。
 * 特徴次元 d が数百程度（PCA 256 次元など）を想定する。
 */
export function ridgeFit(xs: Array<Float64Array>, ys: Float64Array, alpha: number) {
  const n = xs.length;
  const d = xs[0]?.length ?? 0;
  if (n === 0 || d === 0) {
    return { w: new Float64Array(d), bias: 0 };
  }

  const meanX = new Float64Array(d);
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    const row = xs[i];
    if (!row) {
      continue;
    }
    for (let j = 0; j < d; j++) {
      meanX[j] = (meanX[j] ?? 0) + (row[j] ?? 0) / n;
    }
    meanY += (ys[i] ?? 0) / n;
  }

  const xtx = new Float64Array(d * d);
  const xty = new Float64Array(d);
  const xc = new Float64Array(d);

  for (let i = 0; i < n; i++) {
    const row = xs[i];
    if (!row) {
      continue;
    }
    const yc = (ys[i] ?? 0) - meanY;
    for (let j = 0; j < d; j++) {
      xc[j] = (row[j] ?? 0) - (meanX[j] ?? 0);
      xty[j] = (xty[j] ?? 0) + (xc[j] ?? 0) * yc;
    }
    for (let a = 0; a < d; a++) {
      const xa = xc[a] ?? 0;
      const base = a * d;
      for (let b = a; b < d; b++) {
        xtx[base + b] = (xtx[base + b] ?? 0) + xa * (xc[b] ?? 0);
      }
    }
  }

  const reg = Math.max(alpha, 1e-8);
  for (let a = 0; a < d; a++) {
    xtx[a * d + a] = (xtx[a * d + a] ?? 0) + reg;
    for (let b = a + 1; b < d; b++) {
      xtx[b * d + a] = xtx[a * d + b] ?? 0;
    }
  }

  const l = cholesky(xtx, d);
  if (!l) {
    return { w: new Float64Array(d), bias: meanY };
  }
  const w = solveCholesky(l, xty, d);
  let wDotMeanX = 0;
  for (let j = 0; j < d; j++) {
    wDotMeanX += (w[j] ?? 0) * (meanX[j] ?? 0);
  }
  return { w, bias: meanY - wDotMeanX };
}

/** 学習済みリッジ回帰で予測値を返す。 */
export function ridgePredict(model: { w: Float64Array; bias: number }, x: Float64Array): number {
  let acc = model.bias;
  for (let j = 0; j < model.w.length; j++) {
    acc += (model.w[j] ?? 0) * (x[j] ?? 0);
  }
  return acc;
}

/**
 * 決定論的 K-fold 交差検証でリッジ回帰の R² を計算し、最良の alpha と R² を返す。
 * fold 割り当ては index % folds（入力がランダム順または独立な投稿列であることを前提とする）。
 */
export function ridgeCvR2(
  xs: Array<Float64Array>,
  ys: Float64Array,
  alphas: Array<number>,
  folds = 5,
) {
  const n = xs.length;
  if (n < folds) {
    return { bestAlpha: alphas[0] ?? 1, r2: 0 };
  }

  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanY += (ys[i] ?? 0) / n;
  }
  let sst = 0;
  for (let i = 0; i < n; i++) {
    const diff = (ys[i] ?? 0) - meanY;
    sst += diff * diff;
  }
  if (sst === 0) {
    return { bestAlpha: alphas[0] ?? 1, r2: 0 };
  }

  let bestAlpha = alphas[0] ?? 1;
  let bestR2 = -Infinity;

  for (const alpha of alphas) {
    let sse = 0;
    for (let fold = 0; fold < folds; fold++) {
      const trainX: Array<Float64Array> = [];
      const trainYList: Array<number> = [];
      const valIdx: Array<number> = [];
      for (let i = 0; i < n; i++) {
        if (i % folds === fold) {
          valIdx.push(i);
        } else {
          const row = xs[i];
          if (row) {
            trainX.push(row);
            trainYList.push(ys[i] ?? 0);
          }
        }
      }
      const model = ridgeFit(trainX, Float64Array.from(trainYList), alpha);
      for (const idx of valIdx) {
        const row = xs[idx];
        if (!row) {
          continue;
        }
        const pred = ridgePredict(model, row);
        const err = (ys[idx] ?? 0) - pred;
        sse += err * err;
      }
    }
    const r2 = 1 - sse / sst;
    if (r2 > bestR2) {
      bestR2 = r2;
      bestAlpha = alpha;
    }
  }

  return { bestAlpha, r2: bestR2 };
}

/**
 * ターゲット変数 y から、内容特徴 features（aspect/mood の multi-hot 等）で線形説明できる成分を除いた残差を返す。
 * 例: リリース年から「era / nostalgic などの内容ラベルで説明できる成分」を引いた純粋な年代残差を得る。
 */
export function residualizeByFeatures(
  targets: Float64Array,
  features: Array<Float64Array>,
  alpha = 1,
): Float64Array {
  const model = ridgeFit(features, targets, alpha);
  const out = new Float64Array(targets.length);
  for (let i = 0; i < targets.length; i++) {
    const f = features[i];
    out[i] = (targets[i] ?? 0) - (f ? ridgePredict(model, f) : 0);
  }
  return out;
}

/**
 * 学習データから「アーティスト間重心（between-artist）」の共分散行列を作り、
 * 上位 rank 方向の正規直交基底（rank × dim、行優先）を返す。
 * 投稿数が 2 件以上あるアーティストのみを重心計算に使う（1 件だけのアーティストは曲内ノイズと区別できないため）。
 */
export function estimateArtistSubspace(
  items: Array<{ artistKey: string; z: Float64Array }>,
  dim: number,
  rank: number,
): Float64Array {
  const groups = new Map<string, Array<Float64Array>>();
  for (const item of items) {
    const list = groups.get(item.artistKey);
    if (list) {
      list.push(item.z);
    } else {
      groups.set(item.artistKey, [item.z]);
    }
  }

  const centroids: Array<{ mean: Float64Array; weight: number }> = [];
  let totalWeight = 0;
  const globalMean = new Float64Array(dim);

  for (const list of groups.values()) {
    if (list.length < 2) {
      continue;
    }
    const mean = new Float64Array(dim);
    for (const v of list) {
      for (let j = 0; j < dim; j++) {
        mean[j] = (mean[j] ?? 0) + (v[j] ?? 0) / list.length;
      }
    }
    centroids.push({ mean, weight: list.length });
    totalWeight += list.length;
    for (let j = 0; j < dim; j++) {
      globalMean[j] = (globalMean[j] ?? 0) + (mean[j] ?? 0) * list.length;
    }
  }

  if (centroids.length === 0 || totalWeight === 0) {
    return new Float64Array(rank * dim);
  }
  for (let j = 0; j < dim; j++) {
    globalMean[j] = (globalMean[j] ?? 0) / totalWeight;
  }

  const cov = new Float64Array(dim * dim);
  const diff = new Float64Array(dim);
  for (const { mean, weight } of centroids) {
    const w = weight / totalWeight;
    for (let j = 0; j < dim; j++) {
      diff[j] = (mean[j] ?? 0) - (globalMean[j] ?? 0);
    }
    for (let a = 0; a < dim; a++) {
      const da = (diff[a] ?? 0) * w;
      const base = a * dim;
      for (let b = a; b < dim; b++) {
        cov[base + b] = (cov[base + b] ?? 0) + da * (diff[b] ?? 0);
      }
    }
  }
  for (let a = 0; a < dim; a++) {
    for (let b = a + 1; b < dim; b++) {
      cov[b * dim + a] = cov[a * dim + b] ?? 0;
    }
  }

  const actualRank = Math.min(rank, dim, Math.max(1, centroids.length - 1));
  const { values, vectors } = topEigen(cov, dim, actualRank);

  // 低ランク共分散で数値誤差が出ても必ず正規直交基底になるよう、有効な固有値のみ Gram-Schmidt 正規直交化する。
  const kept: Array<Float64Array> = [];
  for (let r = 0; r < actualRank; r++) {
    if ((values[r] ?? 0) <= 1e-12) {
      continue;
    }
    const v = Float64Array.from(vectors.subarray(r * dim, (r + 1) * dim));
    for (const prev of kept) {
      let dot = 0;
      for (let j = 0; j < dim; j++) {
        dot += (v[j] ?? 0) * (prev[j] ?? 0);
      }
      for (let j = 0; j < dim; j++) {
        v[j] = (v[j] ?? 0) - dot * (prev[j] ?? 0);
      }
    }
    let sq = 0;
    for (let j = 0; j < dim; j++) {
      const val = v[j] ?? 0;
      sq += val * val;
    }
    const norm = Math.sqrt(sq);
    if (norm > 1e-12) {
      for (let j = 0; j < dim; j++) {
        v[j] = (v[j] ?? 0) / norm;
      }
      kept.push(v);
    }
  }

  const out = new Float64Array(kept.length * dim);
  for (let r = 0; r < kept.length; r++) {
    const row = kept[r];
    if (row) {
      out.set(row, r * dim);
    }
  }
  return out;
}

/**
 * 正規直交基底 basis（rank × dim、行優先）が張る部分空間の成分を係数 lambda で減衰する。
 * z' = z - lambda * Σ_r ⟨u_r, z⟩ u_r
 */
export function attenuateSubspace(
  z: Float64Array,
  basis: Float64Array,
  rank: number,
  lambda: number,
): Float64Array {
  const dim = z.length;
  const out = Float64Array.from(z);
  if (lambda === 0 || rank === 0) {
    return out;
  }
  for (let r = 0; r < rank; r++) {
    const base = r * dim;
    let dot = 0;
    for (let j = 0; j < dim; j++) {
      dot += (basis[base + j] ?? 0) * (z[j] ?? 0);
    }
    const scale = lambda * dot;
    for (let j = 0; j < dim; j++) {
      out[j] = (out[j] ?? 0) - scale * (basis[base + j] ?? 0);
    }
  }
  return out;
}

/** 単位方向ベクトル u に沿った成分を係数 lambda で減衰する。 */
export function attenuateDirection(
  z: Float64Array,
  unitDir: Float64Array,
  lambda: number,
): Float64Array {
  const out = Float64Array.from(z);
  if (lambda === 0) {
    return out;
  }
  let dot = 0;
  for (let j = 0; j < z.length; j++) {
    dot += (unitDir[j] ?? 0) * (z[j] ?? 0);
  }
  const scale = lambda * dot;
  for (let j = 0; j < z.length; j++) {
    out[j] = (out[j] ?? 0) - scale * (unitDir[j] ?? 0);
  }
  return out;
}

export type DiagnosticItem = {
  artistKey: string;
  decade: string | null;
  aspectPrimary: string;
  aspects: Array<string>;
  moodPrimary: string;
  moods: Array<string>;
  contentLength: number;
  mentionsArtist: boolean;
  mentionsTitle: boolean;
};

export type LeakageSummary = {
  aspectAt3: number;
  moodAt3: number;
  aspectAt10: number;
  moodAt10: number;
  sameArtistAt3: number;
  sameArtistAt10: number;
  sameArtistNonArtistAspectAt3: number;
  sameArtistNonArtistAspectAt10: number;
  sameArtistWhenMentionedAt10: number;
  sameArtistWhenNotMentionedAt10: number;
  sameDecadeAllAt10: number;
  sameDecadeNonEraAt3: number;
  sameDecadeNonEraAt10: number;
  lengthCorrAt3: number;
};

function normalizeVector(v: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(v);
  let sq = 0;
  for (let i = 0; i < out.length; i++) {
    const x = out[i] ?? 0;
    sq += x * x;
  }
  const norm = Math.sqrt(sq) || 1;
  for (let i = 0; i < out.length; i++) {
    out[i] = (out[i] ?? 0) / norm;
  }
  return out;
}

function safeDiv(sum: number, count: number): number {
  return count > 0 ? sum / count : 0;
}

function pearson(xs: Array<number>, ys: Array<number>): number {
  const n = xs.length;
  if (n < 2) {
    return 0;
  }
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += (xs[i] ?? 0) / n;
    my += (ys[i] ?? 0) / n;
  }
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = (xs[i] ?? 0) - mx;
    const b = (ys[i] ?? 0) - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const denom = Math.sqrt(dx * dy);
  return denom > 0 ? num / denom : 0;
}

/**
 * 評価セット上の類似検索（各投稿を起点に自分以外をコサイン降順で並べる）について、
 * 内容一致度（aspect/mood）と不要因子の漏れ率（同一アーティスト・同一年代・文字数相関）を計測する。
 */
export function evaluateLeakage(
  items: Array<DiagnosticItem>,
  vectors: Array<ArrayLike<number>>,
): LeakageSummary {
  const n = items.length;
  const vs = vectors.map(normalizeVector);

  // 評価セット内に同一アーティストの別投稿が 1 件以上あるか
  const artistFreq = new Map<string, number>();
  for (const item of items) {
    artistFreq.set(item.artistKey, (artistFreq.get(item.artistKey) ?? 0) + 1);
  }

  let aspect3 = 0;
  let mood3 = 0;
  let aspect10 = 0;
  let mood10 = 0;

  let sameArtist3 = 0;
  let sameArtist10 = 0;
  let sameArtistCount = 0;

  let sameArtistNonArt3 = 0;
  let sameArtistNonArt10 = 0;
  let sameArtistNonArtCount = 0;

  let sameArtistMention10 = 0;
  let sameArtistMentionCount = 0;
  let sameArtistNoMention10 = 0;
  let sameArtistNoMentionCount = 0;

  let sameDecadeAll10 = 0;
  let sameDecadeAllCount = 0;

  let sameDecadeNonEra3 = 0;
  let sameDecadeNonEra10 = 0;
  let sameDecadeNonEraCount = 0;

  const queryLengths: Array<number> = [];
  const top3MeanLengths: Array<number> = [];

  for (let i = 0; i < n; i++) {
    const qi = items[i];
    const vi = vs[i];
    if (!qi || !vi) {
      continue;
    }

    const scored: Array<[number, number]> = [];
    for (let j = 0; j < n; j++) {
      if (j === i) {
        continue;
      }
      const vj = vs[j];
      if (!vj) {
        continue;
      }
      let dot = 0;
      for (let t = 0; t < vi.length; t++) {
        dot += (vi[t] ?? 0) * (vj[t] ?? 0);
      }
      scored.push([dot, j]);
    }
    scored.sort((a, b) => b[0] - a[0]);

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

    aspect3 += top3.filter((r) => r.aspectPrimary === qi.aspectPrimary).length / k3;
    mood3 += top3.filter((r) => r.moodPrimary === qi.moodPrimary).length / k3;
    aspect10 += top10.filter((r) => r.aspectPrimary === qi.aspectPrimary).length / k10;
    mood10 += top10.filter((r) => r.moodPrimary === qi.moodPrimary).length / k10;

    queryLengths.push(qi.contentLength);
    top3MeanLengths.push(top3.reduce((s, r) => s + r.contentLength, 0) / k3);

    if ((artistFreq.get(qi.artistKey) ?? 0) >= 2) {
      const a3 = top3.filter((r) => r.artistKey === qi.artistKey).length / k3;
      const a10 = top10.filter((r) => r.artistKey === qi.artistKey).length / k10;
      sameArtist3 += a3;
      sameArtist10 += a10;
      sameArtistCount++;

      if (qi.aspectPrimary !== "artist") {
        sameArtistNonArt3 += a3;
        sameArtistNonArt10 += a10;
        sameArtistNonArtCount++;
      }
      if (qi.mentionsArtist) {
        sameArtistMention10 += a10;
        sameArtistMentionCount++;
      } else {
        sameArtistNoMention10 += a10;
        sameArtistNoMentionCount++;
      }
    }

    if (qi.decade) {
      const d3 = top3.filter((r) => r.decade === qi.decade).length / k3;
      const d10 = top10.filter((r) => r.decade === qi.decade).length / k10;
      sameDecadeAll10 += d10;
      sameDecadeAllCount++;

      if (!qi.aspects.includes("era") && !qi.moods.includes("nostalgic")) {
        sameDecadeNonEra3 += d3;
        sameDecadeNonEra10 += d10;
        sameDecadeNonEraCount++;
      }
    }
  }

  return {
    aspectAt3: safeDiv(aspect3, n),
    moodAt3: safeDiv(mood3, n),
    aspectAt10: safeDiv(aspect10, n),
    moodAt10: safeDiv(mood10, n),
    sameArtistAt3: safeDiv(sameArtist3, sameArtistCount),
    sameArtistAt10: safeDiv(sameArtist10, sameArtistCount),
    sameArtistNonArtistAspectAt3: safeDiv(sameArtistNonArt3, sameArtistNonArtCount),
    sameArtistNonArtistAspectAt10: safeDiv(sameArtistNonArt10, sameArtistNonArtCount),
    sameArtistWhenMentionedAt10: safeDiv(sameArtistMention10, sameArtistMentionCount),
    sameArtistWhenNotMentionedAt10: safeDiv(sameArtistNoMention10, sameArtistNoMentionCount),
    sameDecadeAllAt10: safeDiv(sameDecadeAll10, sameDecadeAllCount),
    sameDecadeNonEraAt3: safeDiv(sameDecadeNonEra3, sameDecadeNonEraCount),
    sameDecadeNonEraAt10: safeDiv(sameDecadeNonEra10, sameDecadeNonEraCount),
    lengthCorrAt3: pearson(queryLengths, top3MeanLengths),
  };
}

/** ランダムに並べたときの期待値（ベースライン）。 */
export function randomLeakageBaseline(items: Array<DiagnosticItem>) {
  const n = items.length;
  if (n < 2) {
    return {
      aspect: 0,
      mood: 0,
      sameArtistEligible: 0,
      sameDecadeAll: 0,
      sameDecadeNonEra: 0,
    };
  }
  const countBy = (fn: (item: DiagnosticItem) => string | null) => {
    const map = new Map<string, number>();
    for (const item of items) {
      const k = fn(item);
      if (k !== null) {
        map.set(k, (map.get(k) ?? 0) + 1);
      }
    }
    return map;
  };
  const artistCounts = countBy((it) => it.artistKey);
  const decadeCounts = countBy((it) => it.decade);
  const aspectCounts = countBy((it) => it.aspectPrimary);
  const moodCounts = countBy((it) => it.moodPrimary);

  let aspect = 0;
  let mood = 0;
  let sameArtistSum = 0;
  let sameArtistCount = 0;
  let sameDecadeAllSum = 0;
  let sameDecadeAllCount = 0;
  let sameDecadeNonEraSum = 0;
  let sameDecadeNonEraCount = 0;

  for (const item of items) {
    aspect += ((aspectCounts.get(item.aspectPrimary) ?? 1) - 1) / (n - 1);
    mood += ((moodCounts.get(item.moodPrimary) ?? 1) - 1) / (n - 1);

    const ac = artistCounts.get(item.artistKey) ?? 1;
    if (ac >= 2) {
      sameArtistSum += (ac - 1) / (n - 1);
      sameArtistCount++;
    }
    if (item.decade) {
      const dc = decadeCounts.get(item.decade) ?? 1;
      const p = (dc - 1) / (n - 1);
      sameDecadeAllSum += p;
      sameDecadeAllCount++;
      if (!item.aspects.includes("era") && !item.moods.includes("nostalgic")) {
        sameDecadeNonEraSum += p;
        sameDecadeNonEraCount++;
      }
    }
  }

  return {
    aspect: aspect / n,
    mood: mood / n,
    sameArtistEligible: sameArtistCount > 0 ? sameArtistSum / sameArtistCount : 0,
    sameDecadeAll: sameDecadeAllCount > 0 ? sameDecadeAllSum / sameDecadeAllCount : 0,
    sameDecadeNonEra: sameDecadeNonEraCount > 0 ? sameDecadeNonEraSum / sameDecadeNonEraCount : 0,
  };
}
