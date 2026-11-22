import { cholesky } from "../../src/embeddings/linear-algebra.ts";
import { topEigen } from "./pca.ts";

/** コレスキー因子 L（下三角、n × n）を使って L Lᵀ x = b を解く。 */
function solveCholeskyVec(l: Float64Array, b: Float64Array, n: number): Float64Array {
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

/** 対称行列（m × m）のヤコビ法固有分解。固有値の降順に返す（vectors は列優先）。 */
function jacobiSym(matrix: Float64Array, m: number) {
  const a = Float64Array.from(matrix);
  const v = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    v[i * m + i] = 1;
  }
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) {
        const val = a[i * m + j] ?? 0;
        off += val * val;
      }
    }
    if (off < 1e-24) {
      break;
    }
    for (let p = 0; p < m; p++) {
      for (let q = p + 1; q < m; q++) {
        const apq = a[p * m + q] ?? 0;
        if (Math.abs(apq) < 1e-30) {
          continue;
        }
        const theta = ((a[q * m + q] ?? 0) - (a[p * m + p] ?? 0)) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cos = 1 / Math.sqrt(t * t + 1);
        const sin = t * cos;
        for (let k = 0; k < m; k++) {
          const akp = a[k * m + p] ?? 0;
          const akq = a[k * m + q] ?? 0;
          a[k * m + p] = cos * akp - sin * akq;
          a[k * m + q] = sin * akp + cos * akq;
        }
        for (let k = 0; k < m; k++) {
          const apk = a[p * m + k] ?? 0;
          const aqk = a[q * m + k] ?? 0;
          a[p * m + k] = cos * apk - sin * aqk;
          a[q * m + k] = sin * apk + cos * aqk;
        }
        for (let k = 0; k < m; k++) {
          const vkp = v[k * m + p] ?? 0;
          const vkq = v[k * m + q] ?? 0;
          v[k * m + p] = cos * vkp - sin * vkq;
          v[k * m + q] = sin * vkp + cos * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: m }, (_, i) => i).sort(
    (x, y) => (a[y * m + y] ?? 0) - (a[x * m + x] ?? 0),
  );
  return { values: order.map((i) => a[i * m + i] ?? 0), vectors: v, order };
}

/**
 * Step 1 用：中心化済み x（n × d）とスカラー不要因子 y（長さ n）の共分散ベクトル v = Cov(x, y) を求める。
 */
export function crossCovarianceScalar(
  xs: Array<ArrayLike<number>>,
  meanX: Float64Array,
  ys: Float64Array,
  d: number,
): Float64Array {
  const n = xs.length;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanY += (ys[i] ?? 0) / n;
  }
  const v = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const row = xs[i];
    if (!row) {
      continue;
    }
    const yc = ((ys[i] ?? 0) - meanY) / n;
    for (let j = 0; j < d; j++) {
      v[j] = (v[j] ?? 0) + ((row[j] ?? 0) - (meanX[j] ?? 0)) * yc;
    }
  }
  return v;
}

/**
 * ランク 1 の線形除去作用素 P(x) = x - lambda * (wᵀ x / denom) * v。
 * - 共分散直交射影（α = ∞）：w = v、denom = ‖v‖²
 * - Shrinkage-LEACE（α < ∞）：w = (Σ_XX + αI)⁻¹ v、denom = wᵀ v
 */
export type Rank1Eraser = {
  v: Float64Array;
  w: Float64Array;
  denom: number;
};

/** 共分散方向の直交射影（Shrinkage-LEACE の α → ∞ 極限）。 */
export function fitCovOrthoEraser(v: Float64Array): Rank1Eraser {
  let sq = 0;
  for (let j = 0; j < v.length; j++) {
    const val = v[j] ?? 0;
    sq += val * val;
  }
  return { v: Float64Array.from(v), w: Float64Array.from(v), denom: sq || 1 };
}

/**
 * Shrinkage-LEACE（Belrose et al., NeurIPS 2023 の正則化版）。
 * covX は d × d の共分散行列（行優先）、v は Cov(x, y)（長さ d）。
 */
export function fitShrinkageLeaceEraser(
  covX: Float64Array,
  v: Float64Array,
  d: number,
  alpha: number,
): Rank1Eraser {
  const mat = Float64Array.from(covX);
  const reg = Math.max(alpha, 1e-8);
  for (let i = 0; i < d; i++) {
    mat[i * d + i] = (mat[i * d + i] ?? 0) + reg;
  }
  const l = cholesky(mat, d);
  if (!l) {
    return fitCovOrthoEraser(v);
  }
  const w = solveCholeskyVec(l, v, d);
  let denom = 0;
  for (let j = 0; j < d; j++) {
    denom += (w[j] ?? 0) * (v[j] ?? 0);
  }
  if (Math.abs(denom) < 1e-18) {
    return fitCovOrthoEraser(v);
  }
  return { v: Float64Array.from(v), w, denom };
}

/** 中心化済みベクトル x に Rank1Eraser を強さ lambda で適用する。 */
export function applyEraser(
  x: ArrayLike<number>,
  eraser: Rank1Eraser,
  lambda: number,
): Float64Array {
  const d = eraser.v.length;
  const out = new Float64Array(d);
  if (lambda === 0) {
    for (let j = 0; j < d; j++) {
      out[j] = x[j] ?? 0;
    }
    return out;
  }
  let wDotX = 0;
  for (let j = 0; j < d; j++) {
    wDotX += (eraser.w[j] ?? 0) * (x[j] ?? 0);
  }
  const scale = (lambda * wDotX) / eraser.denom;
  for (let j = 0; j < d; j++) {
    out[j] = (x[j] ?? 0) - scale * (eraser.v[j] ?? 0);
  }
  return out;
}

/**
 * Woodbury 恒等式（双対リッジ回帰）で多出力リッジ回帰 W = (X_cᵀ X_c + αI)⁻¹ X_cᵀ Y_c を高速に解く。
 * n < d（例：n = 560、d = 2048）のとき n × n コレスキー分解 1 回で厳密解が出る。
 * 返り値は m × d（行優先：各行が 1 出力の重みベクトル w_c）。
 */
export function dualRidgeMultiFit(
  xs: Array<Float64Array>,
  ys: Array<Float64Array>,
  d: number,
  alpha: number,
): Float64Array {
  const n = xs.length;
  const m = ys[0]?.length ?? 0;
  if (n === 0 || m === 0) {
    return new Float64Array(m * d);
  }

  const meanX = new Float64Array(d);
  const meanY = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    const xi = xs[i];
    const yi = ys[i];
    if (xi) {
      for (let j = 0; j < d; j++) {
        meanX[j] = (meanX[j] ?? 0) + (xi[j] ?? 0) / n;
      }
    }
    if (yi) {
      for (let c = 0; c < m; c++) {
        meanY[c] = (meanY[c] ?? 0) + (yi[c] ?? 0) / n;
      }
    }
  }

  const xc = xs.map((row) => {
    const out = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      out[j] = (row[j] ?? 0) - (meanX[j] ?? 0);
    }
    return out;
  });

  // Gram 行列 K = X_c X_cᵀ + αI（n × n）
  const gram = new Float64Array(n * n);
  const reg = Math.max(alpha, 1e-8);
  for (let i = 0; i < n; i++) {
    const xi = xc[i];
    if (!xi) {
      continue;
    }
    for (let j = i; j < n; j++) {
      const xj = xc[j];
      if (!xj) {
        continue;
      }
      let dot = 0;
      for (let t = 0; t < d; t++) {
        dot += (xi[t] ?? 0) * (xj[t] ?? 0);
      }
      gram[i * n + j] = dot + (i === j ? reg : 0);
      gram[j * n + i] = dot + (i === j ? reg : 0);
    }
  }

  const l = cholesky(gram, n);
  if (!l) {
    return new Float64Array(m * d);
  }

  // W = X_cᵀ (K⁻¹ Y_c)（行優先 m × d）
  const weights = new Float64Array(m * d);
  const yc = new Float64Array(n);
  for (let c = 0; c < m; c++) {
    for (let i = 0; i < n; i++) {
      yc[i] = (ys[i]?.[c] ?? 0) - (meanY[c] ?? 0);
    }
    const dual = solveCholeskyVec(l, yc, n);
    const base = c * d;
    for (let i = 0; i < n; i++) {
      const a = dual[i] ?? 0;
      const xi = xc[i];
      if (!xi) {
        continue;
      }
      for (let j = 0; j < d; j++) {
        weights[base + j] = (weights[base + j] ?? 0) + a * (xi[j] ?? 0);
      }
    }
  }
  return weights;
}

/**
 * クロス共分散 C = (1/n) X_cᵀ Y_c の SVD（左特異ベクトル × 特異値^power）を求める（PLS ブロック）。
 * 返り値は m × d（行優先）。特異値重みにより弱いノイズ方向が自然に縮む。
 */
export function fitPlsBlock(
  xs: Array<Float64Array>,
  ys: Array<Float64Array>,
  d: number,
  singularPower = 1,
): Float64Array {
  const n = xs.length;
  const m = ys[0]?.length ?? 0;
  if (n === 0 || m === 0) {
    return new Float64Array(m * d);
  }

  const meanX = new Float64Array(d);
  const meanY = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    const xi = xs[i];
    const yi = ys[i];
    if (xi) {
      for (let j = 0; j < d; j++) {
        meanX[j] = (meanX[j] ?? 0) + (xi[j] ?? 0) / n;
      }
    }
    if (yi) {
      for (let c = 0; c < m; c++) {
        meanY[c] = (meanY[c] ?? 0) + (yi[c] ?? 0) / n;
      }
    }
  }

  // C（行優先 m × d：行 c が Cov(x, y_c)）
  const covXY = new Float64Array(m * d);
  for (let i = 0; i < n; i++) {
    const xi = xs[i];
    const yi = ys[i];
    if (!xi || !yi) {
      continue;
    }
    for (let c = 0; c < m; c++) {
      const yc = ((yi[c] ?? 0) - (meanY[c] ?? 0)) / n;
      const base = c * d;
      for (let j = 0; j < d; j++) {
        covXY[base + j] = (covXY[base + j] ?? 0) + ((xi[j] ?? 0) - (meanX[j] ?? 0)) * yc;
      }
    }
  }

  // m × m の Gram 行列 G = C Cᵀ を固有分解して左特異ベクトルを求める
  const gram = new Float64Array(m * m);
  for (let a = 0; a < m; a++) {
    for (let b = a; b < m; b++) {
      let dot = 0;
      for (let j = 0; j < d; j++) {
        dot += (covXY[a * d + j] ?? 0) * (covXY[b * d + j] ?? 0);
      }
      gram[a * m + b] = dot;
      gram[b * m + a] = dot;
    }
  }

  const { values, vectors, order } = jacobiSym(gram, m);
  const out = new Float64Array(m * d);
  for (let r = 0; r < m; r++) {
    const eig = Math.max(values[r] ?? 0, 0);
    const sigma = Math.sqrt(eig);
    if (sigma < 1e-12) {
      continue;
    }
    const col = order[r] ?? r;
    const scale = sigma ** singularPower / sigma;
    const base = r * d;
    for (let c = 0; c < m; c++) {
      const coeff = (vectors[c * m + col] ?? 0) * scale;
      const src = c * d;
      for (let j = 0; j < d; j++) {
        out[base + j] = (out[base + j] ?? 0) + coeff * (covXY[src + j] ?? 0);
      }
    }
  }
  return out;
}

/**
 * Löwdin 対称直交化＋元のノルム（予測力の大きさ）復元。
 * rows（m × d、行優先）の各行のノルム g_r を記録し、単位化後に G^(-1/2) で最小歪み直交化してから g_r を掛け直す。
 * 順序依存がなく、かつ弱いクラス（ノルムが小さい行）が白色化のように膨らまない。
 */
export function symmetricOrthogonalizeScaled(
  rows: Float64Array,
  m: number,
  d: number,
): Float64Array {
  const norms = new Float64Array(m);
  const unit = new Float64Array(m * d);
  for (let r = 0; r < m; r++) {
    const base = r * d;
    let sq = 0;
    for (let j = 0; j < d; j++) {
      const val = rows[base + j] ?? 0;
      sq += val * val;
    }
    const norm = Math.sqrt(sq);
    norms[r] = norm;
    if (norm > 1e-12) {
      for (let j = 0; j < d; j++) {
        unit[base + j] = (rows[base + j] ?? 0) / norm;
      }
    }
  }

  const gram = new Float64Array(m * m);
  for (let a = 0; a < m; a++) {
    for (let b = a; b < m; b++) {
      let dot = 0;
      for (let j = 0; j < d; j++) {
        dot += (unit[a * d + j] ?? 0) * (unit[b * d + j] ?? 0);
      }
      gram[a * m + b] = dot;
      gram[b * m + a] = dot;
    }
  }

  const { values, vectors, order } = jacobiSym(gram, m);
  // G^(-1/2) = V Λ^(-1/2) Vᵀ（m × m）
  const invSqrt = new Float64Array(m * m);
  for (let r = 0; r < m; r++) {
    const eig = values[r] ?? 0;
    if (eig <= 1e-10) {
      continue;
    }
    const inv = 1 / Math.sqrt(eig);
    const col = order[r] ?? r;
    for (let a = 0; a < m; a++) {
      const va = (vectors[a * m + col] ?? 0) * inv;
      for (let b = 0; b < m; b++) {
        invSqrt[a * m + b] = (invSqrt[a * m + b] ?? 0) + va * (vectors[b * m + col] ?? 0);
      }
    }
  }

  const out = new Float64Array(m * d);
  for (let a = 0; a < m; a++) {
    const g = norms[a] ?? 0;
    const base = a * d;
    for (let b = 0; b < m; b++) {
      const coeff = (invSqrt[a * m + b] ?? 0) * g;
      const src = b * d;
      for (let j = 0; j < d; j++) {
        out[base + j] = (out[base + j] ?? 0) + coeff * (unit[src + j] ?? 0);
      }
    }
  }
  return out;
}

/** 行優先 m × d の行列の行空間を修正 Gram-Schmidt で正規直交化する（有効な行だけ残す）。 */
function orthonormalizeRows(rows: Float64Array, m: number, d: number) {
  const kept: Array<Float64Array> = [];
  for (let r = 0; r < m; r++) {
    const v = Float64Array.from(rows.subarray(r * d, (r + 1) * d));
    for (const prev of kept) {
      let dot = 0;
      for (let j = 0; j < d; j++) {
        dot += (v[j] ?? 0) * (prev[j] ?? 0);
      }
      for (let j = 0; j < d; j++) {
        v[j] = (v[j] ?? 0) - dot * (prev[j] ?? 0);
      }
    }
    let sq = 0;
    for (let j = 0; j < d; j++) {
      const val = v[j] ?? 0;
      sq += val * val;
    }
    const norm = Math.sqrt(sq);
    if (norm > 1e-10) {
      for (let j = 0; j < d; j++) {
        v[j] = (v[j] ?? 0) / norm;
      }
      kept.push(v);
    }
  }
  const basis = new Float64Array(kept.length * d);
  for (let r = 0; r < kept.length; r++) {
    const row = kept[r];
    if (row) {
      basis.set(row, r * d);
    }
  }
  return { basis, rank: kept.length };
}

/**
 * 共分散行列 C（d × d）を、正規直交行基底 Q（rank × d）の直交補空間へ射影する：
 * C_⊥ = (I - QᵀQ) C (I - QᵀQ) = C - Qᵀ(QC) - (QC)ᵀQ + Qᵀ(QCQᵀ)Q
 * データ本体を再走査せずに O(rank * d²) で厳密計算できる。
 */
function projectCovarianceOrthogonal(
  cov: Float64Array,
  qBasis: Float64Array,
  rank: number,
  d: number,
): Float64Array {
  // qc = Q C（rank × d）
  const qc = new Float64Array(rank * d);
  for (let r = 0; r < rank; r++) {
    const qBase = r * d;
    for (let i = 0; i < d; i++) {
      let acc = 0;
      const cBase = i * d;
      for (let j = 0; j < d; j++) {
        acc += (qBasis[qBase + j] ?? 0) * (cov[cBase + j] ?? 0);
      }
      qc[qBase + i] = acc;
    }
  }

  // m = Q C Qᵀ（rank × rank）
  const mid = new Float64Array(rank * rank);
  for (let a = 0; a < rank; a++) {
    for (let b = a; b < rank; b++) {
      let acc = 0;
      for (let j = 0; j < d; j++) {
        acc += (qc[a * d + j] ?? 0) * (qBasis[b * d + j] ?? 0);
      }
      mid[a * rank + b] = acc;
      mid[b * rank + a] = acc;
    }
  }

  // mq = M Q（rank × d）
  const mq = new Float64Array(rank * d);
  for (let a = 0; a < rank; a++) {
    for (let j = 0; j < d; j++) {
      let acc = 0;
      for (let b = 0; b < rank; b++) {
        acc += (mid[a * rank + b] ?? 0) * (qBasis[b * d + j] ?? 0);
      }
      mq[a * d + j] = acc;
    }
  }

  const out = Float64Array.from(cov);
  for (let i = 0; i < d; i++) {
    const row = i * d;
    for (let j = i; j < d; j++) {
      let delta = 0;
      for (let r = 0; r < rank; r++) {
        const qi = qBasis[r * d + i] ?? 0;
        const qj = qBasis[r * d + j] ?? 0;
        const qci = qc[r * d + i] ?? 0;
        const qcj = qc[r * d + j] ?? 0;
        const mqj = mq[r * d + j] ?? 0;
        delta += -qi * qcj - qci * qj + qi * mqj;
      }
      const val = (out[row + j] ?? 0) + delta;
      out[row + j] = val;
      out[j * d + i] = val;
    }
  }
  return out;
}

/** 射影行列 rows（m × d、行優先）で中心化済みデータ xs を射影したときの RMS ノルム √E[‖Ux‖²] を計算する。 */
export function blockRmsNorm(
  xs: Array<Float64Array>,
  rows: Float64Array,
  m: number,
  d: number,
): number {
  const n = xs.length;
  if (n === 0 || m === 0) {
    return 1;
  }
  let sqSum = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    if (!x) {
      continue;
    }
    for (let r = 0; r < m; r++) {
      const base = r * d;
      let dot = 0;
      for (let j = 0; j < d; j++) {
        dot += (rows[base + j] ?? 0) * (x[j] ?? 0);
      }
      sqSum += dot * dot;
    }
  }
  return Math.sqrt(sqSum / n) || 1;
}

/** 射影行列 rows（m × d、行優先）で単一ベクトル x を射影する。 */
export function projectRows(
  x: ArrayLike<number>,
  rows: Float64Array,
  m: number,
  d: number,
): Float64Array {
  const out = new Float64Array(m);
  for (let r = 0; r < m; r++) {
    const base = r * d;
    let acc = 0;
    for (let j = 0; j < d; j++) {
      acc += (rows[base + j] ?? 0) * (x[j] ?? 0);
    }
    out[r] = acc;
  }
  return out;
}

/** 残余空間 C_⊥ から上位 residualDim 方向の正規直交 PCA 基底を求める。 */
export function fitResidualPca(
  covClean: Float64Array,
  supRows: Float64Array,
  supDim: number,
  d: number,
  residualDim: number,
): Float64Array {
  const { basis: qSup, rank } = orthonormalizeRows(supRows, supDim, d);
  const covPerp = projectCovarianceOrthogonal(covClean, qSup, rank, d);
  const { vectors } = topEigen(covPerp, d, residualDim);

  // 反復法の数値誤差で S_sup 成分がわずかに残らないよう、明示的に直交射影して再正規化する。
  const out = Float64Array.from(vectors);
  for (let r = 0; r < residualDim; r++) {
    const base = r * d;
    for (let s = 0; s < rank; s++) {
      const sBase = s * d;
      let dot = 0;
      for (let j = 0; j < d; j++) {
        dot += (out[base + j] ?? 0) * (qSup[sBase + j] ?? 0);
      }
      for (let j = 0; j < d; j++) {
        out[base + j] = (out[base + j] ?? 0) - dot * (qSup[sBase + j] ?? 0);
      }
    }
    let sq = 0;
    for (let j = 0; j < d; j++) {
      const val = out[base + j] ?? 0;
      sq += val * val;
    }
    const norm = Math.sqrt(sq);
    if (norm > 1e-12) {
      for (let j = 0; j < d; j++) {
        out[base + j] = (out[base + j] ?? 0) / norm;
      }
    }
  }
  return out;
}
