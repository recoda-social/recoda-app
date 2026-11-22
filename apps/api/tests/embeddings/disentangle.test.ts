import { describe, expect, it } from "vite-plus/test";
import { ridgeFit } from "../../scripts/lib/diagnose.ts";
import {
  applyEraser,
  crossCovarianceScalar,
  dualRidgeMultiFit,
  fitCovOrthoEraser,
  fitPlsBlock,
  fitResidualPca,
  fitShrinkageLeaceEraser,
  symmetricOrthogonalizeScaled,
} from "../../scripts/lib/disentangle.ts";
import { covariance } from "../../scripts/lib/pca.ts";

describe("disentangle helpers", () => {
  it("erases linear covariance with y exactly for both Cov-Ortho and Shrinkage-LEACE", () => {
    // 20 samples in 6D with anisotropic covariance and strong correlation with y
    const n = 20;
    const d = 6;
    const xs: Array<Float32Array> = [];
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const y = (i - 9.5) / 5;
      ys[i] = y;
      const content = (i % 4) - 1.5;
      xs.push(
        Float32Array.from([
          2 * y + 0.5 * content,
          -1.5 * y + content,
          content * 2,
          (i % 3) - 1,
          Math.sin(i),
          Math.cos(i),
        ]),
      );
    }

    const { mean, c: covX } = covariance(xs, d);
    const v = crossCovarianceScalar(xs, mean, ys, d);
    const centered = xs.map((row) => Float64Array.from(row, (val, j) => val - (mean[j] ?? 0)));

    const covEraser = fitCovOrthoEraser(v);
    const leaceEraser = fitShrinkageLeaceEraser(covX, v, d, 0.5);

    for (const eraser of [covEraser, leaceEraser]) {
      const cleaned = centered.map((x) => applyEraser(x, eraser, 1.0));
      const residualCov = crossCovarianceScalar(cleaned, new Float64Array(d), ys, d);
      for (const val of residualCov) {
        expect(Math.abs(val)).toBeLessThan(1e-10);
      }
    }
  });

  it("matches primal ridgeFit with dualRidgeMultiFit (Woodbury identity)", () => {
    const n = 12;
    const d = 20;
    const xs: Array<Float64Array> = [];
    const y0 = new Float64Array(n);
    const y1 = new Float64Array(n);
    const ys: Array<Float64Array> = [];
    for (let i = 0; i < n; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) {
        row[j] = Math.sin((i + 1) * (j + 1));
      }
      xs.push(row);
      y0[i] = i % 2 === 0 ? 1 : 0;
      y1[i] = i < 6 ? 1 : 0;
      ys.push(Float64Array.from([y0[i] ?? 0, y1[i] ?? 0]));
    }

    const alpha = 5;
    const primal0 = ridgeFit(xs, y0, alpha);
    const primal1 = ridgeFit(xs, y1, alpha);
    const dual = dualRidgeMultiFit(xs, ys, d, alpha);

    for (let j = 0; j < d; j++) {
      expect(dual[j]).toBeCloseTo(primal0.w[j] ?? 0, 8);
      expect(dual[d + j]).toBeCloseTo(primal1.w[j] ?? 0, 8);
    }
  });

  it("produces orthogonal rows with preserved shrinkage magnitudes in symmetricOrthogonalizeScaled", () => {
    const d = 4;
    // Row 0 has norm 2.0, Row 1 has norm 0.1 and is correlated with Row 0
    const rows = Float64Array.from([2, 0, 0, 0, 0.06, 0.08, 0, 0]);
    const ortho = symmetricOrthogonalizeScaled(rows, 2, d);

    let dot = 0;
    let norm0Sq = 0;
    let norm1Sq = 0;
    for (let j = 0; j < d; j++) {
      const a = ortho[j] ?? 0;
      const b = ortho[d + j] ?? 0;
      dot += a * b;
      norm0Sq += a * a;
      norm1Sq += b * b;
    }
    expect(Math.abs(dot)).toBeLessThan(1e-10);
    expect(Math.sqrt(norm0Sq)).toBeCloseTo(2.0, 6);
    expect(Math.sqrt(norm1Sq)).toBeCloseTo(0.1, 6);
  });

  it("makes residual PCA strictly orthogonal to supervised rows and extracts PLS block", () => {
    const n = 30;
    const d = 8;
    const xs: Array<Float32Array> = [];
    const xs64: Array<Float64Array> = [];
    const ys: Array<Float64Array> = [];
    for (let i = 0; i < n; i++) {
      const row = Float32Array.from([
        (i % 2 === 0 ? 1 : -1) * 2,
        (i % 3) - 1,
        Math.sin(i),
        Math.cos(i),
        Math.sin(2 * i),
        Math.cos(2 * i),
        (i % 5) - 2,
        (i % 7) - 3,
      ]);
      xs.push(row);
      xs64.push(Float64Array.from(row));
      ys.push(Float64Array.from([i % 2 === 0 ? 1 : 0, i % 3 === 0 ? 1 : 0]));
    }

    const pls = fitPlsBlock(xs64, ys, d, 1);
    const { c: covX } = covariance(xs, d);
    const res = fitResidualPca(covX, pls, 2, d, 4);

    // Every residual PCA row must be orthogonal to every PLS row
    for (let r = 0; r < 4; r++) {
      for (let s = 0; s < 2; s++) {
        let dot = 0;
        for (let j = 0; j < d; j++) {
          dot += (res[r * d + j] ?? 0) * (pls[s * d + j] ?? 0);
        }
        expect(Math.abs(dot)).toBeLessThan(1e-6);
      }
    }
  });
});
