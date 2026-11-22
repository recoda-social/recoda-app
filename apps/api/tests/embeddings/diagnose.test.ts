import { describe, expect, it } from "vite-plus/test";
import {
  attenuateSubspace,
  decadeOfYear,
  estimateArtistSubspace,
  evaluateLeakage,
  parseIsrcYear,
  randomLeakageBaseline,
  residualizeByFeatures,
  ridgeCvR2,
  ridgeFit,
  ridgePredict,
} from "../../scripts/lib/diagnose.ts";
import type { DiagnosticItem } from "../../scripts/lib/diagnose.ts";

describe("diagnose-embeddings helpers", () => {
  it("parses ISRC year and decade", () => {
    expect(parseIsrcYear("JPB609631901")).toBe(1996);
    expect(parseIsrcYear("JPU902604202")).toBe(2026);
    expect(parseIsrcYear("JPTK08800560")).toBe(1988);
    expect(decadeOfYear(1996)).toBe("1990s");
    expect(decadeOfYear(2026)).toBe("2020s");
    expect(parseIsrcYear("SHORT")).toBeNull();
  });

  it("recovers linear relationship via ridgeFit and ridgeCvR2", () => {
    const xs: Array<Float64Array> = [];
    const ys = new Float64Array(40);
    for (let i = 0; i < 40; i++) {
      const a = (i % 7) - 3;
      const b = Math.floor(i / 7) - 2.5;
      xs.push(Float64Array.from([a, b, (i % 3) - 1]));
      ys[i] = 2 * a - 1.5 * b + 3;
    }
    const model = ridgeFit(xs, ys, 1e-4);
    expect(ridgePredict(model, Float64Array.from([1, 2, 0]))).toBeCloseTo(2, 2);

    const cv = ridgeCvR2(xs, ys, [0.001, 0.1, 10], 5);
    expect(cv.r2).toBeGreaterThan(0.99);
  });

  it("removes feature-explained variance via residualizeByFeatures", () => {
    const features: Array<Float64Array> = [];
    const targets = new Float64Array(20);
    for (let i = 0; i < 20; i++) {
      const isOld = i < 10 ? 1 : 0;
      features.push(Float64Array.from([isOld]));
      targets[i] = isOld ? 1990 : 2024;
    }
    const residual = residualizeByFeatures(targets, features, 1e-4);
    for (const v of residual) {
      expect(Math.abs(v)).toBeLessThan(1e-2);
    }
  });

  it("attenuating artist subspace reduces artist leakage while preserving orthogonal content", () => {
    // 4 artists × 4 items (each artist has only 2 lyrics and 2 sound):
    // Before attenuation, top-3 stays inside the same artist so at most 1/3 matches aspect;
    // After attenuation, top-3 crosses artists and matches aspect 100%.
    const artistCoords: Array<[string, number, number]> = [
      ["artist-a", 4, 0],
      ["artist-b", -4, 0],
      ["artist-c", 0, 4],
      ["artist-d", 0, -4],
    ];
    const items: Array<DiagnosticItem> = [];
    const vectors: Array<Float64Array> = [];
    for (const [artistKey, ax, ay] of artistCoords) {
      for (let k = 0; k < 4; k++) {
        const isLyrics = k < 2;
        items.push({
          artistKey,
          decade: "2020s",
          aspectPrimary: isLyrics ? "lyrics" : "sound",
          aspects: [isLyrics ? "lyrics" : "sound"],
          moodPrimary: "moved",
          moods: ["moved"],
          contentLength: 100,
          mentionsArtist: false,
          mentionsTitle: false,
        });
        vectors.push(Float64Array.from([isLyrics ? 1 : -1, ax, ay, k * 0.01]));
      }
    }

    const before = evaluateLeakage(items, vectors);
    const basis = estimateArtistSubspace(
      items.map((it, i) => ({ artistKey: it.artistKey, z: vectors[i] ?? new Float64Array(4) })),
      4,
      2,
    );
    const afterVectors = vectors.map((v) => attenuateSubspace(v, basis, 2, 1.0));
    const after = evaluateLeakage(items, afterVectors);

    expect(before.sameArtistAt3).toBe(1);
    expect(before.aspectAt3).toBeLessThan(0.5);
    expect(after.aspectAt3).toBe(1);
    expect(after.sameArtistAt3).toBeLessThan(before.sameArtistAt3);

    const base = randomLeakageBaseline(items);
    expect(base.sameArtistEligible).toBeCloseTo(3 / 15, 5);
  });
});
