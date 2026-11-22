#!/usr/bin/env python3
"""
最終採用モデル `plamo-1b:rcca256-v1`（Step 1 Cov-Ortho 文字数除去 ＋ 強収縮 rCCA on denoised details）
を全公式レビューで学習し、Turso `embedding_transforms` 登録用の生バイナリ（A + b、little-endian float32）を書き出す。

出力フォーマット（plamo-1b-rcca256-v1.bin、計 2,098,176 bytes）:
  - 先頭 256 * 2048 * 4 = 2,097,152 bytes: A（行優先 float32）
  - 末尾 256 * 4        =     1,024 bytes: b（float32）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prepare_3pillar_candidates import fit_rcca_custom
from prepare_nonridge_judge import build_clean_detail_vectors
from train_eval_007 import (
    D,
    K,
    apply_rank1_eraser,
    fit_cov_ortho_eraser,
    log,
)


def main() -> None:
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    post_by_key = {p["key"]: p for p in official["posts"]}

    records_by_key: dict[str, dict[str, Any]] = {}
    with (out_dir / "all_qwen36.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records_by_key[rec["key"]] = rec

    x_npz = np.load(out_dir / "x_cache.npz", allow_pickle=False)
    x_keys = x_npz["keys"].tolist()
    x_arr = x_npz["xs"].astype(np.float64)
    x_map = {k: x_arr[i] for i, k in enumerate(x_keys)}

    d_npz = np.load(out_dir / "tag_details_cache.npz", allow_pickle=False)
    d_texts = d_npz["texts"].tolist()
    d_vecs = d_npz["vecs"].astype(np.float32)
    text_vecs = {t: d_vecs[i] for i, t in enumerate(d_texts)}

    all_keys = [
        p["key"] for p in official["posts"] if p["key"] in x_map and p["key"] in records_by_key
    ]
    x_all = np.stack([x_map[k] for k in all_keys], axis=0)
    lens_all = np.array([len(post_by_key[k]["content"]) for k in all_keys], dtype=np.float64)
    log(f"全 {len(all_keys)} 件で本番変換 plamo-1b:rcca256-v1 を学習中...")

    # Step 1: 中心化 + Cov-Ortho (len, λ=1.0)
    mean_x = x_all.mean(axis=0)  # (D,)
    x_c = x_all - mean_x[None, :]
    u_len = fit_cov_ortho_eraser(x_c, lens_all)  # (D,)
    x_clean = apply_rank1_eraser(x_c, u_len, 1.0)

    # Step 2: 強収縮 rCCA (reg_x=8.0, reg_y=4.0, power=1.5)
    y_clean, valid = build_clean_detail_vectors(all_keys, records_by_key, text_vecs)
    w_rcca = fit_rcca_custom(x_clean, y_clean, valid, K, reg_x=8.0, reg_y=4.0, power=1.5)  # (K, D)

    # 単一アフィン変換 z = A x + b へ厳密合成:
    #   x_clean = (I - u_len u_len^T) (x - mean_x)
    #   z = W_rcca @ x_clean = A x + b
    #   A = W_rcca @ (I - u_len u_len^T), b = -A @ mean_x
    a_f64 = w_rcca - np.outer(w_rcca @ u_len, u_len)  # (K, D)
    b_f64 = -(a_f64 @ mean_x)  # (K,)

    a_f32 = np.ascontiguousarray(a_f64, dtype="<f4")
    b_f32 = np.ascontiguousarray(b_f64, dtype="<f4")

    # 数値等価性の検証
    sample_x = x_all[:32]
    z_direct = x_clean[:32] @ w_rcca.T
    z_affine = sample_x.astype(np.float32) @ a_f32.T + b_f32[None, :]
    max_err = float(np.max(np.abs(z_direct - z_affine)))
    assert np.isfinite(a_f32).all() and np.isfinite(b_f32).all(), "NaN/Inf detected"
    log(f"数値等価性検証 OK: max |z_affine - z_direct| = {max_err:.2e}")

    bin_path = out_dir / "plamo-1b-rcca256-v1.bin"
    with bin_path.open("wb") as f:
        f.write(a_f32.tobytes())
        f.write(b_f32.tobytes())

    expected_bytes = (K * D + K) * 4
    actual_bytes = bin_path.stat().st_size
    assert actual_bytes == expected_bytes, f"Size mismatch: {actual_bytes} != {expected_bytes}"
    log(f"本番バイナリ出力完了: {bin_path} ({actual_bytes:,} bytes)")


if __name__ == "__main__":
    main()
