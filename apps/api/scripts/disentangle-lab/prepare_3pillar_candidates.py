#!/usr/bin/env python3
"""
「強共鳴（3点）↑ ＋ ハズレ（0点）→ 0%」を狙う 3 本柱の純線形候補（z = Ax + b）を構築し、
定量指標＋新 DRAFT_ASSIST 準拠ジャッジ用の候補データ（完全未見 150 件）を出力する。

比較候補（すべて純線形 z = Ax + b）:
  1. baseline        : 現行 PCA 256
  2. clean_rcca_old  : 前回 rCCA (reg_x=2.0, reg_y=2.0, power=1.0)
  3. rcca_reg8       : 柱① 強収縮 rCCA (reg_x=8.0, reg_y=4.0, power=1.5)
  4. rcca_pca        : 柱①+② [rCCA(128, reg_x=8.0, p=1.5) 55% | Step1-PCA(128) 45%]
  5. rcca_pca_mood10 : 柱①+②+③ [rCCA(128) 48% | Step1-PCA(118) 42% | mood10(γ=0.5) 10%]
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prepare_nonridge_judge import build_clean_detail_vectors
from train_eval_007 import (
    BLOCK_TAGS,
    D,
    K,
    apply_rank1_eraser,
    block_rms_norm,
    build_tag_matched_similarity,
    build_tag_matrix,
    eval_all_metrics,
    fit_cov_ortho_eraser,
    fit_residual_pca,
    fit_ridge_multi,
    log,
    normalize_rows,
    top_pca_basis,
)


def fit_rcca_custom(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    valid: np.ndarray,
    k: int,
    reg_x: float,
    reg_y: float,
    power: float = 1.0,
) -> np.ndarray:
    xv = x_tr[valid]
    yv = y_tr[valid]
    xc = xv - xv.mean(axis=0, keepdims=True)
    yc = yv - yv.mean(axis=0, keepdims=True)
    n = len(xv)
    cxx = (xc.T @ xc) / n
    cyy = (yc.T @ yc) / n
    cxy = (xc.T @ yc) / n

    tx = float(np.trace(cxx)) / D
    ty = float(np.trace(cyy)) / D
    ex, vx = np.linalg.eigh(cxx + (reg_x * tx) * np.eye(D))
    ey, vy = np.linalg.eigh(cyy + (reg_y * ty) * np.eye(D))
    ix = (vx * (1.0 / np.sqrt(np.maximum(ex, 1e-12)))) @ vx.T
    iy = (vy * (1.0 / np.sqrt(np.maximum(ey, 1e-12)))) @ vy.T

    m_mat = ix @ cxy @ iy
    u_svd, s_svd, _ = np.linalg.svd(m_mat, full_matrices=False)
    w_x = (ix @ u_svd[:, :k]) * np.power(np.maximum(s_svd[:k], 1e-12), power)
    return w_x.T.copy()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
    songs_by_isrc = {s["isrc"]: s for s in official["songs"]}
    post_by_key = {p["key"]: p for p in official["posts"]}
    eval_keys_set = {r["key"] for r in eval_data["reviews"]}

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
    query_xs = x_npz["query_xs"].astype(np.float64)

    d_npz = np.load(out_dir / "tag_details_cache.npz", allow_pickle=False)
    d_texts = d_npz["texts"].tolist()
    d_vecs = d_npz["vecs"].astype(np.float32)
    text_vecs = {t: d_vecs[i] for i, t in enumerate(d_texts)}

    train_keys = [
        p["key"]
        for p in official["posts"]
        if p["key"] not in eval_keys_set and p["key"] in x_map and p["key"] in records_by_key
    ]
    eval_reviews = [
        r
        for r in eval_data["reviews"]
        if r["key"] in post_by_key and r["key"] in x_map and r["key"] in records_by_key
    ]
    eval_keys = [r["key"] for r in eval_reviews]

    x_train = np.stack([x_map[k] for k in train_keys], axis=0)
    len_train = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)
    x_eval = np.stack([x_map[k] for k in eval_keys], axis=0)
    eval_lengths = np.array([len(post_by_key[k]["content"]) for k in eval_keys], dtype=np.float64)

    ev_sim, ev_mask = build_tag_matched_similarity(eval_keys, records_by_key, text_vecs, dev)
    eval_tag_mat = build_tag_matrix(eval_keys, records_by_key, non_primary_weight=0.5)

    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c, x_ev_c, x_q_c = x_train - mean_x, x_eval - mean_x, query_xs - mean_x
    u_pca = top_pca_basis((x_tr_c.T @ x_tr_c) / len(x_tr_c), K)

    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)

    y_clean_tr, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)

    # 1. 前回 rCCA (reg_x=2.0, reg_y=2.0, p=1.0)
    log("Fitting rCCA (old reg_x=2, p=1.0) & 柱① (reg_x=8, p=1.5)...")
    w_rcca_old = fit_rcca_custom(x_tr_clean, y_clean_tr, valid_tr, K, reg_x=2.0, reg_y=2.0, power=1.0)
    w_rcca_strong = fit_rcca_custom(x_tr_clean, y_clean_tr, valid_tr, K, reg_x=8.0, reg_y=4.0, power=1.5)

    # 2. 柱①+②: [rCCA(128, reg_x=8, p=1.5) 55% | Step1-PCA(128) 45%]
    w_r128 = w_rcca_strong[:128]
    u_pca128 = top_pca_basis(cov_clean, 128)
    s_r128 = block_rms_norm(x_tr_clean, w_r128)
    s_p128 = block_rms_norm(x_tr_clean, u_pca128)
    w_rcca_pca = np.concatenate(
        [w_r128 * (math.sqrt(0.55) / s_r128), u_pca128 * (math.sqrt(0.45) / s_p128)], axis=0
    )

    # 3. 柱①+②+③: [rCCA(128) 48% | Step1-PCA(118) 42% | mood10(γ=0.5) 10%]
    y_all_tags = build_tag_matrix(train_keys, records_by_key, non_primary_weight=0.5)
    y_mood = y_all_tags[:, -len(BLOCK_TAGS["mood"]) :]  # 末尾 10 列が mood
    w_mood_raw = fit_ridge_multi(x_tr_clean, y_mood, alpha=2000.0)
    mood_stds = (x_tr_clean @ w_mood_raw.T).std(axis=0)
    w_mood = w_mood_raw * np.power(np.maximum(mood_stds, 1e-6), -0.5)[:, None]
    u_pca118 = top_pca_basis(cov_clean, 118)
    s_m10 = block_rms_norm(x_tr_clean, w_mood)
    s_p118 = block_rms_norm(x_tr_clean, u_pca118)
    w_rcca_pca_mood10 = np.concatenate(
        [
            w_r128 * (math.sqrt(0.48) / s_r128),
            u_pca118 * (math.sqrt(0.42) / s_p118),
            w_mood * (math.sqrt(0.10) / s_m10),
        ],
        axis=0,
    )

    cands = {
        "baseline": (x_ev_c @ u_pca.T, x_q_c @ u_pca.T),
        "clean_rcca_old": (x_ev_clean @ w_rcca_old.T, x_q_clean @ w_rcca_old.T),
        "rcca_reg8": (x_ev_clean @ w_rcca_strong.T, x_q_clean @ w_rcca_strong.T),
        "rcca_pca": (x_ev_clean @ w_rcca_pca.T, x_q_clean @ w_rcca_pca.T),
        "rcca_pca_mood10": (x_ev_clean @ w_rcca_pca_mood10.T, x_q_clean @ w_rcca_pca_mood10.T),
    }

    print("\n=== 3 本柱候補の定量指標（完全未見 700 件＋53 クエリ） ===")
    print(f"{'候補':<18} {'asp@3':>6} {'mod@3':>6} {'言葉@10':>7} {'len@3':>6} | {'S@3Avg':>6} {'nDCG@3':>6}")
    print("-" * 72)
    for name, (ze, zq) in cands.items():
        m = eval_all_metrics(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)
        print(
            f"{name:<18} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
            f"{m['text@10']*100:6.1f}% {m['lenCorr@3']:6.2f} | {m['S@3_macro']:6.3f} {m['nDCG@3_macro']:6.3f}"
        )

    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(len(eval_reviews)), 150))

    def pack_post(idx: int) -> dict[str, Any]:
        r = eval_reviews[idx]
        p = post_by_key[r["key"]]
        s = songs_by_isrc.get(r["isrc"], {})
        return {
            "key": r["key"],
            "artist": s.get("artistName", ""),
            "title": s.get("title", ""),
            "content": p["content"],
        }

    zs_norm = {k: normalize_rows(v[0]) for k, v in cands.items()}
    items: list[dict[str, Any]] = []
    for qi in sample_indices:
        entry: dict[str, Any] = {"query": pack_post(qi), "top3": {}}
        for mname, z in zs_norm.items():
            sims = z[qi] @ z.T
            sims[qi] = -np.inf
            top3 = np.argsort(-sims)[:3].tolist()
            entry["top3"][mname] = [pack_post(j) for j in top3]
        items.append(entry)

    out_path = out_dir / "judge_candidates_3pillar.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"候補データ保存完了: {out_path} ({len(items)} 件 × {len(cands)} 手法)")


if __name__ == "__main__":
    main()
