#!/usr/bin/env python3
"""
キャッシュ済みデータ（x_cache.npz + tag_details_cache.npz + all_qwen36.jsonl）を使い、
Cloudflare API を呼ばずに GPU 上で以下を直接比較する（所要 ~30 秒）:
  1. Baseline (PCA 256)
  2. B1-Ridge(pri0.5) 純線形（Phase 1）
  3. B1-Ridge(pri0.5) + Phase 2 軽量ゲート（5 ブロック BCE ゲート）
  4. Hybrid (B1-Ridge 23次元 + 007-RCD detail ブロック + 残余 PCA) 純線形（Phase 1）
  5. Hybrid + Phase 2 軽量ゲート（5 ブロック BCE ゲート）
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_eval_007 import (
    ALL_TAGS,
    BLOCK_DIMS,
    BLOCK_TAGS,
    D,
    K,
    SUP_BLOCKS,
    apply_rank1_eraser,
    block_rms_norm,
    build_tag_matched_similarity,
    build_tag_matrix,
    eval_all_metrics,
    fit_007_joint_rcd,
    fit_block_gates,
    fit_cov_ortho_eraser,
    fit_residual_pca,
    fit_ridge_multi,
    log,
    top_pca_basis,
)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
    posts = official["posts"]
    post_by_key = {p["key"]: p for p in posts}
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
        for p in posts
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

    tr_sim, tr_mask = build_tag_matched_similarity(train_keys, records_by_key, text_vecs, dev)
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

    # 1. B1-Ridge (pri0.5, α=2000)
    y_tr_tags = build_tag_matrix(train_keys, records_by_key, non_primary_weight=0.5)
    w_ridge = fit_ridge_multi(x_tr_clean, y_tr_tags, alpha=2000.0)
    b1_slices: dict[str, slice] = {}
    off = 0
    for b in SUP_BLOCKS:
        b1_slices[b] = slice(off, off + len(BLOCK_TAGS[b]))
        off += len(BLOCK_TAGS[b])
    u_res_b1 = fit_residual_pca(cov_clean, w_ridge, K - len(ALL_TAGS))

    # 2. 5 ブロック軽量ゲート
    gates = fit_block_gates(x_tr_clean, tr_mask, l2_reg=0.05)

    def gate_of(x_clean: np.ndarray, b: str, temp: float = 1.0) -> np.ndarray:
        g, c = gates[b]
        z = np.clip((x_clean @ g + c) / temp, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-z)))[:, None]

    # 3. 007-RCD detail ブロック（軽量次元: scene:8, memory:16, music:16, meta:8, mood:16 = 64次元）
    for b, dv in zip(SUP_BLOCKS, (8, 16, 16, 8, 16), strict=True):
        BLOCK_DIMS[b] = dv
    log("Fitting 007-RCD detail blocks (8/16/16/8/16 = 64 dims, β=0.5)...")
    a_rcd = fit_007_joint_rcd(
        x_tr_clean,
        tr_sim,
        tr_mask,
        tau=0.1,
        beta_disentangle=0.5,
        alpha_l2=0.05,
        pca_subspace_dim=256,
        steps=350,
        device=dev,
    )
    sup_hybrid = np.concatenate([w_ridge] + [a_rcd[b] for b in SUP_BLOCKS], axis=0)  # 23 + 64 = 87
    u_res_hyb = fit_residual_pca(cov_clean, sup_hybrid, K - sup_hybrid.shape[0])  # 169

    results: list[tuple[str, dict[str, float]]] = [
        (
            "Baseline (PCA 256)",
            eval_all_metrics(
                x_ev_c @ u_pca.T, x_q_c @ u_pca.T, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat
            ),
        )
    ]

    # B1-Ridge: Phase 1 vs Phase 2 (+Gate)
    b1_blocks = [w_ridge[b1_slices[b]] for b in SUP_BLOCKS] + [u_res_b1]
    b1_rms = [block_rms_norm(x_tr_clean, blk) for blk in b1_blocks]
    b1_ev = [x_ev_clean @ blk.T for blk in b1_blocks]
    b1_q = [x_q_clean @ blk.T for blk in b1_blocks]
    b1_ev_g = [b1_ev[i] * gate_of(x_ev_clean, SUP_BLOCKS[i]) for i in range(5)] + [b1_ev[5]]
    b1_q_g = [b1_q[i] * gate_of(x_q_clean, SUP_BLOCKS[i]) for i in range(5)] + [b1_q[5]]

    for pname, ws_sq in [
        ("sup60/res40", (0.08, 0.14, 0.12, 0.08, 0.18, 0.40)),
        ("sup70/res30", (0.09, 0.16, 0.15, 0.08, 0.22, 0.30)),
    ]:
        ws = [math.sqrt(w) for w in ws_sq]
        ze = np.concatenate([b1_ev[i] * (ws[i] / b1_rms[i]) for i in range(6)], axis=1)
        zq = np.concatenate([b1_q[i] * (ws[i] / b1_rms[i]) for i in range(6)], axis=1)
        results.append((f"B1-Ridge P1 ({pname})", eval_all_metrics(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)))

        zeg = np.concatenate([b1_ev_g[i] * (ws[i] / b1_rms[i]) for i in range(6)], axis=1)
        zqg = np.concatenate([b1_q_g[i] * (ws[i] / b1_rms[i]) for i in range(6)], axis=1)
        results.append((f"B1-Ridge P2+Gate ({pname})", eval_all_metrics(zeg, zqg, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)))

    # Hybrid: [B1-Ridge(23) | 007-RCD(64) | residual(169)] Phase 1 vs Phase 2 (+Gate)
    r_rms = [block_rms_norm(x_tr_clean, w_ridge[b1_slices[b]]) for b in SUP_BLOCKS]
    d_rms = [block_rms_norm(x_tr_clean, a_rcd[b]) for b in SUP_BLOCKS]
    res_rms = block_rms_norm(x_tr_clean, u_res_hyb)

    r_ev = [x_ev_clean @ w_ridge[b1_slices[b]].T for b in SUP_BLOCKS]
    r_q = [x_q_clean @ w_ridge[b1_slices[b]].T for b in SUP_BLOCKS]
    d_ev = [x_ev_clean @ a_rcd[b].T for b in SUP_BLOCKS]
    d_q = [x_q_clean @ a_rcd[b].T for b in SUP_BLOCKS]
    res_ev = x_ev_clean @ u_res_hyb.T
    res_q = x_q_clean @ u_res_hyb.T

    for cfg_name, w_tag_tot, w_det_tot, w_res in [
        ("tag45/det25/res30", 0.45, 0.25, 0.30),
        ("tag35/det35/res30", 0.35, 0.35, 0.30),
        ("tag40/det20/res40", 0.40, 0.20, 0.40),
    ]:
        # 5 ブロック内訳比率（scene/memory/music/meta/mood）
        slot_share = (0.13, 0.23, 0.21, 0.12, 0.31)
        # Phase 1（純線形）
        ze_parts, zq_parts = [], []
        ze_g_parts, zq_g_parts = [], []
        for i, b in enumerate(SUP_BLOCKS):
            ct = math.sqrt(w_tag_tot * slot_share[i]) / r_rms[i]
            cd = math.sqrt(w_det_tot * slot_share[i]) / d_rms[i]
            ge = gate_of(x_ev_clean, b)
            gq = gate_of(x_q_clean, b)
            ze_parts.extend([r_ev[i] * ct, d_ev[i] * cd])
            zq_parts.extend([r_q[i] * ct, d_q[i] * cd])
            ze_g_parts.extend([r_ev[i] * ct * ge, d_ev[i] * cd * ge])
            zq_g_parts.extend([r_q[i] * ct * gq, d_q[i] * cd * gq])
        cr = math.sqrt(w_res) / res_rms
        ze_parts.append(res_ev * cr)
        zq_parts.append(res_q * cr)
        ze_g_parts.append(res_ev * cr)
        zq_g_parts.append(res_q * cr)

        ze = np.concatenate(ze_parts, axis=1)
        zq = np.concatenate(zq_parts, axis=1)
        results.append((f"Hybrid P1 ({cfg_name})", eval_all_metrics(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)))

        zeg = np.concatenate(ze_g_parts, axis=1)
        zqg = np.concatenate(zq_g_parts, axis=1)
        results.append((f"Hybrid P2+Gate ({cfg_name})", eval_all_metrics(zeg, zqg, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)))

    print("\n" + "=" * 144)
    print(
        f"{'条件':<34} {'asp@3':>6} {'mod@3':>6} {'言葉@10':>7} {'len@3':>6} | "
        f"{'S@3Avg':>6} {'nDCG@3':>6} {'空FP':>6} | "
        f"{'scn':>5} {'mem':>5} {'mus':>5} {'met':>5} {'mod':>5}"
    )
    print("-" * 144)
    for name, m in results:
        print(
            f"{name:<34} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
            f"{m['text@10']*100:6.1f}% {m['lenCorr@3']:6.2f} | "
            f"{m['S@3_macro']:6.3f} {m['nDCG@3_macro']:6.3f} {m['emptyFP_macro']*100:5.1f}% | "
            f"{m['nDCG@3_scene']:5.3f} {m['nDCG@3_memory']:5.3f} {m['nDCG@3_music']:5.3f} "
            f"{m['nDCG@3_meta']:5.3f} {m['nDCG@3_mood']:5.3f}"
        )
    print("=" * 144)


if __name__ == "__main__":
    main()
