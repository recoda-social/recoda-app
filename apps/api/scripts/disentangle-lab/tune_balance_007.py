#!/usr/bin/env python3
"""
タグ間の強弱偏り（例: mem_era や tieup_visual が他タグを圧倒する現象）を診断し、
1. タグごとの出力標準偏差のソフト均一化（std_t^(-gamma), gamma ∈ {0, 0.5, 0.8}）
2. primary 重み強調（non_primary_weight ∈ {0.5, 0.3}）
3. ブロック重み比率 & ソフトゲート（floor + (1-floor)*σ）
を比較して、完全外れ(0/3)と偏りを最小化するチューニングスクリプト。
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
    normalize_rows,
    top_pca_basis,
)


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
    n_ev = len(eval_reviews)

    x_train = np.stack([x_map[k] for k in train_keys], axis=0)
    len_train = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)
    x_eval = np.stack([x_map[k] for k in eval_keys], axis=0)
    eval_lengths = np.array([len(post_by_key[k]["content"]) for k in eval_keys], dtype=np.float64)

    tr_sim, tr_mask = build_tag_matched_similarity(train_keys, records_by_key, text_vecs, dev)
    ev_sim, ev_mask = build_tag_matched_similarity(eval_keys, records_by_key, text_vecs, dev)
    eval_tag_mat = build_tag_matrix(eval_keys, records_by_key, non_primary_weight=0.5)

    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c, x_ev_c, x_q_c = x_train - mean_x, x_eval - mean_x, query_xs - mean_x
    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)

    # 診断 1: 現行 B1-Ridge(pri0.5) のタグごとの出力標準偏差を表示（どのタグが過剰に強いか確認）
    y_pri05 = build_tag_matrix(train_keys, records_by_key, non_primary_weight=0.5)
    w_r05 = fit_ridge_multi(x_tr_clean, y_pri05, alpha=2000.0)
    proj_tr05 = x_tr_clean @ w_r05.T  # (N, 23)
    tag_stds = proj_tr05.std(axis=0)
    print("=== 診断: B1-Ridge(pri0.5) 各タグの射影標準偏差（ブロック内での強さの偏り） ===")
    off = 0
    b1_slices: dict[str, slice] = {}
    for b in SUP_BLOCKS:
        tags = BLOCK_TAGS[b]
        b1_slices[b] = slice(off, off + len(tags))
        stds = tag_stds[off : off + len(tags)]
        parts = [f"{t}:{s:.4f}" for t, s in zip(tags, stds, strict=True)]
        ratio = stds.max() / max(stds.min(), 1e-9)
        print(f"  {b:<8} (max/min={ratio:.2f}x): " + ", ".join(parts))
        off += len(tags)

    # 007-RCD detail ブロック & ゲート
    for b, dv in zip(SUP_BLOCKS, (8, 16, 16, 8, 16), strict=True):
        BLOCK_DIMS[b] = dv
    a_rcd = fit_007_joint_rcd(
        x_tr_clean, tr_sim, tr_mask, tau=0.1, beta_disentangle=0.5, alpha_l2=0.05, steps=350, device=dev
    )
    gates = fit_block_gates(x_tr_clean, tr_mask, l2_reg=0.05)

    def gate_of(x_clean: np.ndarray, b: str, floor: float = 0.0, temp: float = 1.0) -> np.ndarray:
        g, c = gates[b]
        z = np.clip((x_clean @ g + c) / temp, -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        return (floor + (1.0 - floor) * sig)[:, None]

    def zero_count(ze: np.ndarray) -> int:
        zn = normalize_rows(ze)
        sim = zn @ zn.T
        np.fill_diagonal(sim, -np.inf)
        top3 = np.argsort(-sim, axis=1)[:, :3]
        zeros = 0
        for i in range(n_ev):
            ap, mp = eval_reviews[i]["aspectPrimary"], eval_reviews[i]["moodPrimary"]
            if not any(eval_reviews[j]["aspectPrimary"] == ap or eval_reviews[j]["moodPrimary"] == mp for j in top3[i]):
                zeros += 1
        return zeros

    # Tomorrow never knows のインデックスを特定
    tnk_idx = next(
        (i for i, r in enumerate(eval_reviews) if songs_by_isrc.get(r["isrc"], {}).get("title") == "Tomorrow never knows"),
        None,
    )

    print("\n" + "=" * 140)
    print(
        f"{'条件':<42} {'asp@3':>6} {'mod@3':>6} {'言葉@10':>7} {'len@3':>6} {'0/3':>4} | "
        f"{'S@3Avg':>6} {'nDCG@3':>6} {'空FP':>6} | {'scn':>5} {'mem':>5} {'mus':>5} {'met':>5} {'mod':>5}"
    )
    print("-" * 140)

    best_ze = None
    best_name = ""

    for npw in (0.5, 0.25):
        y_tags = build_tag_matrix(train_keys, records_by_key, non_primary_weight=npw)
        w_r = fit_ridge_multi(x_tr_clean, y_tags, alpha=2000.0)
        stds = (x_tr_clean @ w_r.T).std(axis=0)
        sup_hyb = np.concatenate([w_r] + [a_rcd[b] for b in SUP_BLOCKS], axis=0)
        u_res = fit_residual_pca(cov_clean, sup_hyb, K - sup_hyb.shape[0])
        d_rms = [block_rms_norm(x_tr_clean, a_rcd[b]) for b in SUP_BLOCKS]
        res_rms = block_rms_norm(x_tr_clean, u_res)

        for gamma in (0.0, 0.5, 0.8):
            # 各タグ行を std^(-gamma) でソフト均一化（極端に強いタグを抑える）
            scale_t = np.power(np.maximum(stds, 1e-6), -gamma)
            w_bal = w_r * scale_t[:, None]
            r_rms = [block_rms_norm(x_tr_clean, w_bal[b1_slices[b]]) for b in SUP_BLOCKS]

            # ブロック比率: memory 過多を抑えて mood/music/res を少し強めたバランス
            slot_share = (0.12, 0.20, 0.22, 0.12, 0.34)
            w_tag_tot, w_det_tot, w_res = 0.40, 0.20, 0.40

            for gate_cfg in [("P1線形", None), ("P2-Soft(f=0.35)", (0.35, 1.5)), ("P2-Full(f=0.0)", (0.0, 1.0))]:
                gname, gparam = gate_cfg
                if npw == 0.5 and gamma == 0.8 and gparam is not None:
                    continue
                if npw == 0.25 and gamma == 0.0 and gparam is not None:
                    continue
                ze_p, zq_p = [], []
                for i, b in enumerate(SUP_BLOCKS):
                    ct = math.sqrt(w_tag_tot * slot_share[i]) / r_rms[i]
                    cd = math.sqrt(w_det_tot * slot_share[i]) / d_rms[i]
                    re_b = x_ev_clean @ w_bal[b1_slices[b]].T
                    rq_b = x_q_clean @ w_bal[b1_slices[b]].T
                    de_b = x_ev_clean @ a_rcd[b].T
                    dq_b = x_q_clean @ a_rcd[b].T
                    if gparam is not None:
                        fl, tp = gparam
                        ge = gate_of(x_ev_clean, b, floor=fl, temp=tp)
                        gq = gate_of(x_q_clean, b, floor=fl, temp=tp)
                        ze_p.extend([re_b * ct * ge, de_b * cd * ge])
                        zq_p.extend([rq_b * ct * gq, dq_b * cd * gq])
                    else:
                        ze_p.extend([re_b * ct, de_b * cd])
                        zq_p.extend([rq_b * ct, dq_b * cd])
                cr = math.sqrt(w_res) / res_rms
                ze_p.append((x_ev_clean @ u_res.T) * cr)
                zq_p.append((x_q_clean @ u_res.T) * cr)

                ze = np.concatenate(ze_p, axis=1)
                zq = np.concatenate(zq_p, axis=1)
                m = eval_all_metrics(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)
                zc = zero_count(ze)
                label = f"{gname} npw={npw} γ={gamma}"
                print(
                    f"{label:<42} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
                    f"{m['text@10']*100:6.1f}% {m['lenCorr@3']:6.2f} {zc:4d} | "
                    f"{m['S@3_macro']:6.3f} {m['nDCG@3_macro']:6.3f} {m['emptyFP_macro']*100:5.1f}% | "
                    f"{m['nDCG@3_scene']:5.3f} {m['nDCG@3_memory']:5.3f} {m['nDCG@3_music']:5.3f} "
                    f"{m['nDCG@3_meta']:5.3f} {m['nDCG@3_mood']:5.3f}"
                )
                if label == "P2-Soft(f=0.35) npw=0.25 γ=0.5":
                    best_ze = ze
                    best_name = label
    print("=" * 140)

    if tnk_idx is not None and best_ze is not None:
        zn = normalize_rows(best_ze)
        sims = zn[tnk_idx] @ zn.T
        sims[tnk_idx] = -np.inf
        top3 = np.argsort(-sims)[:3]
        print(f"\n=== 悪化例だった 'Tomorrow never knows' の変化 ({best_name}) ===")
        for rnk, j in enumerate(top3, 1):
            rr = eval_reviews[j]
            rp = post_by_key[rr["key"]]
            rs = songs_by_isrc.get(rr["isrc"], {})
            print(f"  #{rnk} (cos={sims[j]:.3f}) {rs.get('artistName')} - {rs.get('title')} [{rr['aspectPrimary']}/{rr['moodPrimary']}]: {rp['content'][:70]}...")


if __name__ == "__main__":
    main()
