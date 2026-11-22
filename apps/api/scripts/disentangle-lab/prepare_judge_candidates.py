#!/usr/bin/env python3
"""
PLaMo 本来の微細ニュアンスを保持した軽量線形変換 3 候補 vs Baseline の
定量評価＋LLM-as-a-Judge 候補データ（完全未見 150 件）を作成する。

比較候補（すべて純線形 z = Ax + b、推論コード変更ゼロ）:
  - baseline    : 現行 PCA 256
  - step1_only  : Cov-Ortho (len λ=1.0) → PCA 256（教師なし・文字数除去のみ）
  - light_tag20 : Cov-Ortho + B1-Ridge(23, γ=0.5) 20% + residual PCA(233) 80%
  - light_hyb25 : Cov-Ortho + B1-Ridge(23, γ=0.5) 15% + 007-RCD(64) 10% + residual PCA(169) 75%
"""

from __future__ import annotations

import json
import math
import random
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
    fit_cov_ortho_eraser,
    fit_residual_pca,
    fit_ridge_multi,
    log,
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
    u_pca_clean = top_pca_basis(cov_clean, K)

    # B1-Ridge(pri0.5, γ=0.5)
    y_tags = build_tag_matrix(train_keys, records_by_key, non_primary_weight=0.5)
    w_r = fit_ridge_multi(x_tr_clean, y_tags, alpha=2000.0)
    stds = (x_tr_clean @ w_r.T).std(axis=0)
    w_bal = w_r * np.power(np.maximum(stds, 1e-6), -0.5)[:, None]

    b1_slices: dict[str, slice] = {}
    off = 0
    for b in SUP_BLOCKS:
        b1_slices[b] = slice(off, off + len(BLOCK_TAGS[b]))
        off += len(BLOCK_TAGS[b])
    u_res_tag = fit_residual_pca(cov_clean, w_r, K - len(ALL_TAGS))

    # 007-RCD detail ブロック
    for b, dv in zip(SUP_BLOCKS, (8, 16, 16, 8, 16), strict=True):
        BLOCK_DIMS[b] = dv
    a_rcd = fit_007_joint_rcd(
        x_tr_clean, tr_sim, tr_mask, tau=0.1, beta_disentangle=0.5, alpha_l2=0.05, steps=350, device=dev
    )
    sup_hyb = np.concatenate([w_r] + [a_rcd[b] for b in SUP_BLOCKS], axis=0)
    u_res_hyb = fit_residual_pca(cov_clean, sup_hyb, K - sup_hyb.shape[0])

    slot_share = (0.12, 0.20, 0.22, 0.12, 0.34)
    r_rms = [block_rms_norm(x_tr_clean, w_bal[b1_slices[b]]) for b in SUP_BLOCKS]
    d_rms = [block_rms_norm(x_tr_clean, a_rcd[b]) for b in SUP_BLOCKS]
    res_tag_rms = block_rms_norm(x_tr_clean, u_res_tag)
    res_hyb_rms = block_rms_norm(x_tr_clean, u_res_hyb)

    # 候補 2: light_tag20 (tag 20% / res 80%)
    ze_t20 = np.concatenate(
        [(x_ev_clean @ w_bal[b1_slices[b]].T) * (math.sqrt(0.20 * slot_share[i]) / r_rms[i]) for i, b in enumerate(SUP_BLOCKS)]
        + [(x_ev_clean @ u_res_tag.T) * (math.sqrt(0.80) / res_tag_rms)],
        axis=1,
    )
    zq_t20 = np.concatenate(
        [(x_q_clean @ w_bal[b1_slices[b]].T) * (math.sqrt(0.20 * slot_share[i]) / r_rms[i]) for i, b in enumerate(SUP_BLOCKS)]
        + [(x_q_clean @ u_res_tag.T) * (math.sqrt(0.80) / res_tag_rms)],
        axis=1,
    )

    # 候補 3: light_hyb25 (tag 15% + det 10% / res 75%)
    ze_h25_parts, zq_h25_parts = [], []
    for i, b in enumerate(SUP_BLOCKS):
        ct = math.sqrt(0.15 * slot_share[i]) / r_rms[i]
        cd = math.sqrt(0.10 * slot_share[i]) / d_rms[i]
        ze_h25_parts.extend([(x_ev_clean @ w_bal[b1_slices[b]].T) * ct, (x_ev_clean @ a_rcd[b].T) * cd])
        zq_h25_parts.extend([(x_q_clean @ w_bal[b1_slices[b]].T) * ct, (x_q_clean @ a_rcd[b].T) * cd])
    cr = math.sqrt(0.75) / res_hyb_rms
    ze_h25 = np.concatenate(ze_h25_parts + [(x_ev_clean @ u_res_hyb.T) * cr], axis=1)
    zq_h25 = np.concatenate(zq_h25_parts + [(x_q_clean @ u_res_hyb.T) * cr], axis=1)

    cands = {
        "baseline": (x_ev_c @ u_pca.T, x_q_c @ u_pca.T),
        "step1_only": (x_ev_clean @ u_pca_clean.T, x_q_clean @ u_pca_clean.T),
        "light_tag20": (ze_t20, zq_t20),
        "light_hyb25": (ze_h25, zq_h25),
    }

    print("\n=== 定量指標比較（完全未見 700 件＋53 クエリ） ===")
    print(f"{'候補':<16} {'asp@3':>6} {'mod@3':>6} {'言葉@10':>7} {'len@3':>6} | {'S@3Avg':>6} {'nDCG@3':>6}")
    print("-" * 68)
    for name, (ze, zq) in cands.items():
        m = eval_all_metrics(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)
        print(
            f"{name:<16} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
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

    out_path = out_dir / "judge_candidates_v2.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"LLM-as-a-Judge v2 候補データ保存完了: {out_path} ({len(items)} 件 × 4 手法)")


if __name__ == "__main__":
    main()
