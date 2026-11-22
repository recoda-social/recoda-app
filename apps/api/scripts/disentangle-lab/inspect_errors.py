#!/usr/bin/env python3
"""
完全未見 700 件における「投稿→投稿 Top-3」の勝敗集計＋失敗例（悪化ケース・絶対失敗ケース）を抽出するエラー分析スクリプト。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_eval_007 import (
    BLOCK_DIMS,
    BLOCK_TAGS,
    K,
    SUP_BLOCKS,
    apply_rank1_eraser,
    block_rms_norm,
    build_tag_matched_similarity,
    build_tag_matrix,
    fit_007_joint_rcd,
    fit_block_gates,
    fit_cov_ortho_eraser,
    fit_residual_pca,
    fit_ridge_multi,
    normalize_rows,
    top_pca_basis,
)


def fmt_one(post: dict[str, Any], song: dict[str, Any], r: dict[str, Any]) -> str:
    return (
        f"{song.get('artistName', '?')} - {song.get('title', '?')} "
        f"({len(post['content'])}ch) [{r['aspectPrimary']}/{r['moodPrimary']}]\n"
        f"      {post['content']}"
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

    tr_sim, tr_mask = build_tag_matched_similarity(train_keys, records_by_key, text_vecs, dev)
    ev_sim, ev_mask = build_tag_matched_similarity(eval_keys, records_by_key, text_vecs, dev)

    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c, x_ev_c = x_train - mean_x, x_eval - mean_x
    u_pca = top_pca_basis((x_tr_c.T @ x_tr_c) / len(x_tr_c), K)
    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)

    y_tr_tags = build_tag_matrix(train_keys, records_by_key, non_primary_weight=0.5)
    w_ridge = fit_ridge_multi(x_tr_clean, y_tr_tags, alpha=2000.0)
    b1_slices: dict[str, slice] = {}
    off = 0
    for b in SUP_BLOCKS:
        b1_slices[b] = slice(off, off + len(BLOCK_TAGS[b]))
        off += len(BLOCK_TAGS[b])

    gates = fit_block_gates(x_tr_clean, tr_mask, l2_reg=0.05)

    def gate_of(x_clean: np.ndarray, b: str) -> np.ndarray:
        g, c = gates[b]
        z = np.clip(x_clean @ g + c, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-z)))[:, None]

    for b, dv in zip(SUP_BLOCKS, (8, 16, 16, 8, 16), strict=True):
        BLOCK_DIMS[b] = dv
    a_rcd = fit_007_joint_rcd(
        x_tr_clean, tr_sim, tr_mask, tau=0.1, beta_disentangle=0.5, alpha_l2=0.05, steps=350, device=dev
    )
    sup_hybrid = np.concatenate([w_ridge] + [a_rcd[b] for b in SUP_BLOCKS], axis=0)
    u_res_hyb = fit_residual_pca(cov_clean, sup_hybrid, K - sup_hybrid.shape[0])

    r_rms = [block_rms_norm(x_tr_clean, w_ridge[b1_slices[b]]) for b in SUP_BLOCKS]
    d_rms = [block_rms_norm(x_tr_clean, a_rcd[b]) for b in SUP_BLOCKS]
    res_rms = block_rms_norm(x_tr_clean, u_res_hyb)

    slot_share = (0.13, 0.23, 0.21, 0.12, 0.31)
    w_tag_tot, w_det_tot, w_res = 0.40, 0.20, 0.40
    ze_p1, ze_p2 = [], []
    for i, b in enumerate(SUP_BLOCKS):
        ct = math.sqrt(w_tag_tot * slot_share[i]) / r_rms[i]
        cd = math.sqrt(w_det_tot * slot_share[i]) / d_rms[i]
        re_b = x_ev_clean @ w_ridge[b1_slices[b]].T
        de_b = x_ev_clean @ a_rcd[b].T
        ge = gate_of(x_ev_clean, b)
        ze_p1.extend([re_b * ct, de_b * cd])
        ze_p2.extend([re_b * ct * ge, de_b * cd * ge])
    cr = math.sqrt(w_res) / res_rms
    ze_p1.append((x_ev_clean @ u_res_hyb.T) * cr)
    ze_p2.append((x_ev_clean @ u_res_hyb.T) * cr)

    z_base = normalize_rows(x_ev_c @ u_pca.T)
    z_p1 = normalize_rows(np.concatenate(ze_p1, axis=1))
    z_p2 = normalize_rows(np.concatenate(ze_p2, axis=1))

    def top3_and_score(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)
        top3 = np.argsort(-sim, axis=1)[:, :3]
        am_score = np.zeros(n_ev)
        s_score = np.zeros(n_ev)
        for i in range(n_ev):
            ap, mp = eval_reviews[i]["aspectPrimary"], eval_reviews[i]["moodPrimary"]
            a_hit = sum(1 for j in top3[i] if eval_reviews[j]["aspectPrimary"] == ap) / 3.0
            m_hit = sum(1 for j in top3[i] if eval_reviews[j]["moodPrimary"] == mp) / 3.0
            am_score[i] = 0.5 * (a_hit + m_hit)
            # アクティブブロック平均 S@3
            bs = [float(ev_sim[b][i, top3[i]].mean()) for b in SUP_BLOCKS if ev_mask[b][i]]
            s_score[i] = float(np.mean(bs)) if bs else 0.0
        return top3, sim, am_score, s_score

    top_b, sim_b, am_b, s_b = top3_and_score(z_base)
    top_1, sim_1, am_1, s_1 = top3_and_score(z_p1)
    top_2, sim_2, am_2, s_2 = top3_and_score(z_p2)

    print("\n=== 700 件全体での勝敗集計（投稿→投稿 Top-3） ===")
    for name, am_m, s_m in [("Hybrid-P1 vs Baseline", am_1, s_1), ("Hybrid-P2+Gate vs Baseline", am_2, s_2)]:
        w_am = int(np.sum(am_m > am_b + 1e-6))
        t_am = int(np.sum(np.abs(am_m - am_b) <= 1e-6))
        l_am = int(np.sum(am_m < am_b - 1e-6))
        w_s = int(np.sum(s_m > s_b + 1e-4))
        l_s = int(np.sum(s_m < s_b - 1e-4))
        zero_m = int(np.sum(am_m == 0.0))
        zero_b = int(np.sum(am_b == 0.0))
        print(
            f"  {name:<28}: [aspect+mood@3] 勝={w_am} / 分={t_am} / 負={l_am} | "
            f"[S@3具体] 勝={w_s} / 負={l_s} | 完全外れ(0/3)={zero_m}件 (Base={zero_b}件)"
        )

    # 1. Hybrid-P1 が Baseline より大きく悪化したワースト 4 件
    diff_1 = am_1 - am_b
    worst_reg = np.argsort(diff_1)[:4]
    print("\n" + "#" * 100)
    print("① 悪化例（Baseline > Hybrid-P1 となったワースト 4 件）")
    print("#" * 100)
    for idx in worst_reg:
        qr = eval_reviews[idx]
        qp = post_by_key[qr["key"]]
        qs = songs_by_isrc.get(qr["isrc"], {})
        print(
            f"\n--- QUERY (Base={am_b[idx]:.2f} -> P1={am_1[idx]:.2f}, P2={am_2[idx]:.2f} | "
            f"S@3 Base={s_b[idx]:.2f} -> P1={s_1[idx]:.2f}) ---"
        )
        print("  Q: " + fmt_one(qp, qs, qr))
        print("  [Baseline(PCA)]")
        for rnk, j in enumerate(top_b[idx], 1):
            print(f"    #{rnk} (cos={sim_b[idx, j]:.3f}) " + fmt_one(post_by_key[eval_reviews[j]["key"]], songs_by_isrc.get(eval_reviews[j]["isrc"], {}), eval_reviews[j]))
        print("  [Hybrid-P1(線形)]")
        for rnk, j in enumerate(top_1[idx], 1):
            print(f"    #{rnk} (cos={sim_1[idx, j]:.3f}) " + fmt_one(post_by_key[eval_reviews[j]["key"]], songs_by_isrc.get(eval_reviews[j]["isrc"], {}), eval_reviews[j]))

    # 2. Hybrid-P1 が 0/3（完全外れ）になった例のうち S@3 も低いワースト 3 件
    zero_idx = [i for i in range(n_ev) if am_1[i] == 0.0]
    zero_idx.sort(key=lambda i: s_1[i])
    print("\n" + "#" * 100)
    print("② Hybrid-P1 の絶対失敗例（aspect@3=0 かつ mood@3=0 かつ S@3 ワースト 3 件）")
    print("#" * 100)
    for idx in zero_idx[:3]:
        qr = eval_reviews[idx]
        qp = post_by_key[qr["key"]]
        qs = songs_by_isrc.get(qr["isrc"], {})
        print(f"\n--- QUERY (P1 S@3={s_1[idx]:.2f}, Base am={am_b[idx]:.2f}) ---")
        print("  Q: " + fmt_one(qp, qs, qr))
        print("  [Hybrid-P1(線形)]")
        for rnk, j in enumerate(top_1[idx], 1):
            print(f"    #{rnk} (cos={sim_1[idx, j]:.3f}) " + fmt_one(post_by_key[eval_reviews[j]["key"]], songs_by_isrc.get(eval_reviews[j]["isrc"], {}), eval_reviews[j]))

    # 3. 短い投稿（<=50ch）での比較 2 件
    short_idx = [i for i in range(n_ev) if len(post_by_key[eval_reviews[i]["key"]]["content"]) <= 50]
    short_idx.sort(key=lambda i: -(am_1[i] - am_b[i]))
    print("\n" + "#" * 100)
    print(f"③ 短い投稿（<=50ch、全 {len(short_idx)} 件）での改善例 2 件")
    print("#" * 100)
    for idx in short_idx[:2]:
        qr = eval_reviews[idx]
        qp = post_by_key[qr["key"]]
        qs = songs_by_isrc.get(qr["isrc"], {})
        print(f"\n--- QUERY (Base={am_b[idx]:.2f} -> P1={am_1[idx]:.2f}) ---")
        print("  Q: " + fmt_one(qp, qs, qr))
        print("  [Baseline(PCA)]")
        for rnk, j in enumerate(top_b[idx], 1):
            print(f"    #{rnk} (cos={sim_b[idx, j]:.3f}) " + fmt_one(post_by_key[eval_reviews[j]["key"]], songs_by_isrc.get(eval_reviews[j]["isrc"], {}), eval_reviews[j]))
        print("  [Hybrid-P1(線形)]")
        for rnk, j in enumerate(top_1[idx], 1):
            print(f"    #{rnk} (cos={sim_1[idx, j]:.3f}) " + fmt_one(post_by_key[eval_reviews[j]["key"]], songs_by_isrc.get(eval_reviews[j]["isrc"], {}), eval_reviews[j]))


if __name__ == "__main__":
    main()
