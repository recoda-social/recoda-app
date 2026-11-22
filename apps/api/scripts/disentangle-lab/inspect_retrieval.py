#!/usr/bin/env python3
"""
完全未見の 700 件評価データ ＋ 53 言葉クエリに対して、
  1. Baseline (PCA 256)
  2. Hybrid P1 (tag40/det20/res40, 純線形 z=Ax+b)
  3. Hybrid P2+Gate (tag40/det20/res40, 軽量ゲート付き)
の実際の Top-3 検索結果を並べて目視比較するスクリプト。
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


def fmt_post(post: dict[str, Any], song: dict[str, Any], review: dict[str, Any] | None = None) -> str:
    artist = song.get("artistName", "?")
    title = song.get("title", "?")
    meta = f"{artist} - {title} ({len(post['content'])}ch)"
    if review:
        meta += f" [{review['aspectPrimary']}/{review['moodPrimary']}]"
    return f"{meta}\n      {post['content']}"


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

    tr_sim, tr_mask = build_tag_matched_similarity(train_keys, records_by_key, text_vecs, dev)

    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c, x_ev_c, x_q_c = x_train - mean_x, x_eval - mean_x, query_xs - mean_x
    u_pca = top_pca_basis((x_tr_c.T @ x_tr_c) / len(x_tr_c), K)
    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)

    # Fit Hybrid (tag40/det20/res40) + Gate
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

    ze_p1, zq_p1, ze_p2, zq_p2 = [], [], [], []
    for i, b in enumerate(SUP_BLOCKS):
        ct = math.sqrt(w_tag_tot * slot_share[i]) / r_rms[i]
        cd = math.sqrt(w_det_tot * slot_share[i]) / d_rms[i]
        re_b, rq_b = x_ev_clean @ w_ridge[b1_slices[b]].T, x_q_clean @ w_ridge[b1_slices[b]].T
        de_b, dq_b = x_ev_clean @ a_rcd[b].T, x_q_clean @ a_rcd[b].T
        ge, gq = gate_of(x_ev_clean, b), gate_of(x_q_clean, b)
        ze_p1.extend([re_b * ct, de_b * cd])
        zq_p1.extend([rq_b * ct, dq_b * cd])
        ze_p2.extend([re_b * ct * ge, de_b * cd * ge])
        zq_p2.extend([rq_b * ct * gq, dq_b * cd * gq])
    cr = math.sqrt(w_res) / res_rms
    ze_p1.append((x_ev_clean @ u_res_hyb.T) * cr)
    zq_p1.append((x_q_clean @ u_res_hyb.T) * cr)
    ze_p2.append((x_ev_clean @ u_res_hyb.T) * cr)
    zq_p2.append((x_q_clean @ u_res_hyb.T) * cr)

    models = {
        "Baseline(PCA)": (normalize_rows(x_ev_c @ u_pca.T), normalize_rows(x_q_c @ u_pca.T)),
        "Hybrid-P1(線形)": (normalize_rows(np.concatenate(ze_p1, axis=1)), normalize_rows(np.concatenate(zq_p1, axis=1))),
        "Hybrid-P2(+Gate)": (normalize_rows(np.concatenate(ze_p2, axis=1)), normalize_rows(np.concatenate(zq_p2, axis=1))),
    }

    # 代表的な未見評価投稿を 6 パターン自動選定
    def find_first(pred) -> int | None:
        for i, r in enumerate(eval_reviews):
            p = post_by_key[r["key"]]
            rec = records_by_key[r["key"]]
            if pred(r, p, rec):
                return i
        return None

    has_tag = lambda rec, slot, tag: any(e["tag"] == tag for e in rec["entries"].get(slot, []))
    sample_specs = [
        ("① 記憶（青春・学生）", lambda r, p, rec: has_tag(rec, "context", "mem_youth") and len(p["content"]) >= 80),
        ("② 記憶（恋愛・切ない）", lambda r, p, rec: has_tag(rec, "context", "mem_love") and has_tag(rec, "mood", "bittersweet")),
        ("③ 記憶（辛い時期・励まし）", lambda r, p, rec: has_tag(rec, "context", "mem_hardship") and has_tag(rec, "mood", "encouraged")),
        ("④ サウンド・かっこいい", lambda r, p, rec: r["aspectPrimary"] == "sound" and r["moodPrimary"] == "cool"),
        ("⑤ ライブ体験・高揚感", lambda r, p, rec: has_tag(rec, "meta", "artist_live") and r["moodPrimary"] == "hype"),
        ("⑥ 短い投稿（文字数バイアス検証）", lambda r, p, rec: len(p["content"]) <= 45 and has_tag(rec, "context", "scene_time") or len(p["content"]) <= 45 and r["moodPrimary"] in ("bittersweet", "calm", "warm")),
    ]

    print("\n" + "#" * 100)
    print("経路①：投稿 → 類似投稿 Top-3 比較（完全未見 700 件プール）")
    print("#" * 100)
    for label, pred in sample_specs:
        qi = find_first(pred)
        if qi is None:
            continue
        qr = eval_reviews[qi]
        qp = post_by_key[qr["key"]]
        qs = songs_by_isrc.get(qr["isrc"], {})
        print(f"\n==================== {label} ====================")
        print(f"QUERY: {fmt_post(qp, qs, qr)}")
        for mname, (ze, _) in models.items():
            sims = ze[qi] @ ze.T
            sims[qi] = -np.inf
            top3 = np.argsort(-sims)[:3]
            print(f"\n  [{mname}]")
            for rank, j in enumerate(top3, 1):
                rr = eval_reviews[j]
                rp = post_by_key[rr["key"]]
                rs = songs_by_isrc.get(rr["isrc"], {})
                print(f"    #{rank} (cos={sims[j]:.3f}) {fmt_post(rp, rs, rr)}")

    print("\n" + "#" * 100)
    print("経路②：短い言葉検索 → 投稿 Top-3 比較（53 クエリから 4 件抜粋）")
    print("#" * 100)
    for qi in [0, 10, 25, 40]:
        if qi >= len(eval_data["queries"]):
            continue
        q = eval_data["queries"][qi]
        print(f"\n==================== クエリ: 「{q['text']}」(expect aspect={q['aspect']}, mood={q['mood']}) ====================")
        for mname, (ze, zq) in models.items():
            sims = zq[qi] @ ze.T
            top3 = np.argsort(-sims)[:3]
            print(f"\n  [{mname}]")
            for rank, j in enumerate(top3, 1):
                rr = eval_reviews[j]
                rp = post_by_key[rr["key"]]
                rs = songs_by_isrc.get(rr["isrc"], {})
                hit = ((not q["aspect"]) or (q["aspect"] in rr["aspects"])) and ((not q["mood"]) or (q["mood"] in rr["moods"]))
                mark = "✓" if hit else "✗"
                print(f"    #{rank} {mark} (cos={sims[j]:.3f}) {fmt_post(rp, rs, rr)}")


if __name__ == "__main__":
    main()
