#!/usr/bin/env python3
"""
Metric Learning 用の学習ペア（非評価 2,978 件のみ・リークゼロ）をマイニングする。
800 件の起点投稿それぞれに対し、同一曲を除外した 10 件の候補を抽出：
  - Hard Positive 候補 (5件): {tag, details} の Tag-Matched Detail Similarity 上位（埋もれた 3 点の宝）
  - Hard Negative / 境界候補 (5件): Baseline PCA および rcca_reg8 の上位（現行埋め込みが陥る 0〜1 点の罠）
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prepare_3pillar_candidates import fit_rcca_custom
from prepare_nonridge_judge import build_clean_detail_vectors
from train_eval_007 import (
    K,
    SUP_BLOCKS,
    apply_rank1_eraser,
    build_tag_matched_similarity,
    fit_cov_ortho_eraser,
    log,
    normalize_rows,
    top_pca_basis,
)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
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
    n_tr = len(train_keys)
    train_isrcs = [post_by_key[k]["isrc"] for k in train_keys]
    x_train = np.stack([x_map[k] for k in train_keys], axis=0)
    len_train = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)

    log(f"学習プール: {n_tr} 件から 800 クエリ × 10 候補をマイニング中...")

    # 1. Baseline PCA 256 & rcca_reg8
    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c = x_train - mean_x
    u_pca = top_pca_basis((x_tr_c.T @ x_tr_c) / n_tr, K)
    z_base = normalize_rows(x_tr_c @ u_pca.T)

    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    y_clean, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)
    w_rcca = fit_rcca_custom(x_tr_clean, y_clean, valid_tr, K, reg_x=8.0, reg_y=4.0, power=1.5)
    z_rcca = normalize_rows(x_tr_clean @ w_rcca.T)

    # 2. Tag-Matched Detail Similarity（全 5 ブロック合計）
    tr_sim, _ = build_tag_matched_similarity(train_keys, records_by_key, text_vecs, dev)
    s_total = sum(tr_sim[b] for b in SUP_BLOCKS)

    # 類似度行列
    sim_base = z_base @ z_base.T
    sim_rcca = z_rcca @ z_rcca.T

    rng = random.Random(42)
    query_indices = sorted(rng.sample(range(n_tr), 800))

    pairs_out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for qi in query_indices:
        q_key = train_keys[qi]
        q_isrc = train_isrcs[qi]
        q_text = post_by_key[q_key]["content"]

        # 同一曲を除外するマスク
        valid_j = np.array([j != qi and train_isrcs[j] != q_isrc for j in range(n_tr)], dtype=bool)

        # Baseline / rCCA 上位（Hard Negative / 境界候補）
        sb = np.where(valid_j, sim_base[qi], -np.inf)
        sr = np.where(valid_j, sim_rcca[qi], -np.inf)
        st = np.where(valid_j, s_total[qi], -np.inf)

        top_base = np.argsort(-sb)[:10].tolist()
        top_rcca = np.argsort(-sr)[:10].tolist()
        top_tag = np.argsort(-st)[:15].tolist()

        chosen: list[tuple[int, str]] = []
        chosen_set: set[int] = set()

        # まず Baseline Top-3 + rCCA Top-2（計 5 件の現在の上位＝Hard Negative 含む）
        for j in top_base[:3]:
            if j not in chosen_set:
                chosen.append((j, "base_top"))
                chosen_set.add(j)
        for j in top_rcca:
            if len(chosen) >= 5:
                break
            if j not in chosen_set:
                chosen.append((j, "rcca_top"))
                chosen_set.add(j)

        # 次に {tag, details} 上位から 5 件（Baseline Top-3 に居ない埋もれた 3 点候補＝Hard Positive 優先）
        for j in top_tag:
            if len(chosen) >= 10:
                break
            if j not in chosen_set and st[j] > 0.0:
                chosen.append((j, "detail_mine"))
                chosen_set.add(j)

        # 10 件に満たなければ rCCA / Baseline 次点から補充
        for j in top_rcca + top_base:
            if len(chosen) >= 10:
                break
            if j not in chosen_set:
                chosen.append((j, "fill"))
                chosen_set.add(j)

        for dj, src in chosen:
            d_key = train_keys[dj]
            if (q_key, d_key) in seen_pairs:
                continue
            seen_pairs.add((q_key, d_key))
            pairs_out.append(
                {
                    "q_key": q_key,
                    "d_key": d_key,
                    "source": src,
                    "q_text": q_text,
                    "d_text": post_by_key[d_key]["content"],
                }
            )

    out_path = out_dir / "train_pairs_800.json"
    out_path.write_text(json.dumps(pairs_out, ensure_ascii=False), encoding="utf-8")
    src_counts: dict[str, int] = {}
    for p in pairs_out:
        src_counts[p["source"]] = src_counts.get(p["source"], 0) + 1
    log(f"マイニング完了: {len(pairs_out)} ペア保存 -> {out_path} | 内訳={src_counts}")


if __name__ == "__main__":
    main()
