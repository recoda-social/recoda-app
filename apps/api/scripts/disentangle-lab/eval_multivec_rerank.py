#!/usr/bin/env python3
"""
Stage-2 マルチベクトル Reranking（LLM コストゼロ・x からの線形射影＋ゲートのみ）のローカル検証スクリプト。

確認する内容:
  1. 抽出忠実度（unseen 700件）:
     - 各スロットの W_s x̃ が、Qwen 抽出の具体内容ベクトル Y_s（details の PLaMo 埋め込み）の
       類似構造を「単一ベクトル rcca_reg8」より正確に捉えているか？
     - クロススロット分離（W_context は Y_music より Y_context に強く相関するか？）
     - 存在ゲート p_s(x) の AUROC
  2. Top-20 → Top-3 Reranking の定量比較（unseen 700件）:
     - Stage 1 (rcca_reg8) 単体 vs 4スロット / 5スロット / ゲート方式 / タグ併用
  3. 具体例の確認:
     - 順位が入れ替わった unseen クエリの Top-3 差分とスロット別スコア内訳
     - 公式レビュー内の「隣で」「らしさ」「Only You」のスロット別プロフィール比較
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from prepare_3pillar_candidates import fit_rcca_custom
from prepare_nonridge_judge import build_clean_detail_vectors
from train_eval_007 import (
    ALL_TAGS,
    BLOCK_TAGS,
    D,
    K,
    SUP_BLOCKS,
    apply_rank1_eraser,
    build_tag_matched_similarity,
    build_tag_matrix,
    eval_all_metrics,
    extract_tag_entries,
    fit_cov_ortho_eraser,
    log,
    normalize_rows,
)

SLOTS_4 = ("music", "meta", "context", "mood")
SLOTS_5 = SUP_BLOCKS  # ("scene", "memory", "music", "meta", "mood")

SLOT4_TAGS: dict[str, set[str]] = {
    "music": set(BLOCK_TAGS["music"]),
    "meta": set(BLOCK_TAGS["meta"]),
    "context": set(BLOCK_TAGS["scene"]) | set(BLOCK_TAGS["memory"]),
    "mood": set(BLOCK_TAGS["mood"]),
}
SLOT5_TAGS: dict[str, set[str]] = {b: set(BLOCK_TAGS[b]) for b in SLOTS_5}


def build_slot_detail_vectors(
    keys: list[str],
    records_by_key: dict[str, dict[str, Any]],
    text_vecs: dict[str, np.ndarray],
    slot_tags: dict[str, set[str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    各スロット s について:
      - y_slots[s]: (N, D) 単位正規化された details 埋め込み（無い行は 0）
      - masks[s]:   (N,) bool そのスロットに 1 件以上 entry があるか
      - prims[s]:   (N,) bool そのスロットに primary=True の entry があるか
    """
    n = len(keys)
    y_slots = {s: np.zeros((n, D), dtype=np.float64) for s in slot_tags}
    masks = {s: np.zeros(n, dtype=bool) for s in slot_tags}
    prims = {s: np.zeros(n, dtype=bool) for s in slot_tags}

    for i, k in enumerate(keys):
        entries = extract_tag_entries(records_by_key[k])
        for s, tags in slot_tags.items():
            acc = np.zeros(D, dtype=np.float64)
            has_any = False
            has_prim = False
            for tag, (text, w) in entries.items():
                if tag not in tags:
                    continue
                v = text_vecs.get(text)
                if v is None:
                    continue
                vn = float(np.linalg.norm(v))
                if vn > 1e-12:
                    acc += w * (v.astype(np.float64) / vn)
                    has_any = True
                    if w >= 0.99:
                        has_prim = True
            norm = float(np.linalg.norm(acc))
            if has_any and norm > 1e-12:
                y_slots[s][i] = acc / norm
                masks[s][i] = True
                prims[s][i] = has_prim
    return y_slots, masks, prims


def fit_slot_models(
    x_tr: np.ndarray,
    y_slots_tr: dict[str, np.ndarray],
    masks_tr: dict[str, np.ndarray],
    prims_tr: dict[str, np.ndarray],
    slot_names: tuple[str, ...],
    slot_dim: int = 32,
    reg_x: float = 12.0,
    reg_y: float = 4.0,
    power: float = 1.5,
) -> tuple[dict[str, np.ndarray], dict[str, LogisticRegression], dict[str, LogisticRegression]]:
    w_slots: dict[str, np.ndarray] = {}
    gates_mask: dict[str, LogisticRegression] = {}
    gates_prim: dict[str, LogisticRegression] = {}
    # ロジスティック回帰用に単位正規化した x̃ を使用（正則化スケールを安定させる）
    x_tr_u = normalize_rows(x_tr)
    for s in slot_names:
        m = masks_tr[s]
        p = prims_tr[s]
        w_slots[s] = fit_rcca_custom(
            x_tr, y_slots_tr[s], m, slot_dim, reg_x=reg_x, reg_y=reg_y, power=power
        )
        clf_m = LogisticRegression(C=1.0, max_iter=500)
        clf_m.fit(x_tr_u, m.astype(int))
        gates_mask[s] = clf_m

        clf_p = LogisticRegression(C=1.0, max_iter=500)
        clf_p.fit(x_tr_u, p.astype(int))
        gates_prim[s] = clf_p
    return w_slots, gates_mask, gates_prim


def fit_tag_classifiers(x_tr: np.ndarray, tag_mat_tr: np.ndarray) -> list[LogisticRegression | None]:
    x_tr_u = normalize_rows(x_tr)
    clfs: list[LogisticRegression | None] = []
    for t_idx in range(tag_mat_tr.shape[1]):
        y = (tag_mat_tr[:, t_idx] > 0).astype(int)
        if y.sum() < 5 or y.sum() > len(y) - 5:
            clfs.append(None)
            continue
        clf = LogisticRegression(C=2.0, max_iter=500)
        clf.fit(x_tr_u, y)
        clfs.append(clf)
    return clfs


def predict_tag_probs(x: np.ndarray, clfs: list[LogisticRegression | None]) -> np.ndarray:
    x_u = normalize_rows(x)
    out = np.zeros((len(x), len(clfs)), dtype=np.float64)
    for i, clf in enumerate(clfs):
        if clf is not None:
            out[:, i] = clf.predict_proba(x_u)[:, 1]
    return out


def eval_slot_fidelity(
    slot_names: tuple[str, ...],
    z_stage1_ev: np.ndarray,
    x_ev: np.ndarray,
    w_slots: dict[str, np.ndarray],
    gates: dict[str, LogisticRegression],
    y_slots_ev: dict[str, np.ndarray],
    masks_ev: dict[str, np.ndarray],
) -> None:
    """
    unseen 700件で「W_s x̃ が実際に正解の Y_s（具体内容）に近くなっているか」を定量評価する。
    """
    log("=== 1. スロット別 抽出忠実度 & Disentanglement（unseen 700件） ===")
    z1_u = normalize_rows(z_stage1_ev)
    x_ev_u = normalize_rows(x_ev)
    z_slot_u = {s: normalize_rows(x_ev @ w_slots[s].T) for s in slot_names}

    print(
        f"{'slot':<10} {'n_ev':>5} {'gateAUC':>8} | "
        f"{'r(Stage1,Y_s)':>13} {'r(W_s,Y_s)':>11} {'Δr':>7} | "
        f"{'Rec@10(S1)':>10} {'Rec@10(W_s)':>11} {'ΔRec@10':>8}"
    )
    print("-" * 96)

    for s in slot_names:
        m = masks_ev[s]
        idx = np.where(m)[0]
        p_gate = gates[s].predict_proba(x_ev_u)[:, 1]
        auc = roc_auc_score(m.astype(int), p_gate) if 0 < m.sum() < len(m) else float("nan")

        # 対象スロットを持つペア同士での類似度相関（上三角）
        y_u = normalize_rows(y_slots_ev[s][idx])
        s_true = y_u @ y_u.T
        s_s1 = z1_u[idx] @ z1_u[idx].T
        s_ws = z_slot_u[s][idx] @ z_slot_u[s][idx].T

        triu = np.triu_indices(len(idx), k=1)
        r_s1 = float(np.corrcoef(s_true[triu], s_s1[triu])[0, 1])
        r_ws = float(np.corrcoef(s_true[triu], s_ws[triu])[0, 1])

        # Oracle Y_s の Top-10 近傍をどれだけ再現できるか（Recall@10）
        np.fill_diagonal(s_true, -1e9)
        np.fill_diagonal(s_s1, -1e9)
        np.fill_diagonal(s_ws, -1e9)
        top_true = np.argpartition(-s_true, 10, axis=1)[:, :10]
        top_s1 = np.argpartition(-s_s1, 10, axis=1)[:, :10]
        top_ws = np.argpartition(-s_ws, 10, axis=1)[:, :10]

        rec_s1 = float(
            np.mean([len(set(top_true[i]) & set(top_s1[i])) / 10.0 for i in range(len(idx))])
        )
        rec_ws = float(
            np.mean([len(set(top_true[i]) & set(top_ws[i])) / 10.0 for i in range(len(idx))])
        )

        print(
            f"{s:<10} {len(idx):>5} {auc:>8.3f} | "
            f"{r_s1:>13.4f} {r_ws:>11.4f} {r_ws - r_s1:>+7.4f} | "
            f"{rec_s1 * 100:>9.2f}% {rec_ws * 100:>10.2f}% {(rec_ws - rec_s1) * 100:>+7.2f}%"
        )

    # クロススロット相関行列: 行=予測 W_s, 列=正解 Y_t
    print("\nクロススロット相関行列 corr(cos(W_row), cos(Y_col))（対角が最大なら分離成功）:")
    header = f"{'W \\ Y':<10}" + "".join(f"{t:>10}" for t in slot_names)
    print(header)
    for r_s in slot_names:
        row_str = f"{r_s:<10}"
        for c_s in slot_names:
            idx = np.where(masks_ev[c_s])[0]
            y_u = normalize_rows(y_slots_ev[c_s][idx])
            s_true = y_u @ y_u.T
            s_pred = z_slot_u[r_s][idx] @ z_slot_u[r_s][idx].T
            triu = np.triu_indices(len(idx), k=1)
            corr = float(np.corrcoef(s_true[triu], s_pred[triu])[0, 1])
            row_str += f"{corr:>10.4f}"
        print(row_str)


def compute_reranked_scores(
    sim_stage1: np.ndarray,
    z_slots_u: dict[str, np.ndarray],
    p_gates: dict[str, np.ndarray],
    slot_names: tuple[str, ...],
    isrcs: list[str],
    top_k_cand: int = 20,
    alpha: float = 0.35,
    gate_mode: str = "prod",
    p_tags: np.ndarray | None = None,
    beta_tag: float = 0.0,
) -> np.ndarray:
    """
    Stage 1（rcca_reg8）の上位 top_k_cand 件（同一曲除外後）を Stage 2 で並べ替えたスコア行列を返す。
    候補外は -1e9 のまま。
    """
    n = sim_stage1.shape[0]
    out = np.full((n, n), -1e9, dtype=np.float64)

    # 各スロットのコサイン類似度行列を前計算
    sim_slots = {s: z_slots_u[s] @ z_slots_u[s].T for s in slot_names}
    sim_tag = None
    if p_tags is not None and beta_tag > 0:
        pt_u = normalize_rows(p_tags)
        sim_tag = pt_u @ pt_u.T

    for i in range(n):
        # 自分自身と同曲を除外して Stage 1 Top-K を取得
        s1_row = sim_stage1[i].copy()
        for j in range(n):
            if j == i or (isrcs[i] and isrcs[j] == isrcs[i]):
                s1_row[j] = -1e9
        cands = np.argpartition(-s1_row, top_k_cand)[:top_k_cand]

        for j in cands:
            if s1_row[j] < -1e8:
                continue
            num = 0.0
            den = 0.0
            for s in slot_names:
                pq = float(p_gates[s][i])
                pc = float(p_gates[s][j])
                sc = float(sim_slots[s][i, j])
                if gate_mode == "prod":
                    w = pq * pc
                    num += w * sc
                    den += w
                elif gate_mode == "qlead":
                    # クエリが重視する軸で、候補もその軸を持ちかつ内容が一致すると高評価
                    num += pq * (pc * sc)
                    den += pq
                elif gate_mode == "qlead_soft":
                    num += pq * ((0.5 + 0.5 * pc) * sc)
                    den += pq
            slot_score = num / (den + 1e-6)
            total = (1.0 - alpha) * float(s1_row[j]) + alpha * slot_score
            if sim_tag is not None:
                total = (1.0 - beta_tag) * total + beta_tag * float(sim_tag[i, j])
            out[i, j] = total
    return out


def eval_rerank_matrix(
    name: str,
    score_mat: np.ndarray,
    base_top3: list[list[int]],
    isrcs: list[str],
    lens: np.ndarray,
    tag_sim_oracle: np.ndarray,
    y_slots_ev: dict[str, np.ndarray],
    masks_ev: dict[str, np.ndarray],
    slot_names: tuple[str, ...],
) -> tuple[dict[str, float], list[list[int]]]:
    n = score_mat.shape[0]
    top3_list: list[list[int]] = []
    oracle_detail_sims: list[float] = []
    oracle_slot_sims: list[float] = []
    len_pairs_q: list[float] = []
    len_pairs_c: list[float] = []
    changed_q = 0
    changed_top1 = 0

    sim_y_slots = {s: normalize_rows(y_slots_ev[s]) @ normalize_rows(y_slots_ev[s]).T for s in slot_names}

    for i in range(n):
        row = score_mat[i].copy()
        for j in range(n):
            if j == i or (isrcs[i] and isrcs[j] == isrcs[i]):
                row[j] = -1e9
        t3 = np.argsort(-row)[:3].tolist()
        top3_list.append(t3)

        if t3 != base_top3[i]:
            changed_q += 1
        if t3[0] != base_top3[i][0]:
            changed_top1 += 1

        for j in t3:
            oracle_detail_sims.append(float(tag_sim_oracle[i, j]))
            len_pairs_q.append(float(lens[i]))
            len_pairs_c.append(float(lens[j]))

            # Oracle スロット別 details 一致度（両方持つスロットの平均）
            s_vals = [
                float(sim_y_slots[s][i, j])
                for s in slot_names
                if masks_ev[s][i] and masks_ev[s][j]
            ]
            if s_vals:
                oracle_slot_sims.append(float(np.mean(s_vals)))

    len_corr = float(np.corrcoef(len_pairs_q, len_pairs_c)[0, 1])
    res = {
        "tag_detail_sim@3": float(np.mean(oracle_detail_sims)),
        "oracle_slot_sim@3": float(np.mean(oracle_slot_sims)),
        "len_corr@3": len_corr,
        "chg_top3%": 100.0 * changed_q / n,
        "chg_top1%": 100.0 * changed_top1 / n,
    }
    print(
        f"{name:<28} | tagDet@3={res['tag_detail_sim@3']:.4f} "
        f"slotDet@3={res['oracle_slot_sim@3']:.4f} "
        f"lenR@3={res['len_corr@3']:.3f} "
        f"chg@3={res['chg_top3%']:5.1f}% chg@1={res['chg_top1%']:5.1f}%"
    )
    return res, top3_list


def main() -> None:
    t0 = time.time()
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
    eval_key_set = {p["key"] for p in eval_data["reviews"]}
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

    train_keys = [
        p["key"]
        for p in official["posts"]
        if p["key"] not in eval_key_set and p["key"] in x_map and p["key"] in records_by_key
    ]
    eval_keys = [
        p["key"]
        for p in eval_data["reviews"]
        if p["key"] in x_map and p["key"] in records_by_key
    ]
    log(f"Train={len(train_keys)}, Eval(unseen)={len(eval_keys)}")

    x_tr_raw = np.stack([x_map[k] for k in train_keys], axis=0)
    x_ev_raw = np.stack([x_map[k] for k in eval_keys], axis=0)
    lens_tr = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)
    lens_ev = np.array([len(post_by_key[k]["content"]) for k in eval_keys], dtype=np.float64)
    isrcs_ev = [post_by_key[k].get("isrc", "") for k in eval_keys]

    # Step 1: 中心化 + Cov-Ortho(len, λ=1.0)（train のみでフィット）
    mean_x = x_tr_raw.mean(axis=0)
    u_len = fit_cov_ortho_eraser(x_tr_raw - mean_x[None, :], lens_tr)
    x_tr = apply_rank1_eraser(x_tr_raw - mean_x[None, :], u_len, 1.0)
    x_ev = apply_rank1_eraser(x_ev_raw - mean_x[None, :], u_len, 1.0)

    # Stage 1: 現行本番モデル rcca_reg8（train のみでフィット）
    y_clean_tr, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)
    w_rcca = fit_rcca_custom(x_tr, y_clean_tr, valid_tr, K, reg_x=8.0, reg_y=4.0, power=1.5)
    z1_ev = x_ev @ w_rcca.T
    z1_ev_u = normalize_rows(z1_ev)
    sim_stage1 = z1_ev_u @ z1_ev_u.T

    # 4スロット & 5スロットの正解 details ベクトル構築（train & eval）
    y4_tr, m4_tr, pr4_tr = build_slot_detail_vectors(train_keys, records_by_key, text_vecs, SLOT4_TAGS)
    y4_ev, m4_ev, pr4_ev = build_slot_detail_vectors(eval_keys, records_by_key, text_vecs, SLOT4_TAGS)

    y5_tr, m5_tr, pr5_tr = build_slot_detail_vectors(train_keys, records_by_key, text_vecs, SLOT5_TAGS)
    y5_ev, m5_ev, pr5_ev = build_slot_detail_vectors(eval_keys, records_by_key, text_vecs, SLOT5_TAGS)

    log("4スロット rCCA (d=32) + mask/primary ゲート学習中...")
    w4_32, g4_mask, g4_prim = fit_slot_models(
        x_tr, y4_tr, m4_tr, pr4_tr, SLOTS_4, slot_dim=32, reg_x=12.0, reg_y=4.0, power=1.5
    )
    log("5スロット rCCA (d=32) + mask/primary ゲート学習中...")
    w5_32, g5_mask, g5_prim = fit_slot_models(
        x_tr, y5_tr, m5_tr, pr5_tr, SLOTS_5, slot_dim=32, reg_x=12.0, reg_y=4.0, power=1.5
    )

    # 23タグ分類器
    tag_tr = build_tag_matrix(train_keys, records_by_key)
    tag_clfs = fit_tag_classifiers(x_tr, tag_tr)
    p_tags_ev = predict_tag_probs(x_ev, tag_clfs)

    # 1. primary ゲートの分布と AUROC 確認
    print()
    log("=== 1. primary ゲートの陽性率・AUROC・平均確率（unseen 700件） ===")
    x_ev_u = normalize_rows(x_ev)
    p4_mask_ev = {s: g4_mask[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_4}
    p4_prim_ev = {s: g4_prim[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_4}
    p4_prim2_ev = {s: p4_prim_ev[s] ** 2 for s in SLOTS_4}

    p5_mask_ev = {s: g5_mask[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_5}
    p5_prim_ev = {s: g5_prim[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_5}
    p5_prim2_ev = {s: p5_prim_ev[s] ** 2 for s in SLOTS_5}

    print(f"{'slot':<10} | {'mask%':>6} {'mAUC':>6} {'mMean':>6} | {'prim%':>6} {'pAUC':>6} {'pMean':>6} {'pStd':>6}")
    print("-" * 68)
    for s in SLOTS_4:
        m_auc = roc_auc_score(m4_ev[s].astype(int), p4_mask_ev[s])
        p_auc = roc_auc_score(pr4_ev[s].astype(int), p4_prim_ev[s])
        print(
            f"{s:<10} | {m4_ev[s].mean()*100:5.1f}% {m_auc:6.3f} {p4_mask_ev[s].mean():6.3f} | "
            f"{pr4_ev[s].mean()*100:5.1f}% {p_auc:6.3f} {p4_prim_ev[s].mean():6.3f} {p4_prim_ev[s].std():6.3f}"
        )
    print("-" * 68)
    for s in SLOTS_5:
        m_auc = roc_auc_score(m5_ev[s].astype(int), p5_mask_ev[s])
        p_auc = roc_auc_score(pr5_ev[s].astype(int), p5_prim_ev[s])
        print(
            f"{s:<10} | {m5_ev[s].mean()*100:5.1f}% {m_auc:6.3f} {p5_mask_ev[s].mean():6.3f} | "
            f"{pr5_ev[s].mean()*100:5.1f}% {p_auc:6.3f} {p5_prim_ev[s].mean():6.3f} {p5_prim_ev[s].std():6.3f}"
        )

    # 2. Top-20 → Top-3 Reranking の定量比較
    print()
    log("=== 2. Stage-2 Reranking 定量比較（unseen 700件、Top-20 → Top-3） ===")
    sim_by_block, _ = build_tag_matched_similarity(
        eval_keys, records_by_key, text_vecs, torch.device("cpu")
    )
    tag_sim_oracle = sum(sim_by_block[b] for b in SUP_BLOCKS)

    # Stage 1 単体の Top-3 を基準に取得
    base_top3: list[list[int]] = []
    for i in range(len(eval_keys)):
        row = sim_stage1[i].copy()
        for j in range(len(eval_keys)):
            if j == i or (isrcs_ev[i] and isrcs_ev[j] == isrcs_ev[i]):
                row[j] = -1e9
        base_top3.append(np.argsort(-row)[:3].tolist())

    eval_rerank_matrix(
        "Stage1: rcca_reg8 (現行)",
        sim_stage1,
        base_top3,
        isrcs_ev,
        lens_ev,
        tag_sim_oracle,
        y4_ev,
        m4_ev,
        SLOTS_4,
    )

    z4_32_ev_u = {s: normalize_rows(x_ev @ w4_32[s].T) for s in SLOTS_4}
    z5_32_ev_u = {s: normalize_rows(x_ev @ w5_32[s].T) for s in SLOTS_5}

    variants = [
        ("4slot mask prod α=0.35", z4_32_ev_u, p4_mask_ev, SLOTS_4, 0.35, "prod", None, 0.0),
        ("4slot prim prod α=0.20", z4_32_ev_u, p4_prim_ev, SLOTS_4, 0.20, "prod", None, 0.0),
        ("4slot prim prod α=0.35", z4_32_ev_u, p4_prim_ev, SLOTS_4, 0.35, "prod", None, 0.0),
        ("4slot prim^2 prod α=0.35", z4_32_ev_u, p4_prim2_ev, SLOTS_4, 0.35, "prod", None, 0.0),
        ("4slot prim qlead α=0.35", z4_32_ev_u, p4_prim_ev, SLOTS_4, 0.35, "qlead", None, 0.0),
        ("5slot prim prod α=0.25", z5_32_ev_u, p5_prim_ev, SLOTS_5, 0.25, "prod", None, 0.0),
        ("5slot prim prod α=0.35", z5_32_ev_u, p5_prim_ev, SLOTS_5, 0.35, "prod", None, 0.0),
        ("5slot prim^2 prod α=0.35", z5_32_ev_u, p5_prim2_ev, SLOTS_5, 0.35, "prod", None, 0.0),
        ("5slot prim qlead α=0.35", z5_32_ev_u, p5_prim_ev, SLOTS_5, 0.35, "qlead", None, 0.0),
        ("5slot prim+tag α=0.30 β=0.10", z5_32_ev_u, p5_prim_ev, SLOTS_5, 0.30, "prod", p_tags_ev, 0.10),
    ]

    saved_top3: dict[str, list[list[int]]] = {}
    for vname, z_u, pg, snames, alpha, gmode, pt, btag in variants:
        smat = compute_reranked_scores(
            sim_stage1,
            z_u,
            pg,
            snames,
            isrcs_ev,
            top_k_cand=20,
            alpha=alpha,
            gate_mode=gmode,
            p_tags=pt,
            beta_tag=btag,
        )
        _, t3 = eval_rerank_matrix(
            vname, smat, base_top3, isrcs_ev, lens_ev, tag_sim_oracle, y4_ev, m4_ev, SLOTS_4
        )
        saved_top3[vname] = t3

    # 3. 先ほどの5クエリ（#10, #19, #22, #28, #32）の primary ゲート値と新#1 を確認
    print()
    log("=== 3. 注目クエリ (#10, #19, #22, #28, #32) の mask vs primary 比較 ===")
    t3_mask = saved_top3["4slot mask prod α=0.35"]
    t3_p4 = saved_top3["4slot prim prod α=0.35"]
    t3_p5 = saved_top3["5slot prim prod α=0.35"]
    for i in [10, 19, 22, 28, 32]:
        q_post = records_by_key[eval_keys[i]]
        b0 = records_by_key[eval_keys[base_top3[i][0]]]
        m0 = records_by_key[eval_keys[t3_mask[i][0]]]
        p4_0 = records_by_key[eval_keys[t3_p4[i][0]]]
        p5_0 = records_by_key[eval_keys[t3_p5[i][0]]]
        mg_str = " ".join(f"{s}={p4_mask_ev[s][i]:.2f}" for s in SLOTS_4)
        pg_str = " ".join(f"{s}={p4_prim_ev[s][i]:.2f}" for s in SLOTS_4)
        p5g_str = " ".join(f"{s}={p5_prim_ev[s][i]:.2f}" for s in SLOTS_5)
        print(f"\n[Query #{i}] {q_post['title']} / {q_post['artistName']}")
        print(f"  mask4 : {mg_str}")
        print(f"  prim4 : {pg_str}")
        print(f"  prim5 : {p5g_str}")
        print(f"  本文  : {q_post['content'][:100]}")
        print(f"  Stage1: {b0['title']} / {b0['artistName']} — {b0['content'][:80]}")
        print(f"  mask4 : {m0['title']} / {m0['artistName']} — {m0['content'][:80]}")
        print(f"  prim4 : {p4_0['title']} / {p4_0['artistName']} — {p4_0['content'][:80]}")
        print(f"  prim5 : {p5_0['title']} / {p5_0['artistName']} — {p5_0['content'][:80]}")

    log(f"完了 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
