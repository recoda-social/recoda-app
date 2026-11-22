#!/usr/bin/env python3
"""
Ridge / 23タグボトルネックを一切使わず、「ノイズ除去済み details 埋め込み y_clean」の
連続的な意味空間そのものを 256 次元線形変換 z = Ax + b へ写像する 4 手法を構築・評価する。

比較手法（すべて Step 1 Cov-Ortho 文字数除去済み x̃ に対して適用、純線形 z = Ax + b）:
  1. baseline         : 現行 PCA 256
  2. clean_pls        : PLS-SVD（Cov(x̃, y_clean) の上位 256 左特異ベクトル × σ^0.5）
  3. clean_procrustes : 直交 Procrustes / Löwdin（Cov(x̃, y_clean) の上位 256 正規直交左特異ベクトル、σ^0）
  4. clean_rcca       : 正則化 CCA（x̃ と y_clean の正準相関方向 256 本 × ρ_k）
  5. clean_rcd        : 256 次元 Relational Cosine Distillation（cos(y_clean) の順位分布を直接 A へ蒸留）
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from train_eval_007 import (
    D,
    K,
    apply_rank1_eraser,
    build_tag_matched_similarity,
    build_tag_matrix,
    eval_all_metrics,
    extract_tag_entries,
    fit_cov_ortho_eraser,
    log,
    normalize_rows,
    top_pca_basis,
)


def build_clean_detail_vectors(
    keys: list[str],
    records_by_key: dict[str, dict[str, Any]],
    text_vecs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    各投稿 i のノイズ除去済み details 埋め込み y_clean_i = normalize(sum_{t in i} w_{i,t} * e_{i,t}) を構築。
    """
    n = len(keys)
    y_mat = np.zeros((n, D), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, k in enumerate(keys):
        entries = extract_tag_entries(records_by_key[k])
        acc = np.zeros(D, dtype=np.float64)
        for text, w in entries.values():
            v = text_vecs.get(text)
            if v is not None:
                vn = float(np.linalg.norm(v))
                if vn > 1e-12:
                    acc += w * (v.astype(np.float64) / vn)
        norm = float(np.linalg.norm(acc))
        if norm > 1e-12:
            y_mat[i] = acc / norm
            valid[i] = True
    return y_mat, valid


def fit_clean_pls_and_procrustes(
    x_tr: np.ndarray, y_tr: np.ndarray, valid: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    クロス共分散 C_XY = (1/n) X_c^T Y_c の SVD から:
      - clean_pls        : U_k^T * diag(s_k^0.5)
      - clean_procrustes : U_k^T（正規直交射影 / Löwdin 相当）
    """
    xv = x_tr[valid]
    yv = y_tr[valid]
    xc = xv - xv.mean(axis=0, keepdims=True)
    yc = yv - yv.mean(axis=0, keepdims=True)
    cxy = (xc.T @ yc) / len(xv)  # (D, D)
    u, s, _ = np.linalg.svd(cxy, full_matrices=False)
    u_k = u[:, :k].T.copy()  # (k, D)
    s_k = np.maximum(s[:k], 1e-12)
    w_pls = u_k * np.sqrt(s_k)[:, None]
    w_proc = u_k.copy()
    return w_pls, w_proc


def fit_clean_rcca(
    x_tr: np.ndarray, y_tr: np.ndarray, valid: np.ndarray, k: int, reg_x: float = 50.0, reg_y: float = 0.05
) -> np.ndarray:
    """
    正則化 CCA（閉形式）: x̃ と y_clean の共有方向上位 k 本（正準相関重み付き）。
    ※ x は未正規化（分散 ~数千）、y は単位正規化（分散 ~1/2048）なので各共分散トレース比に合わせて reg を設定。
    """
    xv = x_tr[valid]
    yv = y_tr[valid]
    xc = xv - xv.mean(axis=0, keepdims=True)
    yc = yv - yv.mean(axis=0, keepdims=True)
    n = len(xv)
    cxx = (xc.T @ xc) / n
    cyy = (yc.T @ yc) / n
    cxy = (xc.T @ yc) / n

    # スケール不変な収縮正則化（平均固有値の一定倍）
    tx = float(np.trace(cxx)) / D
    ty = float(np.trace(cyy)) / D
    ex, vx = np.linalg.eigh(cxx + (reg_x * tx) * np.eye(D))
    ey, vy = np.linalg.eigh(cyy + (reg_y * ty) * np.eye(D))
    ix = (vx * (1.0 / np.sqrt(np.maximum(ex, 1e-12)))) @ vx.T
    iy = (vy * (1.0 / np.sqrt(np.maximum(ey, 1e-12)))) @ vy.T

    m_mat = ix @ cxy @ iy
    u_svd, s_svd, _ = np.linalg.svd(m_mat, full_matrices=False)
    w_x = (ix @ u_svd[:, :k]) * s_svd[:k]  # (D, k)
    return w_x.T.copy()


def fit_clean_rcd_256(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    valid: np.ndarray,
    k: int,
    tau: float = 0.07,
    alpha_l2: float = 0.02,
    sub_dim: int = 384,
    steps: int = 350,
    lr: float = 5e-3,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    y_clean 同士のコサイン順位分布を 256 次元線形行列 A へ直接 Relational Cosine Distillation。
    過学習防止のため上位 sub_dim PCA 部分空間内で最適化し、PCA 256 を初期値にする。
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xv = x_tr[valid]
    yv = y_tr[valid]
    n = len(xv)

    cov_x = (xv.T @ xv) / n
    u_sub = top_pca_basis(cov_x, sub_dim)  # (sub_dim, D)
    x_sub = xv @ u_sub.T  # (n, sub_dim)

    x_t = torch.tensor(x_sub, dtype=torch.float32, device=dev)
    y_t = torch.tensor(yv, dtype=torch.float32, device=dev)
    diag = torch.eye(n, dtype=torch.bool, device=dev)

    sim_y = (y_t @ y_t.T).masked_fill(diag, -1e9)
    p_teacher = F.softmax(sim_y / tau, dim=-1)

    # 初期値: 上位 k PCA（単位射影）から開始して y_clean 方向へ滑らかに変形
    w0 = np.zeros((k, sub_dim), dtype=np.float32)
    for i in range(k):
        w0[i, i] = 1.0
    w_param = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW([w_param], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    t0 = time.time()
    for step in range(1, steps + 1):
        z = x_t @ w_param.T
        zn = F.normalize(z, dim=-1, eps=1e-6)
        sim_z = (zn @ zn.T).masked_fill(diag, -1e9)
        log_q = F.log_softmax(sim_z / tau, dim=-1)
        kl = F.kl_div(log_q, p_teacher, reduction="batchmean")
        # 初期値（PCA）からの逸脱に対する L2 正則化（PLaMo の良い構造を壊しすぎない）
        l2 = (w_param - torch.tensor(w0, device=dev)).pow(2).sum() / k
        loss = kl + alpha_l2 * l2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step == 1 or step % 100 == 0 or step == steps:
            log(f"  [CleanRCD step {step:3d}/{steps}] kl={kl.item():.4f} l2={l2.item():.4f} ({time.time() - t0:.1f}s)")

    return (w_param.detach().cpu().numpy().astype(np.float64) @ u_sub)


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

    log("ノイズ除去済み details ベクトル y_clean を構築中...")
    y_clean_tr, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)
    log(f"有効な y_clean: {int(valid_tr.sum())}/{len(valid_tr)} 件")

    log("Fitting CleanDetails PLS-SVD & Procrustes/Löwdin...")
    w_pls, w_proc = fit_clean_pls_and_procrustes(x_tr_clean, y_clean_tr, valid_tr, K)

    log("Fitting CleanDetails rCCA...")
    w_rcca = fit_clean_rcca(x_tr_clean, y_clean_tr, valid_tr, K, reg_x=2.0, reg_y=2.0)

    log("Fitting CleanDetails RCD-256...")
    w_rcd = fit_clean_rcd_256(x_tr_clean, y_clean_tr, valid_tr, K, tau=0.07, alpha_l2=0.05, sub_dim=384, steps=350, device=dev)

    cands = {
        "baseline": (x_ev_c @ u_pca.T, x_q_c @ u_pca.T),
        "clean_pls": (x_ev_clean @ w_pls.T, x_q_clean @ w_pls.T),
        "clean_procrustes": (x_ev_clean @ w_proc.T, x_q_clean @ w_proc.T),
        "clean_rcca": (x_ev_clean @ w_rcca.T, x_q_clean @ w_rcca.T),
        "clean_rcd": (x_ev_clean @ w_rcd.T, x_q_clean @ w_rcd.T),
    }

    print("\n=== Ridge 不使用 4 手法の定量指標（完全未見 700 件＋53 クエリ） ===")
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

    out_path = out_dir / "judge_candidates_nonridge.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"候補データ保存完了: {out_path} ({len(items)} 件 × 5 手法)")


if __name__ == "__main__":
    main()
