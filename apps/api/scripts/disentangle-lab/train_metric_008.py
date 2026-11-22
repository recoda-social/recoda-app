#!/usr/bin/env python3
"""
StrictResonanceJudge（0〜3 点）で採点した学習 8,000 ペア（非評価 800 クエリ × 10 候補）を使い、
「強共鳴（3点）を押し上げ、ハズレ（0〜1点）を押し下げる」Metric Learning を実行する。

比較モデル（すべて完全未見 700 件＋53 クエリでホールドアウト評価）:
  1. baseline      : 現行 PCA 256
  2. rcca_reg8     : 強収縮 rCCA（ペア教師なしの最良線形）
  3. metric_linear : 純線形 z = Ax + b（rcca_reg8 初期化＋Listwise Graded KL ＋ Hard-Negative Margin）
  4. metric_mlp    : 軽量残差 MLP z = Ax + b + W2·GELU(W1 x̃)（Turso 256次元 F16 インデックス変更なし）
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
import torch.nn as nn
import torch.nn.functional as F

from prepare_3pillar_candidates import fit_rcca_custom
from prepare_nonridge_judge import build_clean_detail_vectors
from train_eval_007 import (
    K,
    apply_rank1_eraser,
    build_tag_matched_similarity,
    build_tag_matrix,
    eval_all_metrics,
    fit_cov_ortho_eraser,
    log,
    normalize_rows,
    top_pca_basis,
)

# 3 点を最も強く引き上げ、0〜1 点を抑える段階的重み
GRADE_VALUE = {0: 0.0, 1: 0.15, 2: 0.65, 3: 1.5}


class LinearAnchorModel(nn.Module):
    def __init__(self, w_anchor: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("w_anchor", torch.tensor(w_anchor, dtype=torch.float32))
        self.delta = nn.Parameter(torch.zeros_like(self.w_anchor))

    def forward(self, x_sub: torch.Tensor) -> torch.Tensor:
        w = self.w_anchor + self.delta
        return x_sub @ w.T

    def reg_loss(self) -> torch.Tensor:
        return self.delta.pow(2).sum() / self.w_anchor.shape[0]


class ResidualMlpModel(nn.Module):
    def __init__(self, w_anchor: np.ndarray, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        k_out, sub_dim = w_anchor.shape
        self.register_buffer("w_anchor", torch.tensor(w_anchor, dtype=torch.float32))
        self.delta = nn.Parameter(torch.zeros_like(self.w_anchor))
        self.fc1 = nn.Linear(sub_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, k_out)
        # 初期状態は厳密に rcca_reg8 と一致させる
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x_sub: torch.Tensor) -> torch.Tensor:
        base = x_sub @ (self.w_anchor + self.delta).T
        res = self.fc2(self.drop(F.gelu(self.fc1(x_sub))))
        return base + res

    def reg_loss(self) -> torch.Tensor:
        l_lin = self.delta.pow(2).sum() / self.w_anchor.shape[0]
        l_mlp = (self.fc1.weight.pow(2).sum() + self.fc2.weight.pow(2).sum()) / self.w_anchor.shape[0]
        return l_lin + 0.5 * l_mlp


def compute_batch_loss(
    model: nn.Module,
    x_all: torch.Tensor,
    query_groups: list[tuple[int, list[int], list[int]]],
    tau_q: float = 0.08,
    tau_p: float = 0.45,
    margin_weight: float = 1.0,
    n_rand_neg: int = 64,
) -> tuple[torch.Tensor, float]:
    """
    フルプール幾何を守るため、各クエリの 10 件（ハード正負例）に加えて
    プール全体からランダム負例 n_rand_neg 件（0点）を同時に入れて
    Listwise Graded KL + Hard-Negative Pairwise Margin Loss を計算する。
    """
    z_all = F.normalize(model(x_all), dim=-1, eps=1e-6)
    n_pool = z_all.shape[0]
    loss_list = torch.tensor(0.0, device=x_all.device)
    loss_margin = torch.tensor(0.0, device=x_all.device)
    n_list = 0
    n_margin = 0
    top3_scores: list[float] = []

    for qi, d_indices, r_scores in query_groups:
        zq = z_all[qi]
        zd = z_all[d_indices]
        cos_mined = zd @ zq  # (m,)

        # 内部検証用: 採点済み 10 件内での予測上位 3 件の平均教師スコア
        with torch.no_grad():
            order = torch.argsort(cos_mined, descending=True)[:3].tolist()
            top3_scores.append(sum(r_scores[idx] for idx in order) / max(len(order), 1))

        if max(r_scores) == min(r_scores):
            continue

        # 1. Listwise Graded KL（採点済み 10 件 + ランダム負例 n_rand_neg 件）
        if n_rand_neg > 0:
            rand_idx = torch.randint(0, n_pool, (n_rand_neg,), device=x_all.device)
            cos_rand = z_all[rand_idx] @ zq
            cos_all = torch.cat([cos_mined, cos_rand], dim=0)
            g_list = [GRADE_VALUE[r] for r in r_scores] + [0.0] * n_rand_neg
        else:
            cos_all = cos_mined
            g_list = [GRADE_VALUE[r] for r in r_scores]

        g_t = torch.tensor(g_list, dtype=torch.float32, device=x_all.device)
        p_t = F.softmax(g_t / tau_p, dim=0)
        log_q = F.log_softmax(cos_all / tau_q, dim=0)
        loss_list = loss_list + F.kl_div(log_q, p_t, reduction="sum")
        n_list += 1

        # 2. Hard-Negative Pairwise Margin（採点済み 10 件内）
        m_len = len(r_scores)
        for a in range(m_len):
            for b in range(m_len):
                if r_scores[a] > r_scores[b]:
                    diff_r = r_scores[a] - r_scores[b]
                    margin = 0.05 * diff_r
                    w = 1.0
                    if r_scores[b] == 0:
                        w *= 2.0
                    if r_scores[a] == 3:
                        w *= 1.5
                    loss_margin = loss_margin + w * F.relu(margin - (cos_mined[a] - cos_mined[b]))
                    n_margin += 1

    total = (loss_list / max(n_list, 1)) + margin_weight * (loss_margin / max(n_margin, 1))
    mean_top3 = float(np.mean(top3_scores)) if top3_scores else 0.0
    return total, mean_top3


def train_model(
    model: nn.Module,
    x_tr_sub: torch.Tensor,
    tr_groups: list[tuple[int, list[int], list[int]]],
    val_groups: list[tuple[int, list[int], list[int]]],
    anchor_reg: float = 0.08,
    lr: float = 2e-3,
    epochs: int = 120,
    label: str = "model",
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    best_val = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Epoch 0 (初期値 rcca_reg8) の内部検証スコア
    model.eval()
    with torch.no_grad():
        val_loss0, val_top3_0 = compute_batch_loss(model, x_tr_sub, val_groups)
    best_val = val_top3_0
    log(f"  [{label}] init(ep=0): val_loss={val_loss0.item():.4f} val_top3_score={val_top3_0:.4f}")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        loss_rank, _ = compute_batch_loss(model, x_tr_sub, tr_groups)
        loss_reg = model.reg_loss()  # type: ignore[operator]
        total = loss_rank + anchor_reg * loss_reg
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v_loss, v_top3 = compute_batch_loss(model, x_tr_sub, val_groups)
            if v_top3 > best_val + 1e-5:
                best_val = v_top3
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep % 30 == 0 or ep == epochs:
                log(
                    f"  [{label}] ep={ep:3d}/{epochs} tr_loss={loss_rank.item():.4f} "
                    f"reg={loss_reg.item():.4f} val_loss={v_loss.item():.4f} "
                    f"val_top3={v_top3:.4f} (best={best_val:.4f}) ({time.time()-t0:.1f}s)"
                )

    model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
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
    train_key_to_idx = {k: i for i, k in enumerate(train_keys)}
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

    # 中心化 & Step 1 Cov-Ortho
    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c, x_ev_c, x_q_c = x_train - mean_x, x_eval - mean_x, query_xs - mean_x
    u_pca = top_pca_basis((x_tr_c.T @ x_tr_c) / len(x_tr_c), K)

    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)

    # アンカー: rcca_reg8
    y_clean_tr, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)
    w_rcca = fit_rcca_custom(x_tr_clean, y_clean_tr, valid_tr, K, reg_x=8.0, reg_y=4.0, power=1.5)

    # 上位 384 主成分部分空間へ射影（過学習防止）
    sub_dim = 384
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)
    u_sub = top_pca_basis(cov_clean, sub_dim)  # (384, D)
    w_anchor = w_rcca @ u_sub.T  # (256, 384)

    x_tr_sub = torch.tensor(x_tr_clean @ u_sub.T, dtype=torch.float32, device=dev)
    x_ev_sub = torch.tensor(x_ev_clean @ u_sub.T, dtype=torch.float32, device=dev)
    x_q_sub = torch.tensor(x_q_clean @ u_sub.T, dtype=torch.float32, device=dev)

    # 採点済み学習ペアの読み込み（800 クエリ → 640 train / 160 internal val）
    by_q: dict[int, list[tuple[int, int]]] = {}
    with (out_dir / "train_pairs_scored.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qi = train_key_to_idx.get(rec["q_key"])
            di = train_key_to_idx.get(rec["d_key"])
            if qi is not None and di is not None:
                by_q.setdefault(qi, []).append((di, int(rec["score"])))

    all_groups = [
        (qi, [d for d, _ in pairs], [s for _, s in pairs])
        for qi, pairs in sorted(by_q.items())
        if len(pairs) >= 2
    ]
    rng = random.Random(42)
    rng.shuffle(all_groups)
    split = int(len(all_groups) * 0.8)
    tr_groups = all_groups[:split]
    val_groups = all_groups[split:]
    log(f"Metric Learning グループ: 学習={len(tr_groups)} クエリ / 内部検証={len(val_groups)} クエリ")

    # 1. 純線形 Metric Learning (z = Ax + b, フルプール負例 + 強アンカー正則化)
    log("Training metric_linear (z = Ax + b, in-pool neg=64, anchor_reg=5.0)...")
    lin_model = LinearAnchorModel(w_anchor).to(dev)
    lin_model = train_model(
        lin_model, x_tr_sub, tr_groups, val_groups, anchor_reg=5.0, lr=1e-3, epochs=60, label="metric_linear"
    )

    # 2. 軽量残差 MLP (z = Ax + b + W2·GELU(W1 x̃))
    log("Training metric_mlp (residual 1-hidden MLP, in-pool neg=64, anchor_reg=5.0)...")
    mlp_model = ResidualMlpModel(w_anchor, hidden_dim=256, dropout=0.3).to(dev)
    mlp_model = train_model(
        mlp_model, x_tr_sub, tr_groups, val_groups, anchor_reg=5.0, lr=1e-3, epochs=60, label="metric_mlp"
    )

    with torch.no_grad():
        ze_lin = lin_model(x_ev_sub).cpu().numpy().astype(np.float64)
        zq_lin = lin_model(x_q_sub).cpu().numpy().astype(np.float64)
        ze_mlp = mlp_model(x_ev_sub).cpu().numpy().astype(np.float64)
        zq_mlp = mlp_model(x_q_sub).cpu().numpy().astype(np.float64)

    cands = {
        "baseline": (x_ev_c @ u_pca.T, x_q_c @ u_pca.T),
        "rcca_reg8": (x_ev_clean @ w_rcca.T, x_q_clean @ w_rcca.T),
        "metric_linear": (ze_lin, zq_lin),
        "metric_mlp": (ze_mlp, zq_mlp),
    }

    print("\n=== Metric Learning (008) 定量指標（完全未見 700 件＋53 クエリ） ===")
    print(f"{'候補':<16} {'asp@3':>6} {'mod@3':>6} {'言葉@10':>7} {'len@3':>6} | {'S@3Avg':>6} {'nDCG@3':>6}")
    print("-" * 68)
    for name, (ze, zq) in cands.items():
        m = eval_all_metrics(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat)
        print(
            f"{name:<16} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
            f"{m['text@10']*100:6.1f}% {m['lenCorr@3']:6.2f} | {m['S@3_macro']:6.3f} {m['nDCG@3_macro']:6.3f}"
        )

    # 同一 150 件の未見評価クエリで StrictResonanceJudge 用候補を保存
    rng_ev = random.Random(42)
    sample_indices = sorted(rng_ev.sample(range(len(eval_reviews)), 150))

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

    out_path = out_dir / "judge_candidates_008.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"評価候補データ保存完了: {out_path} ({len(items)} 件 × 4 手法)")


if __name__ == "__main__":
    main()
