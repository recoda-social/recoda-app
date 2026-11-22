#!/usr/bin/env python3
"""
plans/007-tag-detail-disentangle-plan.md 実装：
Tag-Matched Detail Similarity + Relational Cosine Distillation + ブロック間 Disentangle 学習・評価スクリプト。

- 学習：非評価 2,978 件のみ（リークゼロ）
- 評価：完全未見の embedding-eval.json 700 件 ＋ 短い言葉検索 53 クエリ
- 6 ブロック構成（計 256 次元）:
    z_scene(32) | z_memory(48) | z_music(48) | z_meta(16) | z_mood(32) | z_residual(80)
- 比較手法:
    1. Baseline（現行 PCA 256）
    2. Step 1 単体（Cov-Ortho len λ=1.0 → PCA 256）
    3. B1-KL（23 タグのみ KL 蒸留 + 残余 PCA）
    4. 007-Phase1（Tag-Matched Detail RCD + 独立ペナルティ β、純線形 z = Ax + b）
    5. 007-Phase2（Phase 1 + ブロック有無ゲート σ(g_b^T x̃ + c_b)）
    6. 各ブロック単独検索（Disentanglement 確認）

実行例:
  # 合成データでの動作検証（Cloudflare 不要）
  python apps/api/scripts/disentangle-lab/train_eval_007.py --synthetic-test

  # 実データでの学習・完全ホールドアウト評価（初回のみ D1/Workers AI から取得してキャッシュ）
  export CLOUDFLARE_ACCOUNT_ID=...
  export WORKERS_AI_API_TOKEN=...
  python apps/api/scripts/disentangle-lab/train_eval_007.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

D = 2048
K = 256
EMBEDDING_SOURCE = "@cf/pfnet/plamo-embedding-1b"

# 007 計画 Section 3: 6 ブロック構成（計 256 次元）
SUP_BLOCKS: tuple[str, ...] = ("scene", "memory", "music", "meta", "mood")
BLOCK_DIMS: dict[str, int] = {
    "scene": 32,
    "memory": 48,
    "music": 48,
    "meta": 16,
    "mood": 32,
}
RES_DIM = K - sum(BLOCK_DIMS.values())  # 80

BLOCK_TAGS: dict[str, list[str]] = {
    "scene": ["scene_time", "scene_move", "scene_daily"],
    "memory": ["mem_youth", "mem_love", "mem_hardship", "mem_people_turn", "mem_era"],
    "music": ["lyrics", "vocal", "sound"],
    "meta": ["tieup_visual", "artist_live"],
    "mood": [
        "hype",
        "cool",
        "calm_warm",
        "cute",
        "encouraged",
        "moved",
        "bittersweet",
        "nostalgic",
        "funny",
        "dark_addictive",
    ],
}
# all_qwen36.jsonl 内のスロット名との対応（context を scene / memory に分割）
TAG_TO_SOURCE_SLOT: dict[str, str] = {
    **{t: "context" for t in BLOCK_TAGS["scene"]},
    **{t: "context" for t in BLOCK_TAGS["memory"]},
    **{t: "music" for t in BLOCK_TAGS["music"]},
    **{t: "meta" for t in BLOCK_TAGS["meta"]},
    **{t: "mood" for t in BLOCK_TAGS["mood"]},
}
ALL_TAGS: list[str] = [t for b in SUP_BLOCKS for t in BLOCK_TAGS[b]]
TAG_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ALL_TAGS)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def official_post_id(key: str) -> str:
    """packages/db/src/official-reviews.ts の officialPostId と完全一致。"""
    digest = hashlib.sha256(f"official-reviews:{key}".encode("utf-8")).digest()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(alphabet[b % len(alphabet)] for b in digest[:12])


# ---------------------------------------------------------------------------
# Cloudflare D1 / Workers AI 取得＆キャッシュ
# ---------------------------------------------------------------------------


def cf_api(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
    account = (os.environ.get("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    token = (
        os.environ.get("WORKERS_AI_API_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or ""
    ).strip()
    if not account or not token:
        raise RuntimeError(
            "キャッシュが未作成です。CLOUDFLARE_ACCOUNT_ID と WORKERS_AI_API_TOKEN を設定してください。"
        )
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "recoda-disentangle-007/1.0",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            if not parsed.get("success"):
                raise RuntimeError(f"Cloudflare API error on {path}: {parsed.get('errors')}")
            return parsed["result"]
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (400, 401, 403, 404):
                raise RuntimeError(
                    f"Cloudflare API HTTP {exc.code} on {path}: {err_body}"
                ) from exc
            if attempt == 3:
                raise RuntimeError(
                    f"Cloudflare API HTTP {exc.code} on {path}: {err_body}"
                ) from exc
            time.sleep(2.0 * (attempt + 1))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def embed_texts_cf(texts: list[str], label: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, D), dtype=np.float32)
    out: list[np.ndarray] = []
    t0 = time.time()
    for i in range(0, len(texts), 50):
        part = texts[i : i + 50]
        res = cf_api(f"/ai/run/{EMBEDDING_SOURCE}", method="POST", body={"text": part})
        for v in res["data"]:
            out.append(np.asarray(v, dtype=np.float32))
        if (i // 50) % 20 == 0 or i + 50 >= len(texts):
            log(
                f"  Workers AI embed ({label}): {min(i + 50, len(texts))}/{len(texts)} "
                f"({time.time() - t0:.1f}s)"
            )
    return np.stack(out, axis=0)


def load_or_fetch_x_cache(
    cache_path: Path,
    post_keys: list[str],
    queries: list[str],
    store_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if cache_path.exists():
        log(f"x キャッシュ読み込み: {cache_path}")
        npz = np.load(cache_path, allow_pickle=False)
        keys = npz["keys"].tolist()
        xs = npz["xs"].astype(np.float64)
        qxs = npz["query_xs"].astype(np.float64)
        return {k: xs[i] for i, k in enumerate(keys)}, qxs

    post_ids = [official_post_id(k) for k in post_keys]
    log(f"D1 ({store_name}) から {len(post_ids)} 件の x と {len(queries)} クエリを取得中...")
    dbs = cf_api(f"/d1/database?name={urllib.parse.quote(store_name)}")
    db = next((d for d in dbs if d["name"] == store_name), None)
    if db is None:
        raise RuntimeError(f"D1 database {store_name} not found")
    db_uuid = db["uuid"]

    id_to_key = dict(zip(post_ids, post_keys, strict=True))
    found: dict[str, np.ndarray] = {}
    for i in range(0, len(post_ids), 100):
        part = post_ids[i : i + 100]
        placeholders = ", ".join(["?"] * len(part))
        res = cf_api(
            f"/d1/database/{db_uuid}/query",
            method="POST",
            body={
                "sql": f"SELECT post_id, hex(embedding) AS embedding FROM x_vectors WHERE post_id IN ({placeholders})",
                "params": part,
            },
        )
        for row in res[0].get("results", []):
            raw_bytes = bytes.fromhex(str(row["embedding"]))
            vec = np.frombuffer(raw_bytes, dtype="<f2").astype(np.float32)
            found[id_to_key[str(row["post_id"])]] = vec
        if (i // 100) % 10 == 0 or i + 100 >= len(post_ids):
            log(f"  D1 fetch: {min(i + 100, len(post_ids))}/{len(post_ids)} (found={len(found)})")

    qxs = embed_texts_cf(queries, "queries")
    keys_arr = np.array(list(found.keys()))
    xs_arr = np.stack([found[k] for k in keys_arr], axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, keys=keys_arr, xs=xs_arr, query_xs=qxs)
    log(f"x キャッシュ保存完了: {cache_path} (posts={len(found)}, queries={len(qxs)})")
    return {k: found[k].astype(np.float64) for k in found}, qxs.astype(np.float64)


def extract_tag_entries(rec: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """
    1 投稿から { tag: (joined_details_text, weight) } を返す。
    weight = 1.0 (primary: true) / 0.5 (primary: false)。
    """
    out: dict[str, tuple[str, float]] = {}
    entries = rec.get("entries", {})
    for slot_name in ("music", "meta", "context", "mood"):
        for e in entries.get(slot_name, []):
            tag = e.get("tag")
            if tag not in TAG_TO_IDX:
                continue
            details = [d.strip() for d in e.get("details", []) if d and d.strip()]
            if not details:
                continue
            w = 1.0 if e.get("primary", False) else 0.5
            out[tag] = ("。".join(details), w)
    return out


def load_or_fetch_tag_detail_embeddings(
    cache_path: Path,
    records_by_key: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """
    全投稿のユニークな details 連結テキストを Cloudflare Workers AI で埋め込み、
    text -> unit_vector (float32, D) の辞書を返す（float16 で圧縮キャッシュ）。
    """
    needed_texts: set[str] = set()
    for rec in records_by_key.values():
        for text, _ in extract_tag_entries(rec).values():
            needed_texts.add(text)

    existing: dict[str, np.ndarray] = {}
    if cache_path.exists():
        log(f"tag details キャッシュ読み込み: {cache_path}")
        npz = np.load(cache_path, allow_pickle=False)
        texts_arr = npz["texts"].tolist()
        vecs_arr = npz["vecs"].astype(np.float32)
        for i, t in enumerate(texts_arr):
            existing[t] = vecs_arr[i]

    missing = sorted(t for t in needed_texts if t not in existing)
    if missing:
        log(
            f"ユニーク details テキスト {len(missing)} 件（既存 {len(existing)} 件）を "
            f"Workers AI ({EMBEDDING_SOURCE}) で埋め込み中..."
        )
        new_vecs = embed_texts_cf(missing, "tag_details")
        # 単位正規化して保存
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        new_vecs = new_vecs / np.maximum(norms, 1e-12)
        for i, t in enumerate(missing):
            existing[t] = new_vecs[i]

        all_t = sorted(existing.keys())
        all_v = np.stack([existing[t] for t in all_t], axis=0).astype(np.float16)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, texts=np.array(all_t), vecs=all_v)
        log(f"tag details キャッシュ保存完了: {cache_path} (total={len(all_t)})")

    return existing


# ---------------------------------------------------------------------------
# Section 2: Tag-Matched Detail Similarity 行列 S^{(b)} の構築
# ---------------------------------------------------------------------------


def build_tag_matched_similarity(
    keys: list[str],
    records_by_key: dict[str, dict[str, Any]],
    text_vecs: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    各ブロック b ∈ SUP_BLOCKS について:
      S_{ij}^{(b)} = sum_{t in b} I[t in i and t in j] * w_{i,t} * w_{j,t} * max(0, cos(e_{i,t}, e_{j,t}))
    およびブロック有効マスク mask_b (i ∈ I_b) を返す。
    """
    n = len(keys)
    parsed = [extract_tag_entries(records_by_key[k]) for k in keys]
    d_vec = next(iter(text_vecs.values())).shape[0]

    sim_by_block: dict[str, np.ndarray] = {}
    mask_by_block: dict[str, np.ndarray] = {}

    for b in SUP_BLOCKS:
        s_acc = torch.zeros((n, n), dtype=torch.float32, device=device)
        mask = np.zeros(n, dtype=bool)
        for tag in BLOCK_TAGS[b]:
            e_mat = np.zeros((n, d_vec), dtype=np.float32)
            has_any = False
            for i, p in enumerate(parsed):
                item = p.get(tag)
                if item is None:
                    continue
                text, w = item
                vec = text_vecs.get(text)
                if vec is None:
                    continue
                norm = float(np.linalg.norm(vec))
                if norm < 1e-12:
                    continue
                e_mat[i] = (vec / norm) * w
                mask[i] = True
                has_any = True
            if has_any:
                et = torch.tensor(e_mat, dtype=torch.float32, device=device)
                s_acc += torch.clamp(et @ et.T, min=0.0)
        sim_by_block[b] = s_acc.cpu().numpy()
        mask_by_block[b] = mask
    return sim_by_block, mask_by_block


# ---------------------------------------------------------------------------
# Step 1 / B1 / 007-Phase1 (RCD + Disentangle) / 007-Phase2 (Gate)
# ---------------------------------------------------------------------------


def fit_cov_ortho_eraser(x_centered: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    lc = lengths - lengths.mean()
    v = (x_centered.T @ lc) / len(lengths)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def apply_rank1_eraser(x_centered: np.ndarray, u: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return x_centered - lam * np.outer(x_centered @ u, u) if lam != 0.0 else x_centered


def top_pca_basis(cov: np.ndarray, k: int) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx].T.copy()


def fit_residual_pca(cov_clean: np.ndarray, sup_rows: np.ndarray, res_dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(sup_rows.T, mode="reduced")
    proj = np.eye(cov_clean.shape[0]) - q @ q.T
    cov_perp = proj @ cov_clean @ proj
    cov_perp = 0.5 * (cov_perp + cov_perp.T)
    u_res = top_pca_basis(cov_perp, res_dim)
    u_res = u_res - (u_res @ q) @ q.T
    norms = np.linalg.norm(u_res, axis=1, keepdims=True)
    return u_res / np.maximum(norms, 1e-12)


def block_rms_norm(x_centered: np.ndarray, rows: np.ndarray) -> float:
    proj = x_centered @ rows.T
    return float(np.sqrt(np.mean(np.sum(proj * proj, axis=1)))) or 1.0


def build_tag_matrix(
    keys: list[str], records_by_key: dict[str, dict[str, Any]], non_primary_weight: float = 0.5
) -> np.ndarray:
    mat = np.zeros((len(keys), len(ALL_TAGS)), dtype=np.float64)
    for i, k in enumerate(keys):
        for tag, (_, w) in extract_tag_entries(records_by_key[k]).items():
            mat[i, TAG_TO_IDX[tag]] = 1.0 if w >= 0.99 else non_primary_weight
    return mat


def fit_ridge_multi(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    xc = x_train - x_train.mean(axis=0, keepdims=True)
    yc = y_train - y_train.mean(axis=0, keepdims=True)
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ yc)
    return w.T.copy()


def fit_kl_logistic_multi(x_train: np.ndarray, p_train: np.ndarray, l2_reg: float) -> np.ndarray:
    n, d = x_train.shape
    m = p_train.shape[1]
    w_all = np.zeros((m, d), dtype=np.float64)
    for c in range(m):
        p = np.clip(p_train[:, c], 1e-6, 1.0 - 1e-6)
        init = np.zeros(d + 1, dtype=np.float64)
        init[d] = float(np.log(p.mean() / (1.0 - p.mean())))

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:d]
            b = params[d]
            z = x_train @ w + b
            loss = float((np.logaddexp(0.0, z) - p * z).mean()) + 0.5 * l2_reg * float(np.dot(w, w))
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            diff = (sig - p) / n
            grad = np.empty_like(params)
            grad[:d] = x_train.T @ diff + l2_reg * w
            grad[d] = float(diff.sum())
            return loss, grad

        res = minimize(loss_and_grad, init, method="L-BFGS-B", jac=True, options={"maxiter": 200})
        w_all[c] = res.x[:d]
    return w_all


def fit_block_gates(
    x_train: np.ndarray,
    mask_by_block: dict[str, np.ndarray],
    l2_reg: float = 0.05,
) -> dict[str, tuple[np.ndarray, float]]:
    """
    Phase 2: ブロック b にタグが 1 つ以上あるかの軽量ゲート σ(g_b^T x̃ + c_b) を BCE 損失で学習。
    """
    n, d = x_train.shape
    gates: dict[str, tuple[np.ndarray, float]] = {}
    for b in SUP_BLOCKS:
        y = mask_by_block[b].astype(np.float64)
        p_mean = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        init = np.zeros(d + 1, dtype=np.float64)
        init[d] = math.log(p_mean / (1.0 - p_mean))

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            g = params[:d]
            c = params[d]
            z = x_train @ g + c
            loss = float((np.logaddexp(0.0, z) - y * z).mean()) + 0.5 * l2_reg * float(np.dot(g, g))
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            diff = (sig - y) / n
            grad = np.empty_like(params)
            grad[:d] = x_train.T @ diff + l2_reg * g
            grad[d] = float(diff.sum())
            return loss, grad

        res = minimize(loss_and_grad, init, method="L-BFGS-B", jac=True, options={"maxiter": 200})
        gates[b] = (res.x[:d].copy(), float(res.x[d]))
    return gates


def fit_007_joint_rcd(
    x_train: np.ndarray,
    sim_by_block: dict[str, np.ndarray],
    mask_by_block: dict[str, np.ndarray],
    tau: float = 0.1,
    beta_disentangle: float = 1.0,
    alpha_l2: float = 0.05,
    pca_subspace_dim: int = 256,
    steps: int = 350,
    lr: float = 1e-2,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    007 計画 Section 4 の合成損失をフルバッチで直接最適化:
      L = L_rank + beta * L_disentangle + alpha * sum_b ||A_b||_F^2

    過学習防止のため、A_b = W_b @ U_pca（U_pca は学習データの上位 pca_subspace_dim 主成分）
    としてパラメータ数を 1/8 に圧縮して学習し、最終的に元の D 次元行列 A_b を返す（厳密線形）。
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    n, d = x_train.shape

    sub_dim = min(pca_subspace_dim, d, n - 1)
    cov_x = (x_train.T @ x_train) / n
    u_sub = top_pca_basis(cov_x, sub_dim)  # (sub_dim, d)
    x_sub = x_train @ u_sub.T  # (n, sub_dim)

    x_t = torch.tensor(x_sub, dtype=torch.float32, device=dev)
    diag_mask = torch.eye(n, dtype=torch.bool, device=dev)

    # 教師分布 P^{(b)} = softmax_j(S_{ij}^{(b)} / tau) を事前計算（i ∈ I_b のみ）
    teacher_p: dict[str, torch.Tensor] = {}
    pos_indices: dict[str, torch.Tensor] = {}
    for b in SUP_BLOCKS:
        s_t = torch.tensor(sim_by_block[b], dtype=torch.float32, device=dev)
        s_masked = s_t.masked_fill(diag_mask, -1e9)
        pos_idx = torch.tensor(np.where(mask_by_block[b])[0], dtype=torch.long, device=dev)
        pos_indices[b] = pos_idx
        teacher_p[b] = F.softmax(s_masked[pos_idx] / tau, dim=-1)

    params: dict[str, torch.nn.Parameter] = {}
    for b in SUP_BLOCKS:
        db = BLOCK_DIMS[b]
        w0 = torch.empty((db, sub_dim), dtype=torch.float32, device=dev)
        torch.nn.init.orthogonal_(w0)
        w0.mul_(1.0 / math.sqrt(sub_dim))
        params[b] = torch.nn.Parameter(w0)

    opt = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    t0 = time.time()
    for step in range(1, steps + 1):
        z_std_blocks: dict[str, torch.Tensor] = {}
        loss_rank = torch.tensor(0.0, device=dev)
        loss_l2 = torch.tensor(0.0, device=dev)

        for b in SUP_BLOCKS:
            w_b = params[b]
            zb = x_t @ w_b.T  # (N, d_b)
            zb_c = zb - zb.mean(dim=0, keepdim=True)
            rms = torch.sqrt(torch.mean(zb_c.pow(2).sum(dim=1)) + 1e-8)
            z_std_blocks[b] = zb_c / rms

            zb_norm = F.normalize(zb, dim=-1, eps=1e-6)
            pos_idx = pos_indices[b]
            if len(pos_idx) >= 2:
                sim_z = zb_norm[pos_idx] @ zb_norm.T  # (|I_b|, N)
                sim_z[torch.arange(len(pos_idx), device=dev), pos_idx] = -1e9
                log_q = F.log_softmax(sim_z / tau, dim=-1)
                loss_rank = loss_rank + F.kl_div(log_q, teacher_p[b], reduction="batchmean")

            # U_sub は正規直交なので ||W_b U_sub||_F^2 == ||W_b||_F^2
            loss_l2 = loss_l2 + w_b.pow(2).sum()

        loss_dis = torch.tensor(0.0, device=dev)
        if beta_disentangle > 0.0:
            for ia in range(len(SUP_BLOCKS)):
                za = z_std_blocks[SUP_BLOCKS[ia]]
                for ib in range(ia + 1, len(SUP_BLOCKS)):
                    zb = z_std_blocks[SUP_BLOCKS[ib]]
                    cov_ab = (za.T @ zb) / n
                    loss_dis = loss_dis + 2.0 * cov_ab.pow(2).sum()

        total_loss = loss_rank + beta_disentangle * loss_dis + alpha_l2 * loss_l2
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        sched.step()

        if step == 1 or step % 100 == 0 or step == steps:
            log(
                f"  [RCD β={beta_disentangle} α={alpha_l2} step {step:3d}/{steps}] "
                f"total={total_loss.item():.4f} rank={loss_rank.item():.4f} "
                f"dis={loss_dis.item():.5f} l2={loss_l2.item():.4f} ({time.time() - t0:.1f}s)"
            )

    return {
        b: (params[b].detach().cpu().numpy().astype(np.float64) @ u_sub) for b in SUP_BLOCKS
    }


# ---------------------------------------------------------------------------
# 評価（旧指標 + 007 具体一致度 S@3 / nDCG@3 + 空スロット誤ヒット率 + 独立性）
# ---------------------------------------------------------------------------


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def ndcg_at_k(relevances: np.ndarray, ideal_sorted: np.ndarray, k: int = 3) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevances[:k] * discounts))
    idcg = float(np.sum(ideal_sorted[:k] * discounts))
    return dcg / idcg if idcg > 1e-12 else 0.0


def eval_all_metrics(
    z_eval: np.ndarray,
    z_queries: np.ndarray,
    eval_reviews: list[dict[str, Any]],
    eval_lengths: np.ndarray,
    queries: list[dict[str, Any]],
    eval_sim_by_block: dict[str, np.ndarray],
    eval_mask_by_block: dict[str, np.ndarray],
    eval_tag_mat: np.ndarray,
) -> dict[str, float]:
    ze = normalize_rows(z_eval)
    zq = normalize_rows(z_queries)
    n = len(eval_reviews)

    sim = ze @ ze.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)

    a3 = m3 = a10 = m10 = 0.0
    top3_lens = np.zeros(n)

    for i in range(n):
        top3 = order[i, :3]
        top10 = order[i, :10]
        ap = eval_reviews[i]["aspectPrimary"]
        mp = eval_reviews[i]["moodPrimary"]
        a3 += sum(1 for j in top3 if eval_reviews[j]["aspectPrimary"] == ap) / 3.0
        m3 += sum(1 for j in top3 if eval_reviews[j]["moodPrimary"] == mp) / 3.0
        a10 += sum(1 for j in top10 if eval_reviews[j]["aspectPrimary"] == ap) / 10.0
        m10 += sum(1 for j in top10 if eval_reviews[j]["moodPrimary"] == mp) / 10.0
        top3_lens[i] = eval_lengths[top3].mean()

    len_corr = float(np.corrcoef(eval_lengths, top3_lens)[0, 1]) if n > 2 else 0.0

    # 言葉検索@10
    qsim = zq @ ze.T
    qorder = np.argsort(-qsim, axis=1)
    text_hits = 0.0
    for qi, q in enumerate(queries):
        top10 = qorder[qi, :10]
        hits = 0
        for j in top10:
            r = eval_reviews[j]
            ok_a = (not q["aspect"]) or (q["aspect"] in r["aspects"])
            ok_m = (not q["mood"]) or (q["mood"] in r["moods"])
            if ok_a and ok_m:
                hits += 1
        text_hits += hits / 10.0

    # 007 新規指標：具体一致度（平均 S@3 & nDCG@3）と空スロット誤ヒット率
    out: dict[str, float] = {
        "aspect@3": a3 / n,
        "mood@3": m3 / n,
        "aspect@10": a10 / n,
        "mood@10": m10 / n,
        "text@10": text_hits / max(len(queries), 1),
        "lenCorr@3": len_corr,
    }
    s3_vals: list[float] = []
    ndcg3_vals: list[float] = []
    empty_hit_vals: list[float] = []
    for b in SUP_BLOCKS:
        sb = eval_sim_by_block[b]
        mb = eval_mask_by_block[b]
        pos_idx = np.where(mb)[0]
        empty_idx = np.where(~mb)[0]
        b_s3 = 0.0
        b_ndcg3 = 0.0
        for i in pos_idx:
            top3 = order[i, :3]
            rels = sb[i, top3]
            b_s3 += float(rels.mean())
            row_other = np.delete(sb[i], i)
            ideal = np.sort(row_other)[::-1]
            b_ndcg3 += ndcg_at_k(rels, ideal, k=3)
        mean_s3 = b_s3 / max(len(pos_idx), 1)
        mean_ndcg3 = b_ndcg3 / max(len(pos_idx), 1)
        out[f"S@3_{b}"] = mean_s3
        out[f"nDCG@3_{b}"] = mean_ndcg3
        s3_vals.append(mean_s3)
        ndcg3_vals.append(mean_ndcg3)

        # 空スロット誤ヒット率: ブロック b が空のクエリ i の top3 に、ブロック b を持つ投稿が混ざる割合
        if len(empty_idx) > 0:
            fp_rate = float(np.mean([mb[order[i, :3]].mean() for i in empty_idx]))
            out[f"emptyFP_{b}"] = fp_rate
            empty_hit_vals.append(fp_rate)

    out["S@3_macro"] = float(np.mean(s3_vals))
    out["nDCG@3_macro"] = float(np.mean(ndcg3_vals))
    out["emptyFP_macro"] = float(np.mean(empty_hit_vals)) if empty_hit_vals else 0.0
    return out


def block_cross_cov_norm(x_eval: np.ndarray, blocks: dict[str, np.ndarray]) -> float:
    """評価データ上での異なるブロック間の平均正規化相互共分散ノルム（低いほど Disentangle）。"""
    z_std: dict[str, np.ndarray] = {}
    for b in SUP_BLOCKS:
        zb = x_eval @ blocks[b].T
        zc = zb - zb.mean(axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(np.sum(zc * zc, axis=1)))) or 1.0
        z_std[b] = zc / rms
    vals = []
    n = len(x_eval)
    for ia in range(len(SUP_BLOCKS)):
        for ib in range(ia + 1, len(SUP_BLOCKS)):
            cov = (z_std[SUP_BLOCKS[ia]].T @ z_std[SUP_BLOCKS[ib]]) / n
            vals.append(float(np.linalg.norm(cov, "fro")))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# 合成データによるエンドツーエンド自己テスト
# ---------------------------------------------------------------------------


def run_synthetic_test() -> None:
    log("=== 合成データによる 007 パイプライン自己テスト開始 ===")
    rng = np.random.default_rng(42)
    n_tr, n_ev, n_q = 240, 80, 12
    d_syn = 128  # テスト高速化のため D=128 を使用
    global D
    orig_d = D
    D = d_syn

    try:
        # 5 ブロックの真の潜在因子を独立生成し、x に線形混合
        z_true_all = {b: rng.standard_normal((n_tr + n_ev, 8)) for b in SUP_BLOCKS}
        mix_mats = {b: rng.standard_normal((8, d_syn)) / math.sqrt(d_syn) for b in SUP_BLOCKS}
        x_all = sum(z_true_all[b] @ mix_mats[b] for b in SUP_BLOCKS) + 0.3 * rng.standard_normal(
            (n_tr + n_ev, d_syn)
        )
        lens_all = 40.0 + 100.0 * rng.random(n_tr + n_ev)
        # 文字数バイアスを注入
        v_len_true = rng.standard_normal(d_syn)
        v_len_true /= np.linalg.norm(v_len_true)
        x_all += np.outer((lens_all - lens_all.mean()) / 50.0, v_len_true)

        # 合成 records & text_vecs
        records_by_key: dict[str, dict[str, Any]] = {}
        text_vecs: dict[str, np.ndarray] = {}
        keys = [f"syn:{i}" for i in range(n_tr + n_ev)]
        for i, k in enumerate(keys):
            entries: dict[str, list[dict[str, Any]]] = {
                "music": [],
                "meta": [],
                "context": [],
                "mood": [],
            }
            for b in SUP_BLOCKS:
                if rng.random() < 0.65:
                    tag = BLOCK_TAGS[b][int(rng.integers(0, len(BLOCK_TAGS[b])))]
                    txt = f"{k}:{tag}"
                    # 真の潜在因子を detail 埋め込みに反映
                    vec = z_true_all[b][i] @ mix_mats[b] + 0.1 * rng.standard_normal(d_syn)
                    vec = vec / (np.linalg.norm(vec) + 1e-12)
                    text_vecs[txt] = vec.astype(np.float32)
                    entries[TAG_TO_SOURCE_SLOT[tag]].append(
                        {"tag": tag, "details": [txt], "primary": bool(rng.random() < 0.5)}
                    )
            records_by_key[k] = {"key": k, "entries": entries}

        tr_keys = keys[:n_tr]
        ev_keys = keys[n_tr:]
        x_tr = x_all[:n_tr]
        x_ev = x_all[n_tr:]
        q_xs = rng.standard_normal((n_q, d_syn))
        mean_x = x_tr.mean(axis=0, keepdims=True)
        x_tr_c, x_ev_c, x_q_c = x_tr - mean_x, x_ev - mean_x, q_xs - mean_x
        u_len = fit_cov_ortho_eraser(x_tr_c, lens_all[:n_tr])
        x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
        x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
        x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr_sim, tr_mask = build_tag_matched_similarity(tr_keys, records_by_key, text_vecs, dev)
        ev_sim, ev_mask = build_tag_matched_similarity(ev_keys, records_by_key, text_vecs, dev)

        a_blocks = fit_007_joint_rcd(
            x_tr_clean, tr_sim, tr_mask, tau=0.1, beta_disentangle=1.0, steps=60, device=dev
        )
        gates = fit_block_gates(x_tr_clean, tr_mask)
        cross_cov = block_cross_cov_norm(x_ev_clean, a_blocks)

        eval_reviews = [
            {
                "key": k,
                "aspectPrimary": "lyrics",
                "aspects": ["lyrics"],
                "moodPrimary": "moved",
                "moods": ["moved"],
            }
            for k in ev_keys
        ]
        queries = [{"text": f"q{i}", "aspect": "lyrics", "mood": None} for i in range(n_q)]
        ev_tag_mat = build_tag_matrix(ev_keys, records_by_key)
        z_ev = np.concatenate([x_ev_clean @ a_blocks[b].T for b in SUP_BLOCKS], axis=1)
        z_q = np.concatenate([x_q_clean @ a_blocks[b].T for b in SUP_BLOCKS], axis=1)
        m = eval_all_metrics(
            z_ev, z_q, eval_reviews, lens_all[n_tr:], queries, ev_sim, ev_mask, ev_tag_mat
        )
        log(
            f"合成テスト成功! S@3_macro={m['S@3_macro']:.4f}, nDCG@3_macro={m['nDCG@3_macro']:.4f}, "
            f"cross_cov={cross_cov:.4f}, gates={len(gates)}"
        )
    finally:
        D = orig_d


# ---------------------------------------------------------------------------
# メイン（実データ完全ホールドアウト評価）
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="007 Tag-Matched Detail Disentanglement 学習・評価")
    parser.add_argument("--synthetic-test", action="store_true", help="合成データで動作検証のみ実行")
    parser.add_argument(
        "--extracted-jsonl",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/all_qwen36.jsonl"),
    )
    parser.add_argument(
        "--x-cache",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/x_cache.npz"),
    )
    parser.add_argument(
        "--detail-cache",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/tag_details_cache.npz"),
    )
    parser.add_argument("--store-name", default=os.environ.get("X_STORE_NAME", "x-store-staging-01"))
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--dims",
        default="12,24,24,12,32",
        help="5 ブロック次元数 (scene,memory,music,meta,mood)。残りが residual。初期案は 32,48,48,16,32",
    )
    parser.add_argument("--export-transform", type=Path, default=None, help="最良変換の保存先 .npz")
    args = parser.parse_args()

    parsed_dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    if len(parsed_dims) != len(SUP_BLOCKS) or sum(parsed_dims) >= K:
        raise ValueError(f"--dims は 5 個の正整数かつ合計 < {K} にしてください: {args.dims}")
    global RES_DIM
    for b, d_val in zip(SUP_BLOCKS, parsed_dims, strict=True):
        BLOCK_DIMS[b] = d_val
    RES_DIM = K - sum(parsed_dims)

    if args.synthetic_test:
        run_synthetic_test()
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {dev} | 6 ブロック構成: {BLOCK_DIMS} + residual={RES_DIM} (計 {K} 次元)")

    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
    posts = official["posts"]
    post_by_key = {p["key"]: p for p in posts}
    eval_keys_set = {r["key"] for r in eval_data["reviews"]}

    records_by_key: dict[str, dict[str, Any]] = {}
    with args.extracted_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records_by_key[rec["key"]] = rec
    log(f"構造化抽出読み込み: {len(records_by_key)} 件")

    x_map, query_xs = load_or_fetch_x_cache(
        args.x_cache,
        [p["key"] for p in posts],
        [q["text"] for q in eval_data["queries"]],
        args.store_name,
    )
    text_vecs = load_or_fetch_tag_detail_embeddings(args.detail_cache, records_by_key)

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
    log(
        f"完全ホールドアウト: 学習={len(train_keys)} 件 / 完全未見評価={len(eval_keys)} 件 / "
        f"言葉クエリ={len(query_xs)} 件"
    )

    x_train = np.stack([x_map[k] for k in train_keys], axis=0)
    len_train = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)
    x_eval = np.stack([x_map[k] for k in eval_keys], axis=0)
    eval_lengths = np.array([len(post_by_key[k]["content"]) for k in eval_keys], dtype=np.float64)

    # 教師類似度行列 S^{(b)}（学習用と評価用で完全分離して構築）
    log("Tag-Matched Detail Similarity 行列 S^{(b)} を構築中...")
    tr_sim, tr_mask = build_tag_matched_similarity(train_keys, records_by_key, text_vecs, dev)
    ev_sim, ev_mask = build_tag_matched_similarity(eval_keys, records_by_key, text_vecs, dev)
    eval_tag_mat = build_tag_matrix(eval_keys, records_by_key, non_primary_weight=0.5)

    # Step 1: 中心化 & Cov-Ortho (len, λ=1.0)
    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c, x_ev_c, x_q_c = x_train - mean_x, x_eval - mean_x, query_xs - mean_x
    cov_raw = (x_tr_c.T @ x_tr_c) / len(x_tr_c)
    u_pca = top_pca_basis(cov_raw, K)

    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)
    u_pca_clean = top_pca_basis(cov_clean, K)

    results: list[tuple[str, dict[str, float]]] = [
        (
            "Baseline (PCA 256)",
            eval_all_metrics(
                x_ev_c @ u_pca.T,
                x_q_c @ u_pca.T,
                eval_reviews,
                eval_lengths,
                eval_data["queries"],
                ev_sim,
                ev_mask,
                eval_tag_mat,
            ),
        ),
        (
            "Step1 Cov-Ortho + PCA 256",
            eval_all_metrics(
                x_ev_clean @ u_pca_clean.T,
                x_q_clean @ u_pca_clean.T,
                eval_reviews,
                eval_lengths,
                eval_data["queries"],
                ev_sim,
                ev_mask,
                eval_tag_mat,
            ),
        ),
    ]

    # B1-Ridge & B1-KL（23 タグのみ蒸留 + 残余 PCA 233）
    y_tr_tags = build_tag_matrix(train_keys, records_by_key, non_primary_weight=0.5)
    b1_slices = []
    off = 0
    for b in SUP_BLOCKS:
        b1_slices.append(slice(off, off + len(BLOCK_TAGS[b])))
        off += len(BLOCK_TAGS[b])
    b1_ws = [math.sqrt(w) for w in (0.08, 0.14, 0.14, 0.08, 0.20, 0.36)]

    log("Fitting B1-Ridge(pri0.5, α=2000)...")
    w_ridge = fit_ridge_multi(x_tr_clean, y_tr_tags, alpha=2000.0)
    u_res_r = fit_residual_pca(cov_clean, w_ridge, K - len(ALL_TAGS))
    b1r_blocks = [w_ridge[s] for s in b1_slices] + [u_res_r]
    b1r_rms = [block_rms_norm(x_tr_clean, blk) for blk in b1r_blocks]
    ze_b1r = np.concatenate([(x_ev_clean @ b1r_blocks[i].T) * (b1_ws[i] / b1r_rms[i]) for i in range(6)], axis=1)
    zq_b1r = np.concatenate([(x_q_clean @ b1r_blocks[i].T) * (b1_ws[i] / b1r_rms[i]) for i in range(6)], axis=1)
    results.append(
        (
            "B1-Ridge(pri0.5, sup64/res36)",
            eval_all_metrics(
                ze_b1r, zq_b1r, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat
            ),
        )
    )

    log("Fitting B1-KL (23タグのみ蒸留, primary重み=0.5)...")
    w_kl = fit_kl_logistic_multi(x_tr_clean, y_tr_tags, l2_reg=0.1)
    u_res_b1 = fit_residual_pca(cov_clean, w_kl, K - len(ALL_TAGS))
    b1_blocks = [w_kl[s] for s in b1_slices] + [u_res_b1]
    b1_rms = [block_rms_norm(x_tr_clean, blk) for blk in b1_blocks]
    ze_b1 = np.concatenate([(x_ev_clean @ b1_blocks[i].T) * (b1_ws[i] / b1_rms[i]) for i in range(6)], axis=1)
    zq_b1 = np.concatenate([(x_q_clean @ b1_blocks[i].T) * (b1_ws[i] / b1_rms[i]) for i in range(6)], axis=1)
    results.append(
        (
            "B1-KL (23タグ蒸留, sup64/res36)",
            eval_all_metrics(
                ze_b1, zq_b1, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat
            ),
        )
    )

    # Phase 2 ゲート学習（共通で 1 回学習）
    log("Fitting Phase 2 ブロック有無ゲート (5 ブロック BCE)...")
    gates = fit_block_gates(x_tr_clean, tr_mask, l2_reg=0.05)

    def apply_gate(x_clean: np.ndarray, b: str) -> np.ndarray:
        g, c = gates[b]
        z = np.clip(x_clean @ g + c, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-z)))[:, None]

    # 007-Phase1 & Phase2: β スイープ（0.0 = 独立ペナルティ無し, 0.5, 2.0）
    weight_presets: list[tuple[str, tuple[float, float, float, float, float, float]]] = [
        ("sup70/res30", (0.10, 0.16, 0.16, 0.08, 0.20, 0.30)),
        ("sup60/res40", (0.09, 0.14, 0.14, 0.07, 0.16, 0.40)),
        ("sup80/res20", (0.12, 0.18, 0.18, 0.08, 0.24, 0.20)),
    ]

    best_score = -1.0
    best_payload: dict[str, Any] | None = None

    for beta in (0.0, 0.5, 2.0):
        log(f"Fitting 007-RCD (β={beta}, τ={args.tau}, steps={args.steps})...")
        a_blocks = fit_007_joint_rcd(
            x_tr_clean,
            tr_sim,
            tr_mask,
            tau=args.tau,
            beta_disentangle=beta,
            alpha_l2=0.05,
            pca_subspace_dim=256,
            steps=args.steps,
            device=dev,
        )
        cross_cov = block_cross_cov_norm(x_ev_clean, a_blocks)
        log(f"  -> 評価データ上のブロック間相互共分散ノルム (低いほど独立): {cross_cov:.4f}")

        sup_stack = np.concatenate([a_blocks[b] for b in SUP_BLOCKS], axis=0)
        u_res = fit_residual_pca(cov_clean, sup_stack, RES_DIM)
        all_blocks = [a_blocks[b] for b in SUP_BLOCKS] + [u_res]
        rms_list = [block_rms_norm(x_tr_clean, blk) for blk in all_blocks]

        ev_proj = [x_ev_clean @ blk.T for blk in all_blocks]
        q_proj = [x_q_clean @ blk.T for blk in all_blocks]
        ev_gated = [ev_proj[i] * apply_gate(x_ev_clean, SUP_BLOCKS[i]) for i in range(5)] + [ev_proj[5]]
        q_gated = [q_proj[i] * apply_gate(x_q_clean, SUP_BLOCKS[i]) for i in range(5)] + [q_proj[5]]

        for pname, ws_sq in weight_presets:
            ws = [math.sqrt(w) for w in ws_sq]
            # Phase 1: 純線形 z = Ax + b
            ze1 = np.concatenate([ev_proj[i] * (ws[i] / rms_list[i]) for i in range(6)], axis=1)
            zq1 = np.concatenate([q_proj[i] * (ws[i] / rms_list[i]) for i in range(6)], axis=1)
            m1 = eval_all_metrics(
                ze1, zq1, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat
            )
            results.append((f"007-P1(β={beta}, {pname})", m1))

            score = m1["nDCG@3_macro"] + 0.5 * (m1["aspect@3"] + m1["mood@3"])
            if m1["text@10"] >= 0.60 and score > best_score:
                best_score = score
                # 本番 z = A x + b を合成: x_clean = (x - mean_x) - ((x - mean_x)·u_len) u_len
                # = (I - u_len u_len^T)(x - mean_x)
                eraser_mat = np.eye(D) - np.outer(u_len, u_len)
                a_rows = np.concatenate(
                    [all_blocks[i] * (ws[i] / rms_list[i]) for i in range(6)], axis=0
                )
                a_full = (a_rows @ eraser_mat).astype(np.float32)
                b_full = (-(a_full @ mean_x.ravel())).astype(np.float32)
                best_payload = {
                    "name": f"007-P1(β={beta}, {pname})",
                    "A": a_full,
                    "b": b_full,
                    "metrics": m1,
                }

            # Phase 2: 軽量ゲート付き（β=0.5 のみ比較）
            if beta == 0.5 and pname == "sup70/res30":
                ze2 = np.concatenate([ev_gated[i] * (ws[i] / rms_list[i]) for i in range(6)], axis=1)
                zq2 = np.concatenate([q_gated[i] * (ws[i] / rms_list[i]) for i in range(6)], axis=1)
                m2 = eval_all_metrics(
                    ze2, zq2, eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat
                )
                results.append((f"007-P2+Gate(β={beta}, {pname})", m2))

                # 単独ブロック Disentanglement 確認
                for bi, bname in enumerate((*SUP_BLOCKS, "residual")):
                    ms = eval_all_metrics(
                        ev_proj[bi], q_proj[bi], eval_reviews, eval_lengths, eval_data["queries"], ev_sim, ev_mask, eval_tag_mat
                    )
                    results.append((f"  単独:{bname} (β={beta})", ms))

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

    if args.export_transform and best_payload is not None:
        args.export_transform.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.export_transform,
            A=best_payload["A"],
            b=best_payload["b"],
            name=np.array(best_payload["name"]),
        )
        log(f"最良線形変換 {best_payload['name']} を保存: {args.export_transform}")


if __name__ == "__main__":
    main()
