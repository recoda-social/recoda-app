#!/usr/bin/env python3
"""
plans/006-embedding-disentangle-plan.md フェーズ 2（B1 + B2）完全ホールドアウト学習・評価スクリプト。

- 学習：非評価 2,978 件のみ（リークゼロ）
- 評価：完全未見の embedding-eval.json 700 件 ＋ 短い言葉検索 53 クエリ
- 入力：
  - apps/api/scripts/disentangle-lab/out/all_qwen36.jsonl（Qwen3.6 構造化抽出 3,678 件）
  - Cloudflare D1（x-store-staging-01）の 2048 次元 x（初回取得後 out/x_cache.npz にキャッシュ）
  - Cloudflare Workers AI (@cf/pfnet/plamo-embedding-1b) の 4 スロット details 埋め込み y_i
    （B2 実行時のみ取得し out/slot_y_cache.npz にキャッシュ）

比較手法（すべて本番 z = Ax + b、2048 -> 256 次元にそのまま落とせる線形変換）:
  1. Baseline（現行 PCA 256）
  2. Step 1 単体（Cov-Ortho len λ=1.0 → PCA 256）
  3. B1-Ridge（23 MECE タグ、binary & primary 重み付き）+ 残余 PCA
  4. B1-KL（L2 正則化ソフトラベル・ロジスティック回帰 / KL 蒸留）+ 残余 PCA
  5. B2-rCCA（4 スロット details 埋め込み y_i との正則化 CCA）+ 残余 PCA
  6. B2-RCD（Relational Cosine Distillation: KL(softmax(cos(y)/τ) || softmax(cos(Ax)/τ))）+ 残余 PCA
  7. B1 + B2 ハイブリッド + 単独ブロック Disentanglement 確認

必要な環境変数（キャッシュ未作成時のみ）:
  CLOUDFLARE_ACCOUNT_ID, WORKERS_AI_API_TOKEN, X_STORE_NAME（既定: x-store-staging-01）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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

SLOT_TAGS: dict[str, list[str]] = {
    "music": ["lyrics", "vocal", "sound"],
    "meta": ["tieup_visual", "artist_live"],
    "context": [
        "scene_time",
        "scene_move",
        "scene_daily",
        "mem_youth",
        "mem_love",
        "mem_hardship",
        "mem_people_turn",
        "mem_era",
    ],
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
SLOTS = ("music", "meta", "context", "mood")
ALL_TAGS = [t for s in SLOTS for t in SLOT_TAGS[s]]
TAG_TO_IDX = {t: i for i, t in enumerate(ALL_TAGS)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def official_post_id(key: str) -> str:
    """packages/db/src/official-reviews.ts の officialPostId と完全一致。"""
    digest = hashlib.sha256(f"official-reviews:{key}".encode("utf-8")).digest()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(alphabet[b % len(alphabet)] for b in digest[:12])


def cf_api(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("WORKERS_AI_API_TOKEN")
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
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            if not parsed.get("success"):
                raise RuntimeError(f"Cloudflare API error on {path}: {parsed.get('errors')}")
            return parsed["result"]
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
            log(f"  Workers AI embed ({label}): {min(i + 50, len(texts))}/{len(texts)} ({time.time() - t0:.1f}s)")
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


def slot_detail_text(rec: dict[str, Any], slot: str) -> str:
    """スロット内の details を連結する（primary を先頭に配置）。該当なしは空文字。"""
    entries = rec["entries"].get(slot, [])
    if not entries:
        return ""
    ordered = sorted(entries, key=lambda e: (not e.get("primary", False)))
    parts: list[str] = []
    for e in ordered:
        for d in e.get("details", []):
            ds = d.strip()
            if ds:
                parts.append(ds)
    return "。".join(parts)


def load_or_fetch_slot_y_cache(
    cache_path: Path,
    train_keys: list[str],
    records_by_key: dict[str, dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    学習用 2,978 件の 4 スロット details テキストを Cloudflare Workers AI で埋め込み、
    slot -> (Y_slot [n_train, D], mask [n_train] bool) を返す。空スロットはゼロベクトル。
    """
    if cache_path.exists():
        log(f"slot y キャッシュ読み込み: {cache_path}")
        npz = np.load(cache_path, allow_pickle=False)
        saved_keys = npz["keys"].tolist()
        if saved_keys == train_keys:
            return {
                slot: (npz[f"y_{slot}"].astype(np.float64), npz[f"mask_{slot}"].astype(bool))
                for slot in SLOTS
            }
        log("slot y キャッシュのキー順が異なるため再取得します")

    out_arrays: dict[str, Any] = {"keys": np.array(train_keys)}
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for slot in SLOTS:
        texts = [slot_detail_text(records_by_key[k], slot) for k in train_keys]
        non_empty_idx = [i for i, t in enumerate(texts) if t]
        non_empty_texts = [texts[i] for i in non_empty_idx]
        log(f"スロット {slot}: {len(non_empty_idx)}/{len(train_keys)} 件を Workers AI で埋め込み中...")
        vecs = embed_texts_cf(non_empty_texts, f"slot:{slot}")
        y_mat = np.zeros((len(train_keys), D), dtype=np.float32)
        mask = np.zeros(len(train_keys), dtype=bool)
        for pos, orig_i in enumerate(non_empty_idx):
            y_mat[orig_i] = vecs[pos]
            mask[orig_i] = True
        out_arrays[f"y_{slot}"] = y_mat
        out_arrays[f"mask_{slot}"] = mask
        result[slot] = (y_mat.astype(np.float64), mask)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out_arrays)
    log(f"slot y キャッシュ保存完了: {cache_path}")
    return result


# ---------------------------------------------------------------------------
# 学習アルゴリズム（Step 1 / B1 / B2-rCCA / B2-RCD）
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


def fit_rcca_block(
    x_train: np.ndarray,
    y_slot: np.ndarray,
    mask: np.ndarray,
    out_dim: int,
    reg_x: float = 100.0,
    reg_y: float = 100.0,
) -> np.ndarray:
    """
    正則化 CCA（閉形式）：X と Y_slot の共有角度方向の上位 out_dim 本を抽出する。
    空スロットはゼロベクトルとして含める（有無＋中身の近さを同時学習）。
    返り値: (out_dim, D)
    """
    n = len(x_train)
    xc = x_train - x_train.mean(axis=0, keepdims=True)
    # y_slot は非空を単位正規化し、空はゼロのまま中心化
    yn = y_slot.copy()
    norms = np.linalg.norm(yn, axis=1, keepdims=True)
    yn[mask] = yn[mask] / np.maximum(norms[mask], 1e-12)
    yc = yn - yn.mean(axis=0, keepdims=True)

    cxx = (xc.T @ xc) / n + reg_x * np.eye(D)
    cyy = (yc.T @ yc) / n + reg_y * np.eye(D)
    cxy = (xc.T @ yc) / n

    ex, vx = np.linalg.eigh(cxx)
    ey, vy = np.linalg.eigh(cyy)
    ix = (vx * (1.0 / np.sqrt(np.maximum(ex, 1e-12)))) @ vx.T
    iy = (vy * (1.0 / np.sqrt(np.maximum(ey, 1e-12)))) @ vy.T

    m_mat = ix @ cxy @ iy
    u_svd, s_svd, _ = np.linalg.svd(m_mat, full_matrices=False)
    # 特異値（正準相関）で重み付けして返す
    w_x = (ix @ u_svd[:, :out_dim]) * s_svd[:out_dim]  # (D, out_dim)
    return w_x.T.copy()


def fit_rcd_block(
    x_train: np.ndarray,
    y_slot: np.ndarray,
    mask: np.ndarray,
    out_dim: int,
    tau: float = 0.07,
    l2_reg: float = 1e-3,
    steps: int = 400,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    Relational Cosine Distillation（本命）:
      KL( softmax(cos(y_i, y_j)/tau) || softmax(cos(A x_i, A x_j)/tau) )
    空スロットは 0 ベクトルとし、非空同士・空同士・非空vs空の関係構造を線形行列 A (out_dim x D) に蒸留する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # rCCA で良質な初期値を作ってから RCD で磨く
    w_init = fit_rcca_block(x_train, y_slot, mask, out_dim, reg_x=100.0, reg_y=100.0)
    w_init = w_init / (np.linalg.norm(w_init, axis=1, keepdims=True) + 1e-12)

    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    yn = y_slot.copy()
    norms = np.linalg.norm(yn, axis=1, keepdims=True)
    yn[mask] = yn[mask] / np.maximum(norms[mask], 1e-12)
    y_t = torch.tensor(yn, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)

    a_param = torch.nn.Parameter(torch.tensor(w_init, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([a_param], lr=lr, weight_decay=l2_reg)

    n = len(x_train)
    non_empty_idx = torch.where(mask_t)[0]

    for _ in range(steps):
        # 非空スロットを中心にバッチ抽出（空スロットも 20% 混ぜて有無境界を学習）
        n_pos = min(int(batch_size * 0.8), len(non_empty_idx))
        idx_pos = non_empty_idx[torch.randint(0, len(non_empty_idx), (n_pos,), device=device)]
        idx_all = torch.randint(0, n, (batch_size - n_pos,), device=device)
        idx = torch.cat([idx_pos, idx_all], dim=0)

        xb = x_t[idx]
        yb = y_t[idx]
        mb = mask_t[idx]

        # 教師類似度行列（非空同士は cos(y_i, y_j)、空を含むペアは 0）
        sim_y = yb @ yb.T
        valid_pair = mb[:, None] & mb[None, :]
        sim_y = torch.where(valid_pair, sim_y, torch.zeros_like(sim_y))

        zb = xb @ a_param.T
        zb_norm = F.normalize(zb, dim=-1, eps=1e-6)
        sim_z = zb_norm @ zb_norm.T

        # 対角成分をマスクして KL 蒸留
        diag = torch.eye(len(idx), dtype=torch.bool, device=device)
        logits_y = sim_y.masked_fill(diag, -1e9) / tau
        logits_z = sim_z.masked_fill(diag, -1e9) / tau

        # 非空クエリ行のみで KL を計算（空スロット行は教師分布が一様ノイズになるため）
        pos_rows = torch.where(mb)[0]
        if len(pos_rows) < 2:
            continue
        p_t = F.softmax(logits_y[pos_rows], dim=-1)
        log_q = F.log_softmax(logits_z[pos_rows], dim=-1)
        kl_loss = F.kl_div(log_q, p_t, reduction="batchmean")

        # 空スロットは原点付近に縮める補助項（有無＋中身の近さを 1 ブロックで表現）
        empty_rows = torch.where(~mb)[0]
        empty_loss = (
            (zb[empty_rows].pow(2).mean()) if len(empty_rows) > 0 else torch.tensor(0.0, device=device)
        )
        loss = kl_loss + 0.1 * empty_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return a_param.detach().cpu().numpy().astype(np.float64)


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


# ---------------------------------------------------------------------------
# 評価（完全未見 700 件＋53 クエリ）
# ---------------------------------------------------------------------------


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def build_tag_matrix(records: list[dict[str, Any]], primary_weight: float = 1.0) -> np.ndarray:
    mat = np.zeros((len(records), len(ALL_TAGS)), dtype=np.float64)
    for i, r in enumerate(records):
        for slot in SLOTS:
            for e in r["entries"].get(slot, []):
                idx = TAG_TO_IDX.get(e["tag"])
                if idx is not None:
                    mat[i, idx] = 1.0 if e.get("primary", False) else primary_weight
    return mat


def eval_retrieval(
    z_eval: np.ndarray,
    z_queries: np.ndarray,
    eval_reviews: list[dict[str, Any]],
    eval_lengths: np.ndarray,
    queries: list[dict[str, Any]],
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

    slot_slices: dict[str, slice] = {}
    off = 0
    for s in SLOTS:
        slot_slices[s] = slice(off, off + len(SLOT_TAGS[s]))
        off += len(SLOT_TAGS[s])
    slot_scores = {s: 0.0 for s in SLOTS}
    slot_counts = {s: 0 for s in SLOTS}

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

        for s, sl in slot_slices.items():
            qi = eval_tag_mat[i, sl]
            qn = np.linalg.norm(qi)
            if qn < 1e-6:
                continue
            slot_counts[s] += 1
            sims_s = []
            for j in top3:
                pj = eval_tag_mat[j, sl]
                pn = np.linalg.norm(pj)
                sims_s.append(float(np.dot(qi, pj) / (qn * pn)) if pn > 1e-6 else 0.0)
            slot_scores[s] += float(np.mean(sims_s))

    len_corr = float(np.corrcoef(eval_lengths, top3_lens)[0, 1])

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

    out = {
        "aspect@3": a3 / n,
        "mood@3": m3 / n,
        "aspect@10": a10 / n,
        "mood@10": m10 / n,
        "text@10": text_hits / len(queries),
        "lenCorr@3": len_corr,
    }
    for s in SLOTS:
        out[f"slot_{s}@3"] = slot_scores[s] / max(slot_counts[s], 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 + B2 完全ホールドアウト評価")
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
        "--slot-y-cache",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/slot_y_cache.npz"),
    )
    parser.add_argument("--store-name", default=os.environ.get("X_STORE_NAME", "x-store-staging-01"))
    parser.add_argument("--skip-b2", action="store_true", help="B2（details 埋め込み＋rCCA/RCD）をスキップ")
    args = parser.parse_args()

    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
    posts = official["posts"]
    post_by_key = {p["key"]: p for p in posts}
    eval_keys = {r["key"] for r in eval_data["reviews"]}

    records_by_key: dict[str, dict[str, Any]] = {}
    with args.extracted_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records_by_key[rec["key"]] = rec
    log(f"構造化抽出データ読み込み: {len(records_by_key)} 件")

    x_map, query_xs = load_or_fetch_x_cache(
        args.x_cache,
        [p["key"] for p in posts],
        [q["text"] for q in eval_data["queries"]],
        args.store_name,
    )

    train_keys = [
        p["key"]
        for p in posts
        if p["key"] not in eval_keys and p["key"] in x_map and p["key"] in records_by_key
    ]
    eval_reviews = [
        r
        for r in eval_data["reviews"]
        if r["key"] in post_by_key and r["key"] in x_map and r["key"] in records_by_key
    ]
    log(
        f"完全ホールドアウト構成: 学習={len(train_keys)} 件 / 完全未見評価={len(eval_reviews)} 件 / "
        f"言葉クエリ={len(query_xs)} 件"
    )

    x_train = np.stack([x_map[k] for k in train_keys], axis=0)
    len_train = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)
    train_recs = [records_by_key[k] for k in train_keys]

    x_eval = np.stack([x_map[r["key"]] for r in eval_reviews], axis=0)
    eval_lengths = np.array([len(post_by_key[r["key"]]["content"]) for r in eval_reviews], dtype=np.float64)
    eval_recs = [records_by_key[r["key"]] for r in eval_reviews]
    eval_tag_mat = build_tag_matrix(eval_recs, primary_weight=0.5)

    # 中心化 & Step 1 (Cov-Ortho len λ=1.0)
    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c = x_train - mean_x
    x_ev_c = x_eval - mean_x
    x_q_c = query_xs - mean_x

    cov_raw = (x_tr_c.T @ x_tr_c) / len(x_tr_c)
    u_pca = top_pca_basis(cov_raw, K)

    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)
    u_pca_clean = top_pca_basis(cov_clean, K)

    results: list[tuple[str, dict[str, float]]] = [
        ("Baseline (PCA 256)", eval_retrieval(x_ev_c @ u_pca.T, x_q_c @ u_pca.T, eval_reviews, eval_lengths, eval_data["queries"], eval_tag_mat)),
        ("Step1 Cov-Ortho + PCA 256", eval_retrieval(x_ev_clean @ u_pca_clean.T, x_q_clean @ u_pca_clean.T, eval_reviews, eval_lengths, eval_data["queries"], eval_tag_mat)),
    ]

    # --- B1: 23 MECE タグ（binary & primary 重み付き） ---
    s_slices = [slice(0, 3), slice(3, 5), slice(5, 13), slice(13, 23)]
    res_dim_b1 = K - len(ALL_TAGS)
    b1_presets = [
        ("15/10/20/25/30", (0.15, 0.10, 0.20, 0.25, 0.30)),
        ("12/8/18/22/40", (0.12, 0.08, 0.18, 0.22, 0.40)),
        ("10/10/25/35/20", (0.10, 0.10, 0.25, 0.35, 0.20)),
    ]

    def eval_5blocks(name: str, blocks: list[np.ndarray], presets: list[tuple[str, tuple[float, ...]]]) -> None:
        rms = [block_rms_norm(x_tr_clean, b) for b in blocks]
        ev_b = [x_ev_clean @ b.T for b in blocks]
        q_b = [x_q_clean @ b.T for b in blocks]
        for pname, weights in presets:
            ws = [np.sqrt(w) for w in weights]
            ze = np.concatenate([ev_b[i] * (ws[i] / rms[i]) for i in range(len(blocks))], axis=1)
            zq = np.concatenate([q_b[i] * (ws[i] / rms[i]) for i in range(len(blocks))], axis=1)
            results.append((f"{name} [{pname}]", eval_retrieval(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], eval_tag_mat)))

    for pw_label, pw in [("bin", 1.0), ("pri0.5", 0.5)]:
        y_tags = build_tag_matrix(train_recs, primary_weight=pw)
        log(f"Fitting B1-Ridge ({pw_label}, alpha=2000)...")
        w_r = fit_ridge_multi(x_tr_clean, y_tags, 2000.0)
        u_res = fit_residual_pca(cov_clean, w_r, res_dim_b1)
        eval_5blocks(f"B1-Ridge({pw_label})", [w_r[s] for s in s_slices] + [u_res], b1_presets)

        log(f"Fitting B1-KL ({pw_label}, l2=0.1)...")
        w_kl = fit_kl_logistic_multi(x_tr_clean, y_tags, 0.1)
        u_res_kl = fit_residual_pca(cov_clean, w_kl, res_dim_b1)
        eval_5blocks(f"B1-KL({pw_label})", [w_kl[s] for s in s_slices] + [u_res_kl], b1_presets)

    # --- B2: 4 スロット details 埋め込み y_i（rCCA & Relational Cosine Distillation） ---
    if not args.skip_b2:
        slot_ys = load_or_fetch_slot_y_cache(args.slot_y_cache, train_keys, records_by_key)
        slot_dims = {"music": 16, "meta": 12, "context": 24, "mood": 20}  # 合計 72 + 残余 184 = 256
        res_dim_b2 = K - sum(slot_dims.values())

        log("Fitting B2-rCCA (4 slots: 16/12/24/20 + res 184)...")
        rcca_blocks = [
            fit_rcca_block(x_tr_clean, slot_ys[s][0], slot_ys[s][1], slot_dims[s]) for s in SLOTS
        ]
        u_res_rcca = fit_residual_pca(cov_clean, np.concatenate(rcca_blocks, axis=0), res_dim_b2)
        eval_5blocks("B2-rCCA", rcca_blocks + [u_res_rcca], b1_presets)

        log("Fitting B2-RCD (Relational Cosine Distillation, 4 slots: 16/12/24/20 + res 184)...")
        rcd_blocks = [
            fit_rcd_block(x_tr_clean, slot_ys[s][0], slot_ys[s][1], slot_dims[s]) for s in SLOTS
        ]
        u_res_rcd = fit_residual_pca(cov_clean, np.concatenate(rcd_blocks, axis=0), res_dim_b2)
        solo_presets = [
            ("music単独", (1.0, 0.0, 0.0, 0.0, 0.0)),
            ("meta単独", (0.0, 1.0, 0.0, 0.0, 0.0)),
            ("context単独", (0.0, 0.0, 1.0, 0.0, 0.0)),
            ("mood単独", (0.0, 0.0, 0.0, 1.0, 0.0)),
            ("residual単独", (0.0, 0.0, 0.0, 0.0, 1.0)),
        ]
        eval_5blocks("B2-RCD", rcd_blocks + [u_res_rcd], b1_presets + solo_presets)

    print("\n" + "=" * 124)
    print(
        f"{'条件':<44} {'asp@3':>6} {'mood@3':>6} {'asp@10':>6} {'mood@10':>7} {'言葉@10':>7} {'len@3':>6} | "
        f"{'mus@3':>5} {'met@3':>5} {'ctx@3':>5} {'mod@3':>5}"
    )
    print("-" * 124)
    for name, m in results:
        print(
            f"{name:<44} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
            f"{m['aspect@10']*100:5.1f}% {m['mood@10']*100:6.1f}% "
            f"{m['text@10']*100:6.1f}% {m['lenCorr@3']:6.2f} | "
            f"{m['slot_music@3']*100:4.1f}% {m['slot_meta@3']*100:4.1f}% "
            f"{m['slot_context@3']*100:4.1f}% {m['slot_mood@3']*100:4.1f}%"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
