#!/usr/bin/env python3
"""
フェーズ 2（B1：4 スロット × 23 MECE タグの校正済み確率）の完全ホールドアウト学習・評価スクリプト。

- 学習：非評価 2,978 件のみ（リークゼロ）
- 評価：完全未見の embedding-eval.json 700 件 ＋ 短い言葉検索 53 クエリ
- 比較手法：
  1. Baseline（現行 PCA 256）
  2. Step 1 単体（Cov-Ortho len λ=1.0 → PCA 256）
  3. B1-Ridge（Step 1 + 23 タグ確率への多出力 Ridge + 直交補空間 PCA 233）
  4. B1-KL（Step 1 + 23 タグ確率への L2 正則化 KL 蒸留 / ソフトラベル・ロジスティック回帰 + 直交補空間 PCA 233）
  5. 単独ブロック検索（music / meta / context / mood / residual の Disentanglement 確認）

D1 / Workers AI の 2048 次元ベクトル x は初回のみ取得して out/x_cache.npz にキャッシュする。
必要な環境変数（キャッシュ未作成時のみ）：
  CLOUDFLARE_ACCOUNT_ID, WORKERS_AI_API_TOKEN, X_STORE_NAME（既定: x-store-staging-01）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
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
ALL_TAGS = [t for tags in SLOT_TAGS.values() for t in tags]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def official_post_id(key: str) -> str:
    """packages/db/src/official-reviews.ts の officialPostId と完全に一致する決定論的 ID。"""
    digest = hashlib.sha256(f"official-reviews:{key}".encode("utf-8")).digest()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(alphabet[b % len(alphabet)] for b in digest[:12])


def cf_api(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("WORKERS_AI_API_TOKEN")
    if not account or not token:
        raise RuntimeError(
            "x_cache.npz が未作成です。CLOUDFLARE_ACCOUNT_ID と WORKERS_AI_API_TOKEN を設定してください。"
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    if not parsed.get("success"):
        raise RuntimeError(f"Cloudflare API error on {path}: {parsed.get('errors')}")
    return parsed["result"]


def load_or_fetch_x_cache(
    cache_path: Path,
    post_ids: list[str],
    post_keys: list[str],
    queries: list[str],
    store_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if cache_path.exists():
        log(f"キャッシュ読み込み: {cache_path}")
        npz = np.load(cache_path, allow_pickle=False)
        keys = npz["keys"].tolist()
        xs = npz["xs"].astype(np.float64)
        qxs = npz["query_xs"].astype(np.float64)
        return {k: xs[i] for i, k in enumerate(keys)}, qxs

    log(f"D1 ({store_name}) から {len(post_ids)} 件の 2048 次元 x と {len(queries)} クエリを取得中...")
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
        if (i // 100) % 10 == 0:
            log(f"  D1 fetch: {min(i + 100, len(post_ids))}/{len(post_ids)} (found={len(found)})")

    log(f"Workers AI ({EMBEDDING_SOURCE}) で {len(queries)} 件の言葉クエリを埋め込み中...")
    q_vecs: list[np.ndarray] = []
    for i in range(0, len(queries), 50):
        res = cf_api(
            f"/ai/run/{EMBEDDING_SOURCE}",
            method="POST",
            body={"text": queries[i : i + 50]},
        )
        for v in res["data"]:
            q_vecs.append(np.asarray(v, dtype=np.float32))
    qxs = np.stack(q_vecs, axis=0)

    keys_arr = np.array(list(found.keys()))
    xs_arr = np.stack([found[k] for k in keys_arr], axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, keys=keys_arr, xs=xs_arr, query_xs=qxs)
    log(f"キャッシュ保存完了: {cache_path} (posts={len(found)}, queries={len(qxs)})")
    return {k: found[k].astype(np.float64) for k in found}, qxs.astype(np.float64)


# ---------------------------------------------------------------------------
# 線形代数・学習ロジック（すべて本番 z = Ax + b にそのまま落とせる線形演算）
# ---------------------------------------------------------------------------


def fit_cov_ortho_eraser(x_centered: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """v = Cov(x, len) の単位方向ベクトルを返す。"""
    lc = lengths - lengths.mean()
    v = (x_centered.T @ lc) / len(lengths)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def apply_rank1_eraser(x_centered: np.ndarray, u: np.ndarray, lam: float = 1.0) -> np.ndarray:
    if lam == 0.0:
        return x_centered
    return x_centered - lam * np.outer(x_centered @ u, u)


def top_pca_basis(cov: np.ndarray, k: int) -> np.ndarray:
    """対称共分散行列 cov (d x d) の上位 k 固有ベクトル (k x d) を返す。"""
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx].T.copy()


def fit_ridge_multi(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    """多出力リッジ回帰 W (m x d): min ||X_c W^T - Y_c||_F^2 + alpha ||W||_F^2"""
    xc = x_train - x_train.mean(axis=0, keepdims=True)
    yc = y_train - y_train.mean(axis=0, keepdims=True)
    d = xc.shape[1]
    a = xc.T @ xc + alpha * np.eye(d)
    b = xc.T @ yc
    w = np.linalg.solve(a, b)  # (d, m)
    return w.T.copy()  # (m, d)


def fit_kl_logistic_multi(x_train: np.ndarray, p_train: np.ndarray, l2_reg: float) -> np.ndarray:
    """
    校正済み確率 p_train (n x m) に対する L2 正則化 KL 蒸留（ソフトラベル・ロジスティック回帰）。
    min_{w, b} (1/n) sum_i BCE(p_i, sigmoid(x_i^T w + b)) + (l2_reg / 2) ||w||^2
    返り値 W (m x d)。
    """
    n, d = x_train.shape
    m = p_train.shape[1]
    w_all = np.zeros((m, d), dtype=np.float64)

    for c in range(m):
        p = np.clip(p_train[:, c], 1e-6, 1.0 - 1e-6)
        b0 = float(np.log(p.mean() / (1.0 - p.mean())))
        init = np.zeros(d + 1, dtype=np.float64)
        init[d] = b0

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:d]
            b = params[d]
            z = x_train @ w + b
            # 数値安定な softplus(z) - p * z
            loss_vec = np.logaddexp(0.0, z) - p * z
            loss = float(loss_vec.mean()) + 0.5 * l2_reg * float(np.dot(w, w))
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            diff = (sig - p) / n
            gw = x_train.T @ diff + l2_reg * w
            gb = float(diff.sum())
            grad = np.empty_like(params)
            grad[:d] = gw
            grad[d] = gb
            return loss, grad

        res = minimize(loss_and_grad, init, method="L-BFGS-B", jac=True, options={"maxiter": 200})
        w_all[c] = res.x[:d]
    return w_all


def fit_residual_pca(cov_clean: np.ndarray, sup_rows: np.ndarray, res_dim: int) -> np.ndarray:
    """S_sup の直交補空間上の共分散行列 C_perp = (I - Q^T Q) C (I - Q^T Q) から上位 res_dim 方向を抽出。"""
    q, _ = np.linalg.qr(sup_rows.T, mode="reduced")  # (d, r)
    proj = np.eye(cov_clean.shape[0]) - q @ q.T
    cov_perp = proj @ cov_clean @ proj
    cov_perp = 0.5 * (cov_perp + cov_perp.T)
    u_res = top_pca_basis(cov_perp, res_dim)
    # 数値誤差除去
    u_res = u_res - (u_res @ q) @ q.T
    norms = np.linalg.norm(u_res, axis=1, keepdims=True)
    return u_res / np.maximum(norms, 1e-12)


def block_rms_norm(x_centered: np.ndarray, rows: np.ndarray) -> float:
    proj = x_centered @ rows.T
    return float(np.sqrt(np.mean(np.sum(proj * proj, axis=1)))) or 1.0


# ---------------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------------


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def eval_retrieval(
    z_eval: np.ndarray,
    z_queries: np.ndarray,
    eval_reviews: list[dict[str, Any]],
    eval_lengths: np.ndarray,
    queries: list[dict[str, Any]],
    eval_tag_probs: np.ndarray | None = None,
) -> dict[str, float]:
    ze = normalize_rows(z_eval)
    zq = normalize_rows(z_queries)
    n = len(eval_reviews)

    sim = ze @ ze.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)

    a3 = m3 = a10 = m10 = 0.0
    top3_lens = np.zeros(n)

    # 新 4 スロットのソフトコサイン類似度@3（タグ確率ベクトル同士の近さ）
    slot_slices = {}
    offset = 0
    for sname, tags in SLOT_TAGS.items():
        slot_slices[sname] = slice(offset, offset + len(tags))
        offset += len(tags)
    slot_scores = {sname: 0.0 for sname in SLOT_TAGS}

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

        if eval_tag_probs is not None:
            for sname, sl in slot_slices.items():
                qi = eval_tag_probs[i, sl]
                qn = np.linalg.norm(qi)
                if qn < 1e-6:
                    continue
                sims_s = []
                for j in top3:
                    pj = eval_tag_probs[j, sl]
                    pn = np.linalg.norm(pj)
                    sims_s.append(float(np.dot(qi, pj) / (qn * pn)) if pn > 1e-6 else 0.0)
                slot_scores[sname] += float(np.mean(sims_s))

    # 文字数相関@3
    len_corr = float(np.corrcoef(eval_lengths, top3_lens)[0, 1])

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

    out = {
        "aspect@3": a3 / n,
        "mood@3": m3 / n,
        "aspect@10": a10 / n,
        "mood@10": m10 / n,
        "text@10": text_hits / len(queries),
        "lenCorr@3": len_corr,
    }
    if eval_tag_probs is not None:
        for sname, sl in slot_slices.items():
            active_n = sum(1 for i in range(n) if np.linalg.norm(eval_tag_probs[i, sl]) >= 1e-6)
            out[f"slot_{sname}@3"] = slot_scores[sname] / max(active_n, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 (23 MECE タグ確率) 完全ホールドアウト評価")
    parser.add_argument(
        "--tags-jsonl",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/tags_all_flash.jsonl"),
    )
    parser.add_argument(
        "--x-cache",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/x_cache.npz"),
    )
    parser.add_argument("--store-name", default=os.environ.get("X_STORE_NAME", "x-store-staging-01"))
    args = parser.parse_args()

    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))

    # officialPostId の形式を packages/db/src/official-reviews.ts に合わせる
    # まず Node なしで同じロジックを使うため、official-reviews.ts を確認
    posts = official["posts"]
    post_by_key = {p["key"]: p for p in posts}
    eval_keys = {r["key"] for r in eval_data["reviews"]}

    # tags JSONL 読み込み
    tag_map: dict[str, np.ndarray] = {}
    with args.tags_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tag_map[rec["key"]] = np.array([rec["tags"][t] for t in ALL_TAGS], dtype=np.float64)
    log(f"タグ確率読み込み: {len(tag_map)} 件 ({len(ALL_TAGS)} タグ)")

    # officialPostId の生成（packages/db/src/official-reviews.ts 準拠）
    # official-reviews.ts の形式が不明な場合は --x-cache を TS スクリプトから作るかここで自動一致させる
    post_keys = [p["key"] for p in posts]
    post_ids = [official_post_id(k) for k in post_keys]
    x_map, query_xs = load_or_fetch_x_cache(
        args.x_cache,
        post_ids,
        post_keys,
        [q["text"] for q in eval_data["queries"]],
        args.store_name,
    )

    train_items = [
        (p["key"], len(p["content"]), x_map[p["key"]], tag_map[p["key"]])
        for p in posts
        if p["key"] not in eval_keys and p["key"] in x_map and p["key"] in tag_map
    ]
    eval_items = [
        (r, len(post_by_key[r["key"]]["content"]), x_map[r["key"]], tag_map.get(r["key"]))
        for r in eval_data["reviews"]
        if r["key"] in post_by_key and r["key"] in x_map
    ]
    log(f"完全ホールドアウト構成: 学習={len(train_items)} 件 / 完全未見評価={len(eval_items)} 件 / 言葉クエリ={len(query_xs)} 件")

    x_train = np.stack([t[2] for t in train_items], axis=0)
    y_train = np.stack([t[3] for t in train_items], axis=0)
    len_train = np.array([t[1] for t in train_items], dtype=np.float64)

    eval_reviews = [e[0] for e in eval_items]
    eval_lengths = np.array([e[1] for e in eval_items], dtype=np.float64)
    x_eval = np.stack([e[2] for e in eval_items], axis=0)
    y_eval = (
        np.stack([e[3] for e in eval_items], axis=0)
        if all(e[3] is not None for e in eval_items)
        else None
    )

    # 中心化（学習データ平均のみ使用）
    mean_x = x_train.mean(axis=0, keepdims=True)
    x_tr_c = x_train - mean_x
    x_ev_c = x_eval - mean_x
    x_q_c = query_xs - mean_x

    # 1. Baseline PCA 256
    cov_raw = (x_tr_c.T @ x_tr_c) / len(x_tr_c)
    u_pca = top_pca_basis(cov_raw, K)

    # 2. Step 1 (Cov-Ortho len λ=1.0)
    u_len = fit_cov_ortho_eraser(x_tr_c, len_train)
    x_tr_clean = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev_clean = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_q_clean = apply_rank1_eraser(x_q_c, u_len, 1.0)
    cov_clean = (x_tr_clean.T @ x_tr_clean) / len(x_tr_clean)
    u_pca_clean = top_pca_basis(cov_clean, K)

    results: list[tuple[str, dict[str, float]]] = []
    results.append(
        ("Baseline (PCA 256)", eval_retrieval(x_ev_c @ u_pca.T, x_q_c @ u_pca.T, eval_reviews, eval_lengths, eval_data["queries"], y_eval))
    )
    results.append(
        ("Step1 Cov-Ortho + PCA 256", eval_retrieval(x_ev_clean @ u_pca_clean.T, x_q_clean @ u_pca_clean.T, eval_reviews, eval_lengths, eval_data["queries"], y_eval))
    )

    # ブロック構成: music(3), meta(2), context(8), mood(10), residual(233)
    s_music = slice(0, 3)
    s_meta = slice(3, 5)
    s_ctx = slice(5, 13)
    s_mood = slice(13, 23)
    res_dim = K - len(ALL_TAGS)

    def eval_blocks(label: str, w_sup: np.ndarray, presets: list[tuple[str, float, float, float, float, float]]) -> None:
        u_res = fit_residual_pca(cov_clean, w_sup, res_dim)
        blocks = [w_sup[s_music], w_sup[s_meta], w_sup[s_ctx], w_sup[s_mood], u_res]
        rms = [block_rms_norm(x_tr_clean, b) for b in blocks]
        ev_b = [x_ev_clean @ b.T for b in blocks]
        q_b = [x_q_clean @ b.T for b in blocks]
        for pname, wm, wme, wc, wmo, wr in presets:
            ws = [np.sqrt(wm), np.sqrt(wme), np.sqrt(wc), np.sqrt(wmo), np.sqrt(wr)]
            ze = np.concatenate([ev_b[i] * (ws[i] / rms[i]) for i in range(5)], axis=1)
            zq = np.concatenate([q_b[i] * (ws[i] / rms[i]) for i in range(5)], axis=1)
            results.append((f"{label} [{pname}]", eval_retrieval(ze, zq, eval_reviews, eval_lengths, eval_data["queries"], y_eval)))

    weight_presets = [
        ("均等sup70/res30 (15/10/20/25/30)", 0.15, 0.10, 0.20, 0.25, 0.30),
        ("均等sup60/res40 (12/8/18/22/40)", 0.12, 0.08, 0.18, 0.22, 0.40),
        ("ctx+mood重視 (10/10/25/35/20)", 0.10, 0.10, 0.25, 0.35, 0.20),
        ("均等4slot+res50 (10/10/15/15/50)", 0.10, 0.10, 0.15, 0.15, 0.50),
    ]
    solo_presets = [
        ("music単独", 1.0, 0.0, 0.0, 0.0, 0.0),
        ("meta単独", 0.0, 1.0, 0.0, 0.0, 0.0),
        ("context単独", 0.0, 0.0, 1.0, 0.0, 0.0),
        ("mood単独", 0.0, 0.0, 0.0, 1.0, 0.0),
        ("residual単独", 0.0, 0.0, 0.0, 0.0, 1.0),
    ]

    for alpha in [500.0, 2000.0, 5000.0]:
        log(f"Fitting B1-Ridge (alpha={alpha})...")
        w_ridge = fit_ridge_multi(x_tr_clean, y_train, alpha)
        eval_blocks(f"B1-Ridge(α={int(alpha)})", w_ridge, weight_presets)

    for l2 in [0.01, 0.05, 0.2, 1.0]:
        log(f"Fitting B1-KL Logistic (l2={l2})...")
        w_kl = fit_kl_logistic_multi(x_tr_clean, y_train, l2)
        eval_blocks(f"B1-KL(λ={l2})", w_kl, weight_presets + (solo_presets if l2 == 0.2 else []))

    print("\n" + "=" * 124)
    print(
        f"{'条件':<48} {'asp@3':>6} {'mood@3':>6} {'asp@10':>6} {'mood@10':>7} {'言葉@10':>7} {'len@3':>6} | "
        f"{'mus@3':>5} {'met@3':>5} {'ctx@3':>5} {'mod@3':>5}"
    )
    print("-" * 124)
    for name, m in results:
        print(
            f"{name:<48} {m['aspect@3']*100:5.1f}% {m['mood@3']*100:5.1f}% "
            f"{m['aspect@10']*100:5.1f}% {m['mood@10']*100:6.1f}% "
            f"{m['text@10']*100:6.1f}% {m['lenCorr@3']:6.2f} | "
            f"{m.get('slot_music@3', 0)*100:4.1f}% {m.get('slot_meta@3', 0)*100:4.1f}% "
            f"{m.get('slot_context@3', 0)*100:4.1f}% {m.get('slot_mood@3', 0)*100:4.1f}%"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
