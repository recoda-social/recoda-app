#!/usr/bin/env python3
"""
書き換え後公式レビュー（3,675件: 曲番号300以降すべて書き換え）での完全ホールドアウト検証＋StrictResonanceJudge 採点。

比較手法（すべて Train 2,976 件のみで学習、完全未見 Eval 699 件で評価）:
  1. baseline_pca     : 現行 PCA 256（文字数除去なし）
  2. rcca_reg8        : Cov-Ortho(len, λ=1.0) + rCCA(reg_x=8.0, reg_y=4.0, power=1.5)
  3. rcca_reg4        : Cov-Ortho(len, λ=1.0) + rCCA(reg_x=4.0, reg_y=2.0, power=1.5)
  4. 4slot_prim_qlead : rcca_reg8 Top-20 → 4スロット rCCA + primary qlead ゲート (α=0.35)

※ 書き換え済み投稿は同じ key でも本文が変わっているため、キャッシュ照合は必ず
   (q_key, d_key, q_text, d_text) の完全一致で行う。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eval_multivec_rerank import (
    SLOT4_TAGS,
    SLOTS_4,
    build_slot_detail_vectors,
    compute_reranked_scores,
    fit_slot_models,
)
from prepare_3pillar_candidates import fit_rcca_custom
from prepare_nonridge_judge import build_clean_detail_vectors
from score_train_pairs_vllm import judge_pair
from train_eval_007 import (
    K,
    apply_rank1_eraser,
    fit_cov_ortho_eraser,
    log,
    normalize_rows,
    top_pca_basis,
)


def top3_excluding_same_song(
    sim_row: np.ndarray, qi: int, isrcs: list[str]
) -> list[int]:
    row = sim_row.copy()
    for j in range(len(isrcs)):
        if j == qi or (isrcs[qi] and isrcs[j] == isrcs[qi]):
            row[j] = -1e9
    return np.argsort(-row)[:3].tolist()


def eval_len_corr(sim_mat: np.ndarray, isrcs: list[str], lens: np.ndarray) -> float:
    qs: list[float] = []
    cs: list[float] = []
    for i in range(len(isrcs)):
        for j in top3_excluding_same_song(sim_mat[i], i, isrcs):
            qs.append(float(lens[i]))
            cs.append(float(lens[j]))
    return float(np.corrcoef(qs, cs)[0, 1])


async def main() -> None:
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

    v1_content: dict[str, str] = {}
    v1_path = out_dir / "all_qwen36_v1.jsonl"
    if v1_path.exists():
        with v1_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    v1_content[rec["key"]] = rec["content"]

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
        if p["key"] not in eval_key_set and p["key"] in x_map and p["key"] in records_by_key
    ]
    eval_keys = [
        p["key"]
        for p in eval_data["reviews"]
        if p["key"] in x_map and p["key"] in records_by_key
    ]
    eval_key_to_idx = {k: i for i, k in enumerate(eval_keys)}
    isrcs_ev = [post_by_key[k].get("isrc", "") for k in eval_keys]
    lens_tr = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)
    lens_ev = np.array([len(post_by_key[k]["content"]) for k in eval_keys], dtype=np.float64)
    is_rewritten_ev = [v1_content.get(k) != post_by_key[k]["content"] for k in eval_keys]

    log(
        f"v2 データ: Train={len(train_keys)}, Eval(unseen)={len(eval_keys)} "
        f"(うち書き換え={sum(is_rewritten_ev)}, 未変更={len(eval_keys) - sum(is_rewritten_ev)})"
    )

    x_tr_raw = np.stack([x_map[k] for k in train_keys], axis=0)
    x_ev_raw = np.stack([x_map[k] for k in eval_keys], axis=0)
    mean_x = x_tr_raw.mean(axis=0)
    x_tr_c = x_tr_raw - mean_x[None, :]
    x_ev_c = x_ev_raw - mean_x[None, :]

    # 1. Baseline PCA 256
    log("1. Baseline PCA 256 学習中...")
    cov_tr = (x_tr_c.T @ x_tr_c) / len(x_tr_c)
    w_pca = top_pca_basis(cov_tr, K)
    z_pca_ev_u = normalize_rows(x_ev_c @ w_pca.T)
    sim_pca = z_pca_ev_u @ z_pca_ev_u.T

    # Step 1: Cov-Ortho(len, λ=1.0)
    u_len = fit_cov_ortho_eraser(x_tr_c, lens_tr)
    x_tr = apply_rank1_eraser(x_tr_c, u_len, 1.0)
    x_ev = apply_rank1_eraser(x_ev_c, u_len, 1.0)
    x_ev_u = normalize_rows(x_ev)

    # 2. rcca_reg8 & 3. rcca_reg4
    log("2. rcca_reg8 & rcca_reg4 学習中...")
    y_clean_tr, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)
    w_rcca8 = fit_rcca_custom(x_tr, y_clean_tr, valid_tr, K, reg_x=8.0, reg_y=4.0, power=1.5)
    w_rcca4 = fit_rcca_custom(x_tr, y_clean_tr, valid_tr, K, reg_x=4.0, reg_y=2.0, power=1.5)
    z_r8_ev_u = normalize_rows(x_ev @ w_rcca8.T)
    z_r4_ev_u = normalize_rows(x_ev @ w_rcca4.T)
    sim_r8 = z_r8_ev_u @ z_r8_ev_u.T
    sim_r4 = z_r4_ev_u @ z_r4_ev_u.T

    # 4. 4slot_prim_qlead
    log("3. 4slot_prim_qlead 学習中...")
    y4_tr, m4_tr, pr4_tr = build_slot_detail_vectors(train_keys, records_by_key, text_vecs, SLOT4_TAGS)
    w4, _, g4_prim = fit_slot_models(
        x_tr, y4_tr, m4_tr, pr4_tr, SLOTS_4, slot_dim=32, reg_x=12.0, reg_y=4.0, power=1.5
    )
    z4_u = {s: normalize_rows(x_ev @ w4[s].T) for s in SLOTS_4}
    p4_prim = {s: g4_prim[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_4}
    sim_4slot = compute_reranked_scores(
        sim_r8, z4_u, p4_prim, SLOTS_4, isrcs_ev, top_k_cand=20, alpha=0.35, gate_mode="qlead"
    )

    # 短い言葉検索 (53 queries) のヒット率も確認
    queries_meta = eval_data["queries"]
    eval_meta_by_key = {r["key"]: r for r in eval_data["reviews"]}

    def eval_text_queries(z_ev_u: np.ndarray, w_mat: np.ndarray, use_len_ortho: bool) -> float:
        q_c = query_xs - mean_x[None, :]
        if use_len_ortho:
            q_c = apply_rank1_eraser(q_c, u_len, 1.0)
        qz_u = normalize_rows(q_c @ w_mat.T)
        sim_q = qz_u @ z_ev_u.T  # (53, 699)
        hits = []
        for qi, qinfo in enumerate(queries_meta):
            top10 = np.argsort(-sim_q[qi])[:10]
            want_asp = set(qinfo.get("aspects", []))
            want_mood = set(qinfo.get("moods", []))
            ok = 0
            for idx in top10:
                rm = eval_meta_by_key[eval_keys[idx]]
                a_ok = (not want_asp) or bool(want_asp & set(rm.get("aspects", [])))
                m_ok = (not want_mood) or bool(want_mood & set(rm.get("moods", [])))
                if a_ok and m_ok:
                    ok += 1
            hits.append(ok / 10.0)
        return float(np.mean(hits)) * 100.0

    log("=== オフライン基本指標（v2 書き換え後 Eval 699 件） ===")
    print(f"  baseline_pca     : lenR@3={eval_len_corr(sim_pca, isrcs_ev, lens_ev):.3f} | 言葉@10={eval_text_queries(z_pca_ev_u, w_pca, False):.1f}%")
    print(f"  rcca_reg8        : lenR@3={eval_len_corr(sim_r8, isrcs_ev, lens_ev):.3f} | 言葉@10={eval_text_queries(z_r8_ev_u, w_rcca8, True):.1f}%")
    print(f"  rcca_reg4        : lenR@3={eval_len_corr(sim_r4, isrcs_ev, lens_ev):.3f} | 言葉@10={eval_text_queries(z_r4_ev_u, w_rcca4, True):.1f}%")
    print(f"  4slot_prim_qlead : lenR@3={eval_len_corr(sim_4slot, isrcs_ev, lens_ev):.3f}")

    # 150 クエリ集合の構築（judge_candidates_008.json と同じキー集合をベースに、削除分だけ補完）
    prev_items = json.loads((out_dir / "judge_candidates_008.json").read_text(encoding="utf-8"))
    selected_q_keys = [it["query"]["key"] for it in prev_items if it["query"]["key"] in eval_key_to_idx]
    selected_set = set(selected_q_keys)
    for k in eval_keys:
        if len(selected_q_keys) >= 150:
            break
        if k not in selected_set:
            selected_q_keys.append(k)
            selected_set.add(k)

    method_sims = {
        "baseline_pca": sim_pca,
        "rcca_reg8": sim_r8,
        "rcca_reg4": sim_r4,
        "4slot_prim_qlead": sim_4slot,
    }
    method_names = list(method_sims.keys())

    items: list[dict[str, Any]] = []
    for qk in selected_q_keys:
        qi = eval_key_to_idx[qk]
        q_rec = records_by_key[qk]
        top3_map: dict[str, list[dict[str, Any]]] = {}
        for mname, smat in method_sims.items():
            idxs = top3_excluding_same_song(smat[qi], qi, isrcs_ev)
            top3_map[mname] = [
                {
                    "key": eval_keys[j],
                    "title": records_by_key[eval_keys[j]]["title"],
                    "artistName": records_by_key[eval_keys[j]]["artistName"],
                    "content": records_by_key[eval_keys[j]]["content"],
                }
                for j in idxs
            ]
        items.append(
            {
                "query": {
                    "key": qk,
                    "title": q_rec["title"],
                    "artistName": q_rec["artistName"],
                    "content": q_rec["content"],
                    "rewritten": is_rewritten_ev[qi],
                },
                "top3": top3_map,
            }
        )

    # 既存キャッシュ読み込み（本文まで完全一致するペアのみ再利用！）
    exact_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for prev_file in ("judge_results_008.json", "judge_results_multivec.json"):
        p = out_dir / prev_file
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for v in data.get("scores", {}).values():
            if all(f in v for f in ("q_key", "d_key", "q_text", "d_text", "score")):
                exact_cache[(v["q_key"], v["d_key"], v["q_text"], v["d_text"])] = v
    log(f"本文完全一致キャッシュ読み込み: {len(exact_cache)} ペア")

    scores: dict[tuple[str, str], dict[str, Any]] = {}
    to_judge: dict[tuple[str, str], dict[str, Any]] = {}
    for it in items:
        q = it["query"]
        for m in method_names:
            for d in it["top3"][m]:
                pk = (q["key"], d["key"])
                ck = (q["key"], d["key"], q["content"], d["content"])
                if ck in exact_cache:
                    scores[pk] = exact_cache[ck]
                elif pk not in to_judge:
                    to_judge[pk] = {
                        "q_key": q["key"],
                        "d_key": d["key"],
                        "q_text": q["content"],
                        "d_text": d["content"],
                    }

    total_unique = len(scores) + len(to_judge)
    log(
        f"StrictResonanceJudge (v2): {len(items)} クエリ × {len(method_names)} 手法 | "
        f"ユニーク計 {total_unique} ペア (本文一致キャッシュ={len(scores)}, 新規採点={len(to_judge)})"
    )

    if to_judge:
        sem = asyncio.Semaphore(256)
        limits = httpx.Limits(max_connections=272, max_keepalive_connections=256)
        timeout = httpx.Timeout(180.0, connect=30.0)
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            for wait_i in range(60):
                try:
                    r = await client.get("http://127.0.0.1:8000/v1/models")
                    if r.status_code == 200:
                        log(f"vLLM 疎通確認 OK: {[m['id'] for m in r.json().get('data', [])]}")
                        break
                except Exception:
                    pass
                await asyncio.sleep(5.0)
            else:
                raise RuntimeError("vLLM サーバーが応答しません")

            tj = time.time()
            tasks = [
                asyncio.create_task(
                    judge_pair(client, sem, "http://127.0.0.1:8000", "qwen3.6-35b-a3b", payload)
                )
                for payload in to_judge.values()
            ]
            done = 0
            fail = 0
            for coro in asyncio.as_completed(tasks):
                try:
                    rec = await coro
                    scores[(rec["q_key"], rec["d_key"])] = rec
                except Exception as exc:
                    fail += 1
                    log(f"  [WARN] {exc}")
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    elapsed = time.time() - tj
                    log(
                        f"  [{done}/{len(tasks)}] ok={done-fail} fail={fail} | "
                        f"{done/max(elapsed,1e-6):.1f} req/s ({elapsed:.1f}s)"
                    )

    def print_table(title: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        print("\n" + "=" * 124)
        print(f"{title}（{len(subset)} クエリ × Top-3 = 各 {len(subset)*3} ペア）")
        print("=" * 124)
        print(
            f"{'手法':<18} {'平均点(/3)':>10} {'妥当率(≥2)':>11} {'強共鳴(3)':>10} {'ハズレ(0)':>10} "
            f"{'Top3満足率':>11} {'#1妥当率':>9} | {'0 / 1 / 2 / 3 分布':>22}"
        )
        print("-" * 124)
        out_sum: dict[str, Any] = {}
        for mname in method_names:
            all_s: list[int] = []
            top1_s: list[int] = []
            q_good = 0
            for it in subset:
                qk = it["query"]["key"]
                row_s = [scores[(qk, d["key"])]["score"] for d in it["top3"][mname] if (qk, d["key"]) in scores]
                if not row_s:
                    continue
                all_s.extend(row_s)
                top1_s.append(row_s[0])
                if any(s >= 2 for s in row_s):
                    q_good += 1
            n_tot = len(all_s) or 1
            mean_s = sum(all_s) / n_tot
            valid_r = sum(1 for s in all_s if s >= 2) / n_tot * 100
            strong_r = sum(1 for s in all_s if s == 3) / n_tot * 100
            zero_r = sum(1 for s in all_s if s == 0) / n_tot * 100
            sat_r = q_good / max(len(subset), 1) * 100
            t1_r = sum(1 for s in top1_s if s >= 2) / max(len(top1_s), 1) * 100
            dist = [sum(1 for s in all_s if s == v) for v in (0, 1, 2, 3)]
            dist_str = f"{dist[0]:3d} / {dist[1]:3d} / {dist[2]:3d} / {dist[3]:3d}"
            print(
                f"{mname:<18} {mean_s:10.3f} {valid_r:10.1f}% {strong_r:9.1f}% {zero_r:9.1f}% "
                f"{sat_r:10.1f}% {t1_r:8.1f}% | {dist_str:>22}"
            )
            out_sum[mname] = {
                "mean_score": round(mean_s, 4),
                "valid_rate_ge2": round(valid_r, 2),
                "strong_rate_3": round(strong_r, 2),
                "zero_rate_0": round(zero_r, 2),
                "top3_satisfaction_rate": round(sat_r, 2),
                "top1_valid_rate": round(t1_r, 2),
                "dist_0_1_2_3": dist,
            }
        print("=" * 124)
        return out_sum

    summary_all = print_table("StrictResonanceJudge v2【全体】", items)
    rewritten_items = [it for it in items if it["query"]["rewritten"]]
    unchanged_items = [it for it in items if not it["query"]["rewritten"]]
    summary_rw = print_table("StrictResonanceJudge v2【書き換えクエリのみ】", rewritten_items)
    summary_un = print_table("StrictResonanceJudge v2【未変更クエリのみ】", unchanged_items)

    out_path = out_dir / "judge_results_v2_rewritten.json"
    out_path.write_text(
        json.dumps(
            {
                "summary_all": summary_all,
                "summary_rewritten": summary_rw,
                "summary_unchanged": summary_un,
                "scores": {f"{k[0]}||{k[1]}": v for k, v in scores.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"v2 評価完了・保存: {out_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
