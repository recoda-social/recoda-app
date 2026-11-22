#!/usr/bin/env python3
"""
Stage-2 マルチベクトル Reranking を、現行 rcca_reg8 と完全同一の 150 クエリ × StrictResonanceJudge で比較する。

実行手順（DGX Spark の統合メモリ保護のため、--prepare と --judge を分離）:
  1. python eval_strict_judge_multivec.py --prepare
  2. bash serve_vllm.sh  （別プロセスで vLLM 起動）
  3. python eval_strict_judge_multivec.py --judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eval_multivec_rerank import (
    SLOT4_TAGS,
    SLOT5_TAGS,
    SLOTS_4,
    SLOTS_5,
    build_slot_detail_vectors,
    compute_reranked_scores,
    fit_slot_models,
    fit_tag_classifiers,
    predict_tag_probs,
)
from prepare_3pillar_candidates import fit_rcca_custom
from prepare_nonridge_judge import build_clean_detail_vectors
from score_train_pairs_vllm import judge_pair
from train_eval_007 import (
    K,
    apply_rank1_eraser,
    build_tag_matrix,
    fit_cov_ortho_eraser,
    log,
    normalize_rows,
)


def prepare_candidates(out_dir: Path) -> None:
    t0 = time.time()
    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    eval_data = json.loads(Path("packages/db/data/embedding-eval.json").read_text(encoding="utf-8"))
    prev_items = json.loads((out_dir / "judge_candidates_008.json").read_text(encoding="utf-8"))
    eval_key_set = {p["key"] for p in eval_data["reviews"]}
    post_by_key = {p["key"]: p for p in official["posts"]}

    records_by_key: dict[str, dict[str, Any]] = {}
    with (out_dir / "all_qwen36.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records_by_key[rec["key"]] = rec

    x_npz = np.load(out_dir / "x_cache.npz", allow_pickle=False)
    x_map = {k: x_npz["xs"][i].astype(np.float64) for i, k in enumerate(x_npz["keys"].tolist())}

    d_npz = np.load(out_dir / "tag_details_cache.npz", allow_pickle=False)
    d_vecs = d_npz["vecs"].astype(np.float32)
    text_vecs = {t: d_vecs[i] for i, t in enumerate(d_npz["texts"].tolist())}

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

    x_tr_raw = np.stack([x_map[k] for k in train_keys], axis=0)
    x_ev_raw = np.stack([x_map[k] for k in eval_keys], axis=0)
    lens_tr = np.array([len(post_by_key[k]["content"]) for k in train_keys], dtype=np.float64)

    # Step 1: 中心化 + Cov-Ortho(len, λ=1.0)
    mean_x = x_tr_raw.mean(axis=0)
    u_len = fit_cov_ortho_eraser(x_tr_raw - mean_x[None, :], lens_tr)
    x_tr = apply_rank1_eraser(x_tr_raw - mean_x[None, :], u_len, 1.0)
    x_ev = apply_rank1_eraser(x_ev_raw - mean_x[None, :], u_len, 1.0)
    x_ev_u = normalize_rows(x_ev)

    # Stage 1: rcca_reg8
    log("Stage 1 (rcca_reg8) 学習中...")
    y_clean_tr, valid_tr = build_clean_detail_vectors(train_keys, records_by_key, text_vecs)
    w_rcca = fit_rcca_custom(x_tr, y_clean_tr, valid_tr, K, reg_x=8.0, reg_y=4.0, power=1.5)
    z1_ev_u = normalize_rows(x_ev @ w_rcca.T)
    sim_stage1 = z1_ev_u @ z1_ev_u.T

    # Stage 2: 4スロット / 5スロット + primary ゲート + 23タグ
    log("Stage 2 (4slot & 5slot rCCA + primary gate + 23-tag) 学習中...")
    y4_tr, m4_tr, pr4_tr = build_slot_detail_vectors(train_keys, records_by_key, text_vecs, SLOT4_TAGS)
    y5_tr, m5_tr, pr5_tr = build_slot_detail_vectors(train_keys, records_by_key, text_vecs, SLOT5_TAGS)

    w4, _, g4_prim = fit_slot_models(x_tr, y4_tr, m4_tr, pr4_tr, SLOTS_4, slot_dim=32, reg_x=12.0, reg_y=4.0, power=1.5)
    w5, _, g5_prim = fit_slot_models(x_tr, y5_tr, m5_tr, pr5_tr, SLOTS_5, slot_dim=32, reg_x=12.0, reg_y=4.0, power=1.5)

    tag_tr = build_tag_matrix(train_keys, records_by_key)
    tag_clfs = fit_tag_classifiers(x_tr, tag_tr)
    p_tags_ev = predict_tag_probs(x_ev, tag_clfs)

    z4_u = {s: normalize_rows(x_ev @ w4[s].T) for s in SLOTS_4}
    p4_prim = {s: g4_prim[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_4}
    z5_u = {s: normalize_rows(x_ev @ w5[s].T) for s in SLOTS_5}
    p5_prim = {s: g5_prim[s].predict_proba(x_ev_u)[:, 1] for s in SLOTS_5}

    score_mats = {
        "rcca_reg8": sim_stage1,
        "4slot_prim_qlead": compute_reranked_scores(
            sim_stage1, z4_u, p4_prim, SLOTS_4, isrcs_ev, top_k_cand=20, alpha=0.35, gate_mode="qlead"
        ),
        "5slot_prim_qlead": compute_reranked_scores(
            sim_stage1, z5_u, p5_prim, SLOTS_5, isrcs_ev, top_k_cand=20, alpha=0.35, gate_mode="qlead"
        ),
        "5slot_prim_tag": compute_reranked_scores(
            sim_stage1, z5_u, p5_prim, SLOTS_5, isrcs_ev, top_k_cand=20, alpha=0.30, gate_mode="prod", p_tags=p_tags_ev, beta_tag=0.10
        ),
    }

    items: list[dict[str, Any]] = []
    for prev in prev_items:
        qk = prev["query"]["key"]
        qi = eval_key_to_idx[qk]
        top3_map: dict[str, list[dict[str, Any]]] = {}
        for mname, smat in score_mats.items():
            if mname == "rcca_reg8":
                # 念のため prev と完全一致させる（同曲除外済みの Top-3）
                top3_map[mname] = prev["top3"]["rcca_reg8"]
                continue
            row = smat[qi].copy()
            for j in range(len(eval_keys)):
                if j == qi or (isrcs_ev[qi] and isrcs_ev[j] == isrcs_ev[qi]):
                    row[j] = -1e9
            idxs = np.argsort(-row)[:3].tolist()
            top3_map[mname] = [
                {
                    "key": eval_keys[j],
                    "title": records_by_key[eval_keys[j]]["title"],
                    "artistName": records_by_key[eval_keys[j]]["artistName"],
                    "content": records_by_key[eval_keys[j]]["content"],
                }
                for j in idxs
            ]
        items.append({"query": prev["query"], "top3": top3_map})

    out_path = out_dir / "judge_candidates_multivec.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"候補生成完了: {out_path} ({len(items)} queries, {time.time() - t0:.1f}s)")


async def run_judge(out_dir: Path) -> None:
    items = json.loads((out_dir / "judge_candidates_multivec.json").read_text(encoding="utf-8"))
    method_names = list(items[0]["top3"].keys())

    # 既存の StrictResonanceJudge 採点結果をキャッシュとして読み込む（完全同一条件）
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    prev_res_path = out_dir / "judge_results_008.json"
    if prev_res_path.exists():
        prev_res = json.loads(prev_res_path.read_text(encoding="utf-8"))
        for k_str, v in prev_res.get("scores", {}).items():
            qk, dk = k_str.split("||", 1)
            scores[(qk, dk)] = v
        log(f"既存キャッシュ読み込み: {len(scores)} ペア ({prev_res_path.name})")

    to_judge: dict[tuple[str, str], dict[str, Any]] = {}
    total_unique: set[tuple[str, str]] = set()
    for it in items:
        q = it["query"]
        for m in method_names:
            for d in it["top3"][m]:
                pk = (q["key"], d["key"])
                total_unique.add(pk)
                if pk not in scores and pk not in to_judge:
                    to_judge[pk] = {
                        "q_key": q["key"],
                        "d_key": d["key"],
                        "q_text": q["content"],
                        "d_text": d["content"],
                    }

    log(
        f"完全未見 {len(items)} クエリ × {len(method_names)} 手法 | "
        f"ユニーク計 {len(total_unique)} ペア (キャッシュ済={len(total_unique) - len(to_judge)}, 新規採点={len(to_judge)})"
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
                if wait_i == 0:
                    log("vLLM サーバーの起動完了を待機中...")
                await asyncio.sleep(5.0)
            else:
                raise RuntimeError("vLLM サーバーが起動しませんでした")

            t0 = time.time()
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
                if done % 100 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    log(f"  [{done}/{len(tasks)}] ok={done-fail} fail={fail} | {done/max(elapsed,1e-6):.1f} req/s ({elapsed:.1f}s)")

    print("\n" + "=" * 124)
    print(f"StrictResonanceJudge（完全未見 {len(items)} クエリ × Top-3 = 各 {len(items)*3} ペア）")
    print("=" * 124)
    print(
        f"{'手法':<18} {'平均点(/3)':>10} {'妥当率(≥2)':>11} {'強共鳴(3)':>10} {'ハズレ(0)':>10} "
        f"{'Top3満足率':>11} {'#1妥当率':>9} | {'0 / 1 / 2 / 3 分布':>22}"
    )
    print("-" * 124)
    summary: dict[str, Any] = {}
    per_q_mean: dict[str, list[float]] = {m: [] for m in method_names}
    for mname in method_names:
        all_s: list[int] = []
        top1_s: list[int] = []
        q_has_good = 0
        for it in items:
            qk = it["query"]["key"]
            row_s = [scores[(qk, d["key"])]["score"] for d in it["top3"][mname] if (qk, d["key"]) in scores]
            if not row_s:
                per_q_mean[mname].append(0.0)
                continue
            all_s.extend(row_s)
            top1_s.append(row_s[0])
            per_q_mean[mname].append(sum(row_s) / len(row_s))
            if any(s >= 2 for s in row_s):
                q_has_good += 1
        n_tot = len(all_s) or 1
        mean_s = sum(all_s) / n_tot
        valid_rate = sum(1 for s in all_s if s >= 2) / n_tot * 100
        strong_rate = sum(1 for s in all_s if s == 3) / n_tot * 100
        zero_rate = sum(1 for s in all_s if s == 0) / n_tot * 100
        sat_rate = q_has_good / len(items) * 100
        t1_rate = sum(1 for s in top1_s if s >= 2) / max(len(top1_s), 1) * 100
        dist = [sum(1 for s in all_s if s == v) for v in (0, 1, 2, 3)]
        dist_str = f"{dist[0]:3d} / {dist[1]:3d} / {dist[2]:3d} / {dist[3]:3d}"
        print(
            f"{mname:<18} {mean_s:10.3f} {valid_rate:10.1f}% {strong_rate:9.1f}% {zero_rate:9.1f}% "
            f"{sat_rate:10.1f}% {t1_rate:8.1f}% | {dist_str:>22}"
        )
        summary[mname] = {
            "mean_score": round(mean_s, 4),
            "valid_rate_ge2": round(valid_rate, 2),
            "strong_rate_3": round(strong_rate, 2),
            "zero_rate_0": round(zero_rate, 2),
            "top3_satisfaction_rate": round(sat_rate, 2),
            "top1_valid_rate": round(t1_rate, 2),
            "dist_0_1_2_3": dist,
        }
    print("=" * 124)

    # 現行 rcca_reg8 に対するクエリ単位の勝敗表
    print("\nvs rcca_reg8 クエリ単位の勝敗（Top-3 平均点）:")
    base_qm = per_q_mean["rcca_reg8"]
    for mname in method_names:
        if mname == "rcca_reg8":
            continue
        w = sum(1 for a, b in zip(per_q_mean[mname], base_qm) if a > b + 1e-9)
        l = sum(1 for a, b in zip(per_q_mean[mname], base_qm) if a < b - 1e-9)
        t = len(base_qm) - w - l
        print(f"  {mname:<18}: Win={w:3d}, Loss={l:3d}, Tie={t:3d} (Δ={w - l:+d})")

    out_path = out_dir / "judge_results_multivec.json"
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "scores": {f"{k[0]}||{k[1]}": v for k, v in scores.items() if k in total_unique},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"詳細結果保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    if args.prepare:
        prepare_candidates(out_dir)
    if args.judge:
        asyncio.run(run_judge(out_dir))


if __name__ == "__main__":
    main()
