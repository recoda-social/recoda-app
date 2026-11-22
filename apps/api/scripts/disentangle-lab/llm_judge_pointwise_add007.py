#!/usr/bin/env python3
"""
既存の judge_pointwise_draft_assist.json の採点キャッシュを再利用しつつ、
007 蒸留系（light_hyb25 / p1_linear）を追加採点して全手法を同一 DRAFT_ASSIST 厳格基準で横並び比較する。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from llm_judge_pointwise import log, score_one


async def main() -> None:
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    j_3p = json.loads((out_dir / "judge_candidates_3pillar.json").read_text(encoding="utf-8"))
    j_v1 = json.loads((out_dir / "judge_candidates.json").read_text(encoding="utf-8"))
    j_v2 = json.loads((out_dir / "judge_candidates_v2.json").read_text(encoding="utf-8"))
    prev = json.loads((out_dir / "judge_pointwise_draft_assist.json").read_text(encoding="utf-8"))

    scores: dict[tuple[str, str], dict[str, Any]] = {}
    for k_str, val in prev["scores"].items():
        qk, dk = k_str.split("||", 1)
        scores[(qk, dk)] = val
    log(f"既存 DRAFT_ASSIST 採点キャッシュ読み込み: {len(scores)} ペア")

    method_order = [
        "baseline",
        "rcca_reg8",
        "rcca_pca_mood10",
        "light_hyb25",
        "p1_linear",
    ]
    methods: dict[str, list[dict[str, Any]]] = {m: [] for m in method_order}
    missing_pairs: dict[tuple[str, str], tuple[str, str]] = {}

    for i in range(len(j_3p)):
        q = j_3p[i]["query"]
        per_m = {
            "baseline": j_3p[i]["top3"]["baseline"],
            "rcca_reg8": j_3p[i]["top3"]["rcca_reg8"],
            "rcca_pca_mood10": j_3p[i]["top3"]["rcca_pca_mood10"],
            "light_hyb25": j_v2[i]["top3"]["light_hyb25"],
            "p1_linear": j_v1[i]["top3"]["p1_linear"],
        }
        for mname, docs in per_m.items():
            methods[mname].append({"query": q, "docs": docs})
            for d in docs:
                pk = (q["key"], d["key"])
                if pk not in scores and pk not in missing_pairs:
                    missing_pairs[pk] = (q["content"], d["content"])

    log(f"追加採点が必要なユニークペア数: {len(missing_pairs)} 件")

    if missing_pairs:
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
                    score_one(client, sem, "http://127.0.0.1:8000", "qwen3.6-35b-a3b", pk, qt, dt)
                )
                for pk, (qt, dt) in missing_pairs.items()
            ]
            done = 0
            for coro in asyncio.as_completed(tasks):
                pk, res = await coro
                scores[pk] = res
                done += 1
                if done % 150 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    log(f"  [{done}/{len(tasks)}] {done / max(elapsed, 1e-6):.1f} req/s ({elapsed:.1f}s)")

    print("\n" + "=" * 116)
    print("DRAFT_ASSIST 準拠・厳格絶対品質評価（rCCA 系 vs 007 タグ一致蒸留系 横並び比較）")
    print("=" * 116)
    print(
        f"{'手法':<18} {'平均点(/3)':>10} {'妥当率(≥2)':>11} {'強共鳴(3)':>10} {'ハズレ(0)':>10} "
        f"{'Top3満足率':>11} {'#1妥当率':>9} | {'0 / 1 / 2 / 3 分布':>22}"
    )
    print("-" * 116)
    for mname in method_order:
        qlist = methods[mname]
        all_s: list[int] = []
        top1_s: list[int] = []
        q_has_good = 0
        for item in qlist:
            qk = item["query"]["key"]
            row_s = [scores[(qk, d["key"])]["score"] for d in item["docs"] if (qk, d["key"]) in scores]
            all_s.extend(row_s)
            top1_s.append(row_s[0])
            if any(s >= 2 for s in row_s):
                q_has_good += 1
        n_tot = len(all_s) or 1
        mean_s = sum(all_s) / n_tot
        valid_rate = sum(1 for s in all_s if s >= 2) / n_tot * 100
        strong_rate = sum(1 for s in all_s if s == 3) / n_tot * 100
        zero_rate = sum(1 for s in all_s if s == 0) / n_tot * 100
        sat_rate = q_has_good / len(qlist) * 100
        t1_rate = sum(1 for s in top1_s if s >= 2) / max(len(top1_s), 1) * 100
        dist = [sum(1 for s in all_s if s == v) for v in (0, 1, 2, 3)]
        dist_str = f"{dist[0]:3d} / {dist[1]:3d} / {dist[2]:3d} / {dist[3]:3d}"
        print(
            f"{mname:<18} {mean_s:10.3f} {valid_rate:10.1f}% {strong_rate:9.1f}% {zero_rate:9.1f}% "
            f"{sat_rate:10.1f}% {t1_rate:8.1f}% | {dist_str:>22}"
        )
    print("=" * 116)

    (out_dir / "judge_pointwise_draft_assist.json").write_text(
        json.dumps(
            {"scores": {f"{k[0]}||{k[1]}": v for k, v in scores.items()}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
