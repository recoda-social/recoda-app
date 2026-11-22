#!/usr/bin/env python3
"""
score_train_pairs_vllm.py と完全同一の StrictResonanceJudge (縦・横 MECE 4軸・ピラミッド構造) で、
完全未見 150 クエリ × 4 手法（baseline / rcca_reg8 / metric_linear / metric_mlp）を絶対評価する。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from score_train_pairs_vllm import judge_pair, log


async def main() -> None:
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    items = json.loads((out_dir / "judge_candidates_008.json").read_text(encoding="utf-8"))
    method_names = list(items[0]["top3"].keys())
    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}

    for it in items:
        q = it["query"]
        for m in method_names:
            for d in it["top3"][m]:
                pk = (q["key"], d["key"])
                if pk not in unique_pairs:
                    unique_pairs[pk] = {
                        "q_key": q["key"],
                        "d_key": d["key"],
                        "q_text": q["content"],
                        "d_text": d["content"],
                    }

    log(f"完全未見 {len(items)} クエリ × {len(method_names)} 手法 → ユニーク {len(unique_pairs)} ペアを StrictResonanceJudge で採点")

    sem = asyncio.Semaphore(256)
    limits = httpx.Limits(max_connections=272, max_keepalive_connections=256)
    timeout = httpx.Timeout(180.0, connect=30.0)
    scores: dict[tuple[str, str], dict[str, Any]] = {}

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
            for payload in unique_pairs.values()
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
                elapsed = time.time() - t0
                log(f"  [{done}/{len(tasks)}] ok={len(scores)} fail={fail} | {done/max(elapsed,1e-6):.1f} req/s ({elapsed:.1f}s)")

    print("\n" + "=" * 120)
    print(f"StrictResonanceJudge（ピラミッド MECE 4 軸・完全未見 {len(items)} クエリ × Top-3 = 各 {len(items)*3} ペア）")
    print("=" * 120)
    print(
        f"{'手法':<16} {'平均点(/3)':>10} {'妥当率(≥2)':>11} {'強共鳴(3)':>10} {'ハズレ(0)':>10} "
        f"{'Top3満足率':>11} {'#1妥当率':>9} | {'0 / 1 / 2 / 3 分布':>22}"
    )
    print("-" * 120)
    summary: dict[str, Any] = {}
    for mname in method_names:
        all_s: list[int] = []
        top1_s: list[int] = []
        q_has_good = 0
        for it in items:
            qk = it["query"]["key"]
            row_s = [scores[(qk, d["key"])]["score"] for d in it["top3"][mname] if (qk, d["key"]) in scores]
            if not row_s:
                continue
            all_s.extend(row_s)
            top1_s.append(row_s[0])
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
            f"{mname:<16} {mean_s:10.3f} {valid_rate:10.1f}% {strong_rate:9.1f}% {zero_rate:9.1f}% "
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
    print("=" * 120)

    out_path = out_dir / "judge_results_008.json"
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "scores": {f"{k[0]}||{k[1]}": v for k, v in scores.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"詳細結果保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
