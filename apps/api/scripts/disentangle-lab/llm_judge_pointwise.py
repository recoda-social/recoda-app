#!/usr/bin/env python3
"""
DRAFT_ASSIST_QUESTIONS（5 要素: space / time / sound / memory / emotion_or_action + tone_type + concrete_detail）
に完全準拠した厳格 0〜3 点絶対評価（Pointwise Relevance Scoring）スクリプト。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

SYSTEM_PROMPT = """あなたは音楽レビューアプリの類似検索品質を厳格に絶対評価する評価者です。
このアプリではユーザーに、単なる抽象的絶賛（abstract_only）や客観評論（objective_critique）ではなく、以下の5要素を含む具体的な個人体験（personal_experience）を書くよう促しています：
- space: 場所・景色・天気・環境・共有空間（部屋、雨の窓、帰り道、ドライブ、ライブ会場など）
- time: 時間帯・季節・生活のタイミング（深夜、朝の通勤、終電後、夏など）
- sound: 具体的な音楽要素（歌詞のフレーズ、イントロ、サビ、メロディ、楽器、歌声、アレンジなど）
- memory: 過去の記憶・出来事・懐かしさ・思い出す人物（学生時代、友人、家族、過去の自分など）
- emotion_or_action: 内面の感情変化・身体感覚・行動（落ち着いた、心が軽くなった、足取りが緩んだ、泣けたなど）

「起点の曲レビュー」に対して、検索結果の「別の曲のレビュー」が似た感想としてどれだけ妥当かを、以下の **厳格な 0〜3 点** で採点してください。

採点基準:
- 3点（強く共鳴）: 起点レビューの核となる軸（上記5要素のうち起点が具体的に語っている軸）において、**具体的な描写（concrete_detail）と感情・行動変化の向きまで深く一致**しており、「同じ体験・感性を持つ別の人の投稿」として強く共鳴する。
- 2点（妥当）: 起点レビューの核となる軸と感情の向きが共通しており、語り口（個人体験 vs 客観評論）も大きくズレず、別曲の似た感想として十分に納得できる。
- 1点（弱い関連・惜しい）: 大枠の雰囲気や表層語だけ近いが、**核となる軸がズレている**（例：具体的な個人体験に対して客観評論や抽象的な絶賛「最高！名曲！」のみ、シーン描写に対して単なる歌詞評のみ）。
- 0点（ハズレ）: 核となる軸も噛み合わず無関係、または **感情の向き（快／不快、ユーモア／ダーク、悲劇／甘さ等）が真逆**。

注意:
- 曲名・アーティスト名が違うのは正常（別曲から似た感想を探す機能）。
- 単に「どちらも良い曲・感動した」だけの表面的な一致は 1 点以下とする。
- reason は必ず 35文字以内の短い1文で書くこと。
必ず指定JSONスキーマのみで出力してください。"""


class PointwiseScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: Literal[0, 1, 2, 3]
    reason: str


SCORE_SCHEMA = PointwiseScore.model_json_schema()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def score_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    pair_key: tuple[str, str],
    q_text: str,
    d_text: str,
    max_retries: int = 4,
) -> tuple[tuple[str, str], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"【起点レビュー】\n{q_text}\n\n【検索された別曲レビュー】\n{d_text}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "PointwiseScore",
                "strict": True,
                "schema": SCORE_SCHEMA,
            },
        },
    }
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                parsed = PointwiseScore.model_validate_json(
                    resp.json()["choices"][0]["message"]["content"]
                )
                return pair_key, {"score": int(parsed.score), "reason": parsed.reason}
            except Exception as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(f"Failed {pair_key}: {exc}") from exc
                await asyncio.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    items = json.loads(args.input.read_text(encoding="utf-8"))
    method_names = list(items[0]["top3"].keys())
    methods: dict[str, list[dict[str, Any]]] = {m: [] for m in method_names}
    unique_pairs: dict[tuple[str, str], tuple[str, str]] = {}

    for it in items:
        q = it["query"]
        for mname in method_names:
            docs = it["top3"][mname]
            methods[mname].append({"query": q, "docs": docs})
            for d in docs:
                pk = (q["key"], d["key"])
                if pk not in unique_pairs:
                    unique_pairs[pk] = (q["content"], d["content"])

    log(f"{len(items)} クエリ × {len(method_names)} 手法 → ユニーク (query, doc) ペア数: {len(unique_pairs)} 件")

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency + 16,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(180.0, connect=30.0)

    scores: dict[tuple[str, str], dict[str, Any]] = {}
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for wait_i in range(60):
            try:
                r = await client.get(f"{args.base_url}/v1/models")
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
                score_one(client, sem, args.base_url, args.model, pk, qt, dt)
            )
            for pk, (qt, dt) in unique_pairs.items()
        ]
        done = 0
        failed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                pk, res = await coro
                scores[pk] = res
            except Exception as exc:
                failed += 1
                log(f"  [WARN] {exc}")
            done += 1
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                log(
                    f"  [{done}/{len(tasks)}] ok={len(scores)} fail={failed} | "
                    f"{done / max(elapsed, 1e-6):.1f} req/s ({elapsed:.1f}s)"
                )

    print("\n" + "=" * 116)
    print(f"DRAFT_ASSIST 準拠・厳格絶対品質評価（0〜3 点、完全未見 {len(items)} クエリ × Top-3 = 各 {len(items)*3} ペア）")
    print("=" * 116)
    print(
        f"{'手法':<18} {'平均点(/3)':>10} {'妥当率(≥2)':>11} {'強共鳴(3)':>10} {'ハズレ(0)':>10} "
        f"{'Top3満足率':>11} {'#1妥当率':>9} | {'0 / 1 / 2 / 3 分布':>22}"
    )
    print("-" * 116)
    summary: dict[str, Any] = {}
    for mname, qlist in methods.items():
        all_s: list[int] = []
        top1_s: list[int] = []
        q_has_good = 0
        for item in qlist:
            qk = item["query"]["key"]
            row_s = [scores[(qk, d["key"])]["score"] for d in item["docs"] if (qk, d["key"]) in scores]
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
        sat_rate = q_has_good / len(qlist) * 100
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
    print("=" * 116)

    args.output.write_text(
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
    log(f"詳細結果保存: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/judge_candidates_3pillar.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/judge_pointwise_draft_assist.json"),
    )
    parser.add_argument("--concurrency", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
