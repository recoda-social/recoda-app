#!/usr/bin/env python3
"""
ローカル vLLM (Qwen3.6-35B-A3B-NVFP4-Fast) によるブラインド・位置バイアス除去済み
LLM-as-a-Judge 評価スクリプト（記述内容の直接比較）。

各ペア比較（p1_linear vs baseline / p2_soft vs baseline / p2_soft vs p1_linear）を
正順 (A vs B) と逆順 (B vs A) の両方向で判定し、位置バイアスを完全に除去して集計する。
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

SYSTEM_PROMPT = """あなたは音楽レビューアプリの類似検索品質を評価する厳正な審判です。
「起点の曲レビュー」に対して、2つの検索システム（システムA・システムB）がそれぞれ返した上位3件の類似レビューを比較し、
**記述されている具体的な内容（歌詞・サウンドの着眼点、聴いたシーンや記憶の中身、感情・余韻のニュアンス）が起点レビューとより深く共鳴・一致している方**を選んでください。

評価基準:
1. 曲名やアーティスト名が違うのは正常（別曲から似た感想を探す機能）。
2. 文字数や文体の似通いだけ（例：長いレビュー同士というだけ）で選ばず、「書かれている体験・観点・感情の中身」が近いかを最重視する。
3. 3件中2〜3件が的確に噛み合っている方を高く評価する。
4. ほぼ互角・優劣つけがたい場合のみ "tie" を選ぶ。
5. reason は必ず **40文字以内の短い1文** で簡潔に書くこと（長文禁止）。

必ず指定JSONスキーマのみで出力してください。"""


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner: Literal["A", "B", "tie"]
    reason: str


JUDGE_SCHEMA = JudgeResult.model_json_schema()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_list(posts: list[dict[str, Any]]) -> str:
    return "\n".join(f"  {i}. {p['content']}" for i, p in enumerate(posts, 1))


async def judge_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    pair_name: str,
    idx: int,
    direction: str,
    query_text: str,
    list_a: list[dict[str, Any]],
    list_b: list[dict[str, Any]],
    max_retries: int = 4,
) -> dict[str, Any]:
    user_msg = (
        f"【起点レビュー】\n{query_text}\n\n"
        f"【システムAの上位3件】\n{fmt_list(list_a)}\n\n"
        f"【システムBの上位3件】\n{fmt_list(list_b)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 384,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "JudgeResult",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            },
        },
    }
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                parsed = JudgeResult.model_validate_json(
                    resp.json()["choices"][0]["message"]["content"]
                )
                return {
                    "pair": pair_name,
                    "idx": idx,
                    "direction": direction,
                    "winner": parsed.winner,
                    "reason": parsed.reason,
                }
            except Exception as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(f"Judge failed {pair_name}#{idx}({direction}): {exc}") from exc
                await asyncio.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    items = json.loads(args.input.read_text(encoding="utf-8"))
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    pairs = [(f"{a} vs {b}", a, b) for a, b in pairs]

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency + 16,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(180.0, connect=30.0)

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
        tasks = []
        skipped_identical = {p[0]: 0 for p in pairs}
        for pname, m_left, m_right in pairs:
            for idx, it in enumerate(items):
                la = it["top3"][m_left]
                lb = it["top3"][m_right]
                # 完全に同じ Top-3 なら API を呼ばず引き分け扱い
                if [x["key"] for x in la] == [x["key"] for x in lb]:
                    skipped_identical[pname] += 1
                    continue
                qtxt = it["query"]["content"]
                tasks.append(
                    asyncio.create_task(
                        judge_one(client, sem, args.base_url, args.model, pname, idx, "fwd", qtxt, la, lb)
                    )
                )
                tasks.append(
                    asyncio.create_task(
                        judge_one(client, sem, args.base_url, args.model, pname, idx, "rev", qtxt, lb, la)
                    )
                )

        log(f"合計 {len(tasks)} 判定を並列 {args.concurrency} で実行開始...")
        raw_results: list[dict[str, Any]] = []
        done = 0
        failed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                raw_results.append(res)
            except Exception as exc:
                failed += 1
                log(f"  [WARN] {exc}")
            done += 1
            if done % 150 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                log(f"  [{done}/{len(tasks)}] ok={len(raw_results)} fail={failed} | {done / max(elapsed, 1e-6):.1f} req/s ({elapsed:.1f}s)")

    # 位置バイアス除去集計（fwd の A == rev の B が左勝利）
    by_pair: dict[str, dict[int, dict[str, Any]]] = {p[0]: {} for p in pairs}
    for r in raw_results:
        by_pair[r["pair"]].setdefault(r["idx"], {})[r["direction"]] = r

    print("\n" + "=" * 100)
    print(f"ブラインド LLM-as-a-Judge 記述内容評価（N = {len(items)} 件、正逆両方向判定・位置バイアス除去）")
    print("=" * 100)
    print(
        f"{'比較 (左 vs 右)':<26} {'左勝':>6} {'引分/不一致':>11} {'右勝':>6} | "
        f"{'左勝率(除引分)':>14} {'正逆一致率':>10}"
    )
    print("-" * 100)

    summary_out: dict[str, Any] = {}
    for pname, m_left, m_right in pairs:
        left_w = right_w = tie_or_inc = consistent = 0
        sample_wins: list[str] = []
        sample_losses: list[str] = []
        for idx, dirs in sorted(by_pair[pname].items()):
            fwd = dirs.get("fwd")
            rev = dirs.get("rev")
            if not fwd or not rev:
                continue
            # 左モデル視点のスコア: +1 (左勝), 0 (tie), -1 (右勝)
            s_fwd = 1 if fwd["winner"] == "A" else (-1 if fwd["winner"] == "B" else 0)
            s_rev = 1 if rev["winner"] == "B" else (-1 if rev["winner"] == "A" else 0)
            if s_fwd == s_rev:
                consistent += 1
            tot = s_fwd + s_rev
            if tot > 0:
                left_w += 1
                if len(sample_wins) < 3 and tot == 2:
                    sample_wins.append(f"#{idx}: {fwd['reason']}")
            elif tot < 0:
                right_w += 1
                if len(sample_losses) < 3 and tot == -2:
                    sample_losses.append(f"#{idx}: {fwd['reason']}")
            else:
                tie_or_inc += 1
        tie_or_inc += skipped_identical[pname]
        decided = left_w + right_w
        win_rate = (left_w / decided * 100) if decided > 0 else 50.0
        n_judged = len(by_pair[pname])
        cons_rate = (consistent / n_judged * 100) if n_judged > 0 else 100.0
        print(
            f"{pname:<26} {left_w:6d} {tie_or_inc:11d} {right_w:6d} | "
            f"{win_rate:13.1f}% {cons_rate:9.1f}%"
        )
        summary_out[pname] = {
            "left_win": left_w,
            "tie_or_inconsistent": tie_or_inc,
            "right_win": right_w,
            "win_rate_excl_tie": round(win_rate, 2),
            "consistency_rate": round(cons_rate, 2),
            "sample_left_wins": sample_wins,
            "sample_right_wins": sample_losses,
        }
    print("=" * 100)
    for pname, info in summary_out.items():
        print(f"\n--- {pname} の代表的な判定理由 ---")
        for w in info["sample_left_wins"][:2]:
            print(f"  [左勝] {w}")
        for l in info["sample_right_wins"][:2]:
            print(f"  [右勝] {l}")

    args.output.write_text(json.dumps({"summary": summary_out, "raw": raw_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"詳細ログ保存: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/judge_candidates.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/judge_results.json"),
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument(
        "--pairs",
        default="step1_only:baseline,light_tag20:baseline,light_hyb25:baseline",
        help="カンマ区切りの left:right ペア指定",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
