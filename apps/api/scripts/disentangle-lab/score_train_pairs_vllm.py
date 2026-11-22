#!/usr/bin/env python3
"""
縦の論理・横の論理（6次元 MECE 4軸）・ピラミッドストラクチャーに基づく
StrictResonanceJudge (Pydantic) で学習ペア 8,000 件をローカル vLLM 並列採点するスクリプト。
JSONL 追記保存＋自動レジューム対応。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

Axis = Literal["situation", "memory", "musical_focus", "emotion_shift"]


class StrictResonanceJudge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    core_axes_of_a: list[Axis] = Field(min_length=1, max_length=3)
    axis_match: Literal["deep", "partial", "none"]
    emotion_polarity: Literal["aligned", "neutral", "opposite"]
    concrete_and_tone: Literal["concrete_match", "generic_or_mismatch"]
    score: Literal[0, 1, 2, 3]
    reason: str


JUDGE_SCHEMA = StrictResonanceJudge.model_json_schema()

SYSTEM_PROMPT = """あなたは音楽レビューアプリの類似検索を厳格に評価する審判です。
「起点レビュー A を書き終えた直後の聴き手本人」の視点で、別曲のレビュー B がどれだけ強く共鳴するかをピラミッド構造で段階的に判定し、JSON のみで出力してください。

# 絶対ルール（縦の論理：飛躍・同質化・前提ズレの禁止）
1. **前提条件**: 曲名・アーティスト・タイアップ作品名が違うのは当然（これらの一致は加点しない）。A が具体的な個人体験なのに B が抽象絶賛のみ（「最高！泣ける！」等）や客観評論のみなら `generic_or_mismatch` とする。
2. **異質なものの同質化を禁止**: 粗いカテゴリ（歌詞・恋愛・音域・思い出）が同じでも、中身や向きが異質なら一致としない（例：ユーモア vs ダーク、片思いのワクワク vs 失恋の喪失、歌いやすい低音 vs 難しすぎる高音）。
3. **偶然の必然化を禁止**: 単語の偶然一致（「格闘技」と「筋トレ」等）や文字数の一致から飛躍せず、A と B の具体的な記述同士の対応が説明できない場合は 1 点以下とする。

# 評価の 4 軸（MECE・ダブりなし）
- `situation`: 今・いつ・どこで聴いているか（深夜・帰り道・雨・ドライブ・部屋・季節など。※過去の人生出来事は memory へ分離）
- `memory`: 誰との・どんな時期の・どんな過去の出来事か（学生時代・恋愛・別れ・死別・病気・挫折・家族・時代）
- `musical_focus`: 曲のどこに心を掴まれたか（歌詞のフレーズ/物語・歌声・メロディ/楽器/リズム・MV/ライブ演出）
- `emotion_shift`: 聴いた結果どの方向へ心が動いたか・どう行動したか（軽くなる・泣ける・背中を押される・切ない・ゾッとする・笑える等の極性と変化）

# スコア決定ルール（上記ステップから論理的に決定）
- **3点（強共鳴）**: `axis_match == "deep"` かつ `emotion_polarity == "aligned"` かつ `concrete_and_tone == "concrete_match"`（異質同質化・偶然一致なし）
- **2点（妥当）**: `axis_match` が `"deep"` または `"partial"`、かつ `emotion_polarity != "opposite"`、かつ核となる軸に明確な共通点がある
- **1点（惜しい・表層のみ）**: 表層語や粗いカテゴリだけ近いが具体の中身が異質、または `concrete_and_tone == "generic_or_mismatch"`
- **0点（ハズレ）**: `axis_match == "none"`（無関係・偶然の単語一致のみ）、または `emotion_polarity == "opposite"`（感情の向きが真逆）

※ `reason` は必ず 35 文字以内の短い 1 文で具体的な対応根拠を書くこと。"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    return f"{m}m{s:02d}s" if m > 0 else f"{s}s"


async def judge_pair(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    item: dict[str, Any],
    max_retries: int = 4,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"【起点レビュー A】\n{item['q_text']}\n\n【別曲レビュー B】\n{item['d_text']}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 320,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "StrictResonanceJudge",
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
                parsed = StrictResonanceJudge.model_validate_json(
                    resp.json()["choices"][0]["message"]["content"]
                )
                return {
                    "q_key": item["q_key"],
                    "d_key": item["d_key"],
                    "source": item.get("source", ""),
                    **parsed.model_dump(),
                }
            except Exception as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(
                        f"Failed {item['q_key']}||{item['d_key']}: {exc}"
                    ) from exc
                await asyncio.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    items: list[dict[str, Any]] = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str]] = set()
    dist = [0, 0, 0, 0]
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done_keys.add((rec["q_key"], rec["d_key"]))
                dist[int(rec["score"])] += 1
        log(f"既存 JSONL から {len(done_keys)} 件をレジューム読み込み (分布 0/1/2/3={dist})")

    pending = [it for it in items if (it["q_key"], it["d_key"]) not in done_keys]
    log(f"残り採点対象: {len(pending)} 件 / 合計 {len(items)} 件 | concurrency={args.concurrency}")

    if pending:
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
            tasks = [
                asyncio.create_task(
                    judge_pair(client, sem, args.base_url, args.model, it)
                )
                for it in pending
            ]
            ok = 0
            fail = 0
            with args.output.open("a", encoding="utf-8") as out_f:
                for coro in asyncio.as_completed(tasks):
                    try:
                        rec = await coro
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        dist[int(rec["score"])] += 1
                        ok += 1
                    except Exception as exc:
                        fail += 1
                        log(f"  [WARN] {exc}")
                    done_now = ok + fail
                    if done_now % args.log_every == 0 or done_now == len(pending):
                        out_f.flush()
                        elapsed = time.time() - t0
                        speed = done_now / max(elapsed, 1e-6)
                        eta = (len(pending) - done_now) / max(speed, 1e-6)
                        log(
                            f"[{done_now}/{len(pending)} ({done_now/len(pending)*100:5.1f}%)] "
                            f"ok={ok} fail={fail} | {speed:.1f} req/s | "
                            f"elapsed={fmt_duration(elapsed)} ETA={fmt_duration(eta)} | "
                            f"dist(0/1/2/3)={dist}"
                        )

    total = sum(dist) or 1
    log(
        f"採点完了! 合計 {sum(dist)} 件 | "
        f"0点:{dist[0]}({dist[0]/total*100:.1f}%) / "
        f"1点:{dist[1]}({dist[1]/total*100:.1f}%) / "
        f"2点:{dist[2]}({dist[2]/total*100:.1f}%) / "
        f"3点:{dist[3]}({dist[3]/total*100:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/train_pairs_800.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/api/scripts/disentangle-lab/out/train_pairs_scored.jsonl"),
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
