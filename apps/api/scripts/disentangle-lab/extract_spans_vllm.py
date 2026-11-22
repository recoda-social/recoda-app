#!/usr/bin/env python3
"""
ローカル vLLM (Qwen3.6-35B-A3B-NVFP4-Fast) の OpenAI 互換 API に非同期並列送信し、
曲投稿（packages/db/data/official-reviews.json）から 4 スロットの原文スパン（節・文単位）を抽出する。

特徴:
- Structured Output (json_schema, 空白許可) で 4 キー JSON を安定抽出
- 最大 concurrency 件の非同期並列リクエスト + 自動リトライ（サーバー再起動にも対応）
- JSONL へ逐次追記・自動レジューム
- リアルタイム進捗・スループット・ETA・スロットカバレッジ自動集計

実行例:
  # 200 件パイロット
  python apps/api/scripts/disentangle-lab/extract_spans_vllm.py \
    --limit 200 --concurrency 128 \
    --output apps/api/scripts/disentangle-lab/out/spans_pilot200_qwen36.jsonl

  # 全 3,678 件
  python apps/api/scripts/disentangle-lab/extract_spans_vllm.py \
    --limit 0 --concurrency 256 \
    --output apps/api/scripts/disentangle-lab/out/spans_all_qwen36.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

SLOTS = ("music_span", "meta_span", "context_span", "mood_span")

SYSTEM_PROMPT = """あなたは音楽の感想文（最大140文字）から、以下の4つの観点に対応する表現を原文から抜き出す抽出器です。
必ず指定されたJSONスキーマのみで出力してください。

各スロットの定義:
- music_span: 歌詞・言葉選び・歌声・歌唱力・メロディ・リズム・楽器・アレンジ・音作りなど、曲そのものの聴覚的・言語的特徴について述べている部分。
- meta_span: アニメ・ドラマ・映画・CM・ゲーム等のタイアップ作品、MV・映像・ダンス演出、ライブ・フェス体験、アーティストやメンバー本人について述べている部分。
- context_span: 聴いていた時間帯・季節・天気・場所・移動シーン（ドライブ・電車・帰り道等）・日常シーン、または学生時代・恋愛・家族・友人・苦労・人生の節目・昔などの個人的な記憶や体験を述べている部分。
- mood_span: 聴いたときの感情・気持ち・雰囲気・余韻（泣ける・感動・切ない・懐かしい・癒やされる・温かい・励まされる・盛り上がる・かっこいい・かわいい・面白い・中毒性がある等）を表している部分。

ルール:
1. 可能な限り原文の表現をそのまま抜き出す（複数箇所ある場合は「 / 」でつなぐ）。
2. そのスロットに該当する記述が感想文に無い場合は、無理に作らず空文字 "" にする。
3. 1つの表現が2つのスロットにまたがる場合は、より中心的な方に含めてよい。"""

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "music_span": {"type": "string"},
        "meta_span": {"type": "string"},
        "context_span": {"type": "string"},
        "mood_span": {"type": "string"},
    },
    "required": ["music_span", "meta_span", "context_span", "mood_span"],
    "additionalProperties": False,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def summarize(records: list[dict[str, Any]]) -> None:
    n = len(records)
    if n == 0:
        return
    print("\n" + "=" * 88)
    print(f"4 スロット原文スパン抽出サマリ（N = {n} 件）")
    print("=" * 88)
    for slot in SLOTS:
        non_empty = [r["spans"][slot] for r in records if r["spans"].get(slot, "").strip()]
        cov = len(non_empty) / n * 100
        avg_len = sum(len(s) for s in non_empty) / len(non_empty) if non_empty else 0.0
        print(
            f"  {slot:<14}: カバレッジ {cov:5.1f}% ({len(non_empty):4d}/{n}) | "
            f"平均文字数 {avg_len:4.1f}ch"
        )
    print("\n--- サンプル 5 件 ---")
    for r in records[:5]:
        print("-" * 88)
        print(f"KEY: {r['key']} ({len(r['content'])}ch) | {r['content']}")
        for slot in SLOTS:
            val = r["spans"].get(slot, "").strip()
            if val:
                print(f"  {slot:<14}: {val}")
    print("=" * 88 + "\n", flush=True)


async def extract_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    post: dict[str, Any],
    max_retries: int = 5,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"感想文:\n{post['content']}"},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "song_review_slots",
                "strict": True,
                "schema": JSON_SCHEMA,
            },
        },
    }
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_text)
                spans = {s: str(parsed.get(s, "")).strip() for s in SLOTS}
                return {
                    "key": post["key"],
                    "isrc": post["isrc"],
                    "content": post["content"],
                    "spans": spans,
                }
            except Exception as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(f"Failed for {post['key']}: {exc}") from exc
                await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    posts: list[dict[str, Any]] = raw["posts"]
    if args.limit > 0 and args.limit < len(posts):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(posts)), args.limit))
        target_posts = [posts[i] for i in indices]
        log(f"全 {len(posts)} 件からシード {args.seed} で {len(target_posts)} 件をサンプリング")
    else:
        target_posts = posts
        log(f"全 {len(target_posts)} 件を対象に設定")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_records: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done_records[rec["key"]] = rec
        log(f"既存出力から {len(done_records)} 件をレジューム読み込み")

    pending = [p for p in target_posts if p["key"] not in done_records]
    log(
        f"残り処理対象: {len(pending)} 件 / 合計 {len(target_posts)} 件 | "
        f"endpoint={args.base_url} model={args.model} concurrency={args.concurrency}"
    )

    if pending:
        sem = asyncio.Semaphore(args.concurrency)
        limits = httpx.Limits(
            max_connections=args.concurrency + 16,
            max_keepalive_connections=args.concurrency,
        )
        timeout = httpx.Timeout(180.0, connect=30.0)
        t_start = time.time()
        completed = 0
        failed = 0

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            # 疎通確認
            models_resp = await client.get(f"{args.base_url}/v1/models")
            models_resp.raise_for_status()
            log(f"vLLM 疎通確認 OK: {[m['id'] for m in models_resp.json().get('data', [])]}")

            tasks = [
                asyncio.create_task(extract_one(client, sem, args.base_url, args.model, p))
                for p in pending
            ]
            with args.output.open("a", encoding="utf-8") as out_f:
                for coro in asyncio.as_completed(tasks):
                    try:
                        rec = await coro
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done_records[rec["key"]] = rec
                        completed += 1
                    except Exception as exc:
                        failed += 1
                        log(f"[ERROR] {exc}")
                    if (completed + failed) % args.log_every == 0 or (
                        completed + failed
                    ) == len(pending):
                        out_f.flush()
                        elapsed = time.time() - t_start
                        done_now = completed + failed
                        speed = done_now / elapsed if elapsed > 0 else 0.0
                        eta = (len(pending) - done_now) / speed if speed > 0 else 0.0
                        pct = done_now / len(pending) * 100
                        log(
                            f"[{done_now}/{len(pending)} ({pct:5.1f}%)] "
                            f"ok={completed} fail={failed} | {speed:.2f} post/s | "
                            f"elapsed={fmt_duration(elapsed)} ETA={fmt_duration(eta)}"
                        )

    ordered = [done_records[p["key"]] for p in target_posts if p["key"] in done_records]
    summarize(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM による 4 スロット原文スパン抽出")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("packages/db/data/official-reviews.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
