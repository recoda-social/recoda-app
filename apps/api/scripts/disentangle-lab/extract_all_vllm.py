#!/usr/bin/env python3
"""
plans/006-embedding-disentangle-plan.md Section 3 準拠の
vLLM (Qwen3.6-35B-A3B-NVFP4-Fast) × Pydantic Structured Outputs 抽出スクリプト。

1 投稿ごとに 4 スロット（music / meta / context / mood）の
{ tag, details: list[str], primary: bool } を Pydantic スキーマで厳密抽出し、
JSONL へ逐次保存（自動レジューム対応）する。

200 件試験抽出時の自動チェック（計画 Section 2）:
  1. details への曲名（title）・アーティスト名（artistName）混入率
  2. タグ出現率（3% 未満の剪定候補 / 3〜5% の統合検討）および primary 率
  3. 4 スロットのカバレッジ・平均 details 文字数・目視確認サンプル出力

実行例:
  # 200 件試験抽出
  python apps/api/scripts/disentangle-lab/extract_all_vllm.py \
    --limit 200 --seed 42 --concurrency 128 \
    --output apps/api/scripts/disentangle-lab/out/pilot200_qwen36.jsonl

  # 全 3,678 件一括抽出
  python apps/api/scripts/disentangle-lab/extract_all_vllm.py \
    --limit 0 --concurrency 256 \
    --output apps/api/scripts/disentangle-lab/out/all_qwen36.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Pydantic Structured Outputs スキーマ定義（plans/006 Section 3 準拠）
# ---------------------------------------------------------------------------

MusicTag = Literal["lyrics", "vocal", "sound"]
MetaTag = Literal["tieup_visual", "artist_live"]
ContextTag = Literal[
    "scene_time",
    "scene_move",
    "scene_daily",
    "mem_youth",
    "mem_love",
    "mem_hardship",
    "mem_people_turn",
    "mem_era",
]
MoodTag = Literal[
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
]

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


class MusicEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag: MusicTag
    details: list[str] = Field(min_length=1, max_length=4)
    primary: bool


class MetaEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag: MetaTag
    details: list[str] = Field(min_length=1, max_length=4)
    primary: bool


class ContextEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag: ContextTag
    details: list[str] = Field(min_length=1, max_length=4)
    primary: bool


class MoodEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag: MoodTag
    details: list[str] = Field(min_length=1, max_length=4)
    primary: bool


class SlotEntries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    music: list[MusicEntry]
    meta: list[MetaEntry]
    context: list[ContextEntry]
    mood: list[MoodEntry]


class ReviewExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    entries: SlotEntries


# ---------------------------------------------------------------------------
# System Prompt（plans/006 Section 3 そのまま）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """あなたは音楽レビューの「書かれている内容」を構造化抽出する専門のアノテーターです。
与えられた各曲レビュー（日本語）を読み、4 つのスロット（music / meta / context / mood）に該当するタグと、その具体的な根拠・中身（details）を JSON で抽出してください。

# 絶対ルール

1. **原文にある内容のみ抽出する（推測・補完は厳禁）**
   - 本文に書かれていない要素は絶対に補わず、該当がないスロットは空配列 `[]` にする。
   - 根拠となる `details` を書けないタグは付けない。
2. **曲名・アーティスト名・メンバー名・愛称を `details` に入れない**
   - 入力に与えられる `title` / `artistName` や、本文中のメンバー名・愛称・タイアップ固有作品名は取り除くか、「ボーカル」「アーティスト」「アニメの主題歌」のように一般化する。
3. **`details` の書き方（文量・挨拶ノイズを落とす）**
   - 各要素は **10〜35 文字程度の簡潔な短句** に整える（「最高」「大好き」「おすすめ」だけの空疎な称賛や挨拶は削る）。
   - 同じタグに該当する具体的な記述が本文中の複数箇所にある場合は、同じ `tag` を重複させず **1 エントリの `details` 配列に複数要素** 入れる。
4. **`primary` の基準**
   - その投稿の中で「一番語りたい中心テーマ」にあたるエントリに `primary: true`、補足的な言及には `primary: false` を付ける（1 投稿全体で `primary: true` は概ね 1〜3 個）。
5. **1 投稿内で各 `tag` は最大 1 回のみ**出力する。

# スロットとタグ定義（境界ルール付き）

## 1. music（曲そのもの：聴覚・言語）

- `lyrics`: 歌詞のフレーズ、言葉選び、メッセージ性、物語・世界観
- `vocal`: 歌声、声質、歌い方、コーラス、歌唱力・表現力
- `sound`: メロディ、コード、楽器（ギター・ベース・ピアノ等）、リズム、テンポ、イントロ・転調・構成

## 2. meta（曲の外側の文脈：視覚・場・人）

- `tieup_visual`: アニメ・ドラマ・映画・CM・ゲーム等のタイアップ文脈、MV・PV・ジャケットの映像演出
- `artist_live`: アーティスト自身のエピソード・姿勢・背景、ライブ・フェス・MC での体験

## 3. context（聴き手のシーン・記憶）

※「いつ・どこで聴いているか（scene_*）」と「人生の記憶・出来事（mem_*）」を明確に分ける。

- `scene_time`: 時間帯（夜・深夜・朝・夕方）や季節・天気（夏・冬・雨・雪など）の情景
- `scene_move`: 通勤・通学・帰り道・散歩・ドライブ・旅行など移動中のシーン
- `scene_daily`: 部屋・一人時間・勉強・作業・就寝前など日常の定位置でのシーン
- `mem_youth`: 学生時代・青春・部活・受験・学校行事の記憶
- `mem_love`: 恋愛・片思い・恋人・失恋・別れの記憶
- `mem_hardship`: 辛い時期・仕事や人生の挫折・悩み・喪失・被災・支えられた経験
- `mem_people_turn`: 家族・友人・大切な人との思い出、卒業・上京・結婚など人生の転機
- `mem_era`: 特定の時代・年代・当時の流行や空気感への言及（個人の時間帯である `scene_time` とは分ける）

## 4. mood（感情・余韻）

※単なる「好き・最高」ではなく、具体的な感情・雰囲気のニュアンスで選ぶ。

- `hype`: アガる、爽快、疾走感、テンションが上がる、盛り上がる
- `cool`: かっこいい、痺れる、圧倒される、鳥肌、洗練
- `calm_warm`: 落ち着く、穏やか、温かい、癒やされる、優しい
- `cute`: かわいい、愛おしい、甘い、キュンとする
- `encouraged`: 励まされる、前向き、背中を押される、勇気・希望が出る
- `moved`: 泣ける、涙が出る、心が震える、深い感動
- `bittersweet`: 切ない、胸が苦しい、儚い、寂しい、哀愁
- `nostalgic`: 懐かしい、昔を思い出す、ノスタルジー
- `funny`: 面白い、笑える、ユーモア、遊び心
- `dark_addictive`: 不穏、ダーク、色気、ミステリアス、中毒性・クセになる

# 出力例（Few-shot）

## 例 1

入力:
{"key":"ex1","title":"春愁","artistName":"Mrs. GREEN APPLE","content":"高校の卒業式が終わった夕方の帰り道、イヤホンで聴きながら歩いたのを今でも覚えている。サビの寂しいメロディと大森さんの切ない歌声に思わず涙がこぼれた。"}

出力:
{
"key": "ex1",
"entries": {
"music": [
{ "tag": "sound", "details": ["サビの寂しいメロディ"], "primary": false },
{ "tag": "vocal", "details": ["切ない歌声"], "primary": false }
],
"meta": [],
"context": [
{ "tag": "scene_time", "details": ["夕方の時間帯"], "primary": false },
{ "tag": "scene_move", "details": ["帰り道にイヤホンで歩きながら聴いた"], "primary": true },
{ "tag": "mem_youth", "details": ["高校の卒業式が終わった日の記憶"], "primary": true },
{ "tag": "mem_people_turn", "details": ["卒業という節目の記憶"], "primary": false }
],
"mood": [
{ "tag": "moved", "details": ["思わず涙がこぼれた"], "primary": true },
{ "tag": "bittersweet", "details": ["寂しさと切なさ"], "primary": false }
]
}
}

## 例 2

入力:
{"key":"ex2","title":"KILLA","artistName":"YOASOBI","content":"名古屋ライブ2日目の「まだ世に出してない新曲、やっちゃっていいんですか！？」というikuraちゃんの煽りがそのまま歌詞だった衝撃！緩急の効いた踊れるサウンドと低音のうねりが最高に痺れる。"}

出力:
{
"key": "ex2",
"entries": {
"music": [
{ "tag": "lyrics", "details": ["ライブでの煽りの言葉がそのまま歌詞になっている"], "primary": false },
{ "tag": "sound", "details": ["緩急の効いた踊れるサウンド", "低音のうねり"], "primary": true }
],
"meta": [
{ "tag": "artist_live", "details": ["ライブでの新曲披露時の煽りの衝撃"], "primary": true }
],
"context": [],
"mood": [
{ "tag": "cool", "details": ["痺れるかっこよさ"], "primary": true },
{ "tag": "hype", "details": ["踊れる高揚感"], "primary": false }
]
}
}"""

REVIEW_SCHEMA = ReviewExtraction.model_json_schema()


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


def summarize(records: list[dict[str, Any]], sample_count: int = 10) -> None:
    n = len(records)
    if n == 0:
        return
    print("\n" + "=" * 96)
    print(f"Qwen3.6 Pydantic 構造化抽出サマリ（N = {n} 件）")
    print("=" * 96)
    print(f"{'スロット':<10} {'タグ':<18} {'出現率':>8} {'primary率':>10}  判定")
    print("-" * 96)
    prune_3: list[str] = []
    prune_5: list[str] = []
    for slot, tags in SLOT_TAGS.items():
        for tag in tags:
            cnt = 0
            pri = 0
            for r in records:
                for e in r["entries"][slot]:
                    if e["tag"] == tag:
                        cnt += 1
                        if e["primary"]:
                            pri += 1
                        break
            rate = cnt / n
            prate = pri / n
            flag = ""
            if rate < 0.03:
                flag = "< 3% (剪定候補)"
                prune_3.append(tag)
            elif rate < 0.05:
                flag = "< 5% (統合検討)"
                prune_5.append(tag)
            print(f"{slot:<10} {tag:<18} {rate * 100:7.1f}% {prate * 100:9.1f}%  {flag}")
    print("-" * 96)
    print(f"剪定候補 (<3%): {prune_3 or 'なし'}")
    print(f"統合検討 (3-5%): {prune_5 or 'なし'}")

    print("\n--- 4 スロット カバレッジ & details 長 ---")
    for slot in SLOTS:
        non_empty = [r["entries"][slot] for r in records if r["entries"][slot]]
        cov = len(non_empty) / n * 100
        all_details = [d for entries in non_empty for e in entries for d in e["details"]]
        avg_len = sum(len(d) for d in all_details) / len(all_details) if all_details else 0.0
        print(
            f"  {slot:<10}: カバレッジ {cov:5.1f}% ({len(non_empty):4d}/{n}) | "
            f"details 数 {len(all_details):4d} | 平均 {avg_len:4.1f}ch"
        )

    # 曲名・アーティスト名混入チェック
    title_leaks: list[tuple[str, str, str]] = []
    artist_leaks: list[tuple[str, str, str]] = []
    primary_counts: list[int] = []
    for r in records:
        title = (r.get("title") or "").strip()
        artist = (r.get("artistName") or "").strip()
        p_cnt = 0
        for slot in SLOTS:
            for e in r["entries"][slot]:
                if e["primary"]:
                    p_cnt += 1
                for d in e["details"]:
                    if len(title) >= 2 and title.lower() in d.lower():
                        title_leaks.append((r["key"], title, d))
                    if len(artist) >= 2 and artist.lower() in d.lower():
                        artist_leaks.append((r["key"], artist, d))
        primary_counts.append(p_cnt)

    avg_pri = sum(primary_counts) / n
    print("\n--- 品質チェック ---")
    print(f"  1 投稿あたり primary:true 平均個数: {avg_pri:.2f} 個")
    print(
        f"  details への曲名混入: {len(title_leaks)} 件 ({len(title_leaks) / n * 100:.1f}%)"
    )
    for k, t, d in title_leaks[:5]:
        print(f"    [title leak] {k} title={t!r} -> {d!r}")
    print(
        f"  details へのアーティスト名混入: {len(artist_leaks)} 件 ({len(artist_leaks) / n * 100:.1f}%)"
    )
    for k, a, d in artist_leaks[:5]:
        print(f"    [artist leak] {k} artist={a!r} -> {d!r}")

    print(f"\n--- 目視サンプル ({min(sample_count, n)} 件) ---")
    for r in records[:sample_count]:
        print("-" * 96)
        print(
            f"KEY: {r['key']} | {r.get('artistName', '')} - {r.get('title', '')} "
            f"({len(r['content'])}ch)"
        )
        print(f"  本文: {r['content']}")
        for slot in SLOTS:
            entries = r["entries"][slot]
            if entries:
                parts = []
                for e in entries:
                    star = "★" if e["primary"] else ""
                    parts.append(f"{star}{e['tag']}:{e['details']}")
                print(f"  {slot:<8}: " + " | ".join(parts))
    print("=" * 96 + "\n", flush=True)


async def extract_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    item: dict[str, Any],
    max_retries: int = 5,
) -> dict[str, Any]:
    user_payload = json.dumps(
        {
            "key": item["key"],
            "title": item["title"],
            "artistName": item["artistName"],
            "content": item["content"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"入力:\n{user_payload}"},
        ],
        "temperature": 0.0,
        "max_tokens": 768,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ReviewExtraction",
                "strict": True,
                "schema": REVIEW_SCHEMA,
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
                parsed = ReviewExtraction.model_validate_json(raw_text)
                # スロット内の重複タグを防御的にマージ
                entries_dict: dict[str, list[dict[str, Any]]] = {}
                for slot in SLOTS:
                    by_tag: dict[str, dict[str, Any]] = {}
                    for e in getattr(parsed.entries, slot):
                        if e.tag not in by_tag:
                            by_tag[e.tag] = {
                                "tag": e.tag,
                                "details": [d.strip() for d in e.details if d.strip()],
                                "primary": bool(e.primary),
                            }
                        else:
                            for d in e.details:
                                ds = d.strip()
                                if ds and ds not in by_tag[e.tag]["details"]:
                                    by_tag[e.tag]["details"].append(ds)
                            by_tag[e.tag]["primary"] = by_tag[e.tag]["primary"] or bool(e.primary)
                    entries_dict[slot] = [v for v in by_tag.values() if v["details"]]
                return {
                    "key": item["key"],
                    "isrc": item["isrc"],
                    "title": item["title"],
                    "artistName": item["artistName"],
                    "content": item["content"],
                    "entries": entries_dict,
                }
            except Exception as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(f"Failed for {item['key']}: {exc}") from exc
                await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    songs_by_isrc = {s["isrc"]: s for s in raw["songs"]}
    posts: list[dict[str, Any]] = []
    for p in raw["posts"]:
        song = songs_by_isrc.get(p["isrc"], {})
        posts.append(
            {
                "key": p["key"],
                "isrc": p["isrc"],
                "title": song.get("title") or "",
                "artistName": song.get("artistName") or "",
                "content": p["content"],
            }
        )

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
            for wait_i in range(60):
                try:
                    models_resp = await client.get(f"{args.base_url}/v1/models")
                    if models_resp.status_code == 200:
                        log(
                            f"vLLM 疎通確認 OK: {[m['id'] for m in models_resp.json().get('data', [])]}"
                        )
                        break
                except Exception:
                    pass
                if wait_i == 0:
                    log("vLLM サーバーの起動完了を待機中...")
                await asyncio.sleep(5.0)
            else:
                raise RuntimeError("vLLM サーバーがタイムアウトまでに起動しませんでした")

            t_start = time.time()
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
                    done_now = completed + failed
                    if done_now % args.log_every == 0 or done_now == len(pending):
                        out_f.flush()
                        elapsed = time.time() - t_start
                        speed = done_now / elapsed if elapsed > 0 else 0.0
                        eta = (len(pending) - done_now) / speed if speed > 0 else 0.0
                        pct = done_now / len(pending) * 100
                        log(
                            f"[{done_now}/{len(pending)} ({pct:5.1f}%)] "
                            f"ok={completed} fail={failed} | {speed:.2f} post/s | "
                            f"elapsed={fmt_duration(elapsed)} ETA={fmt_duration(eta)}"
                        )

    ordered = [done_records[p["key"]] for p in target_posts if p["key"] in done_records]
    summarize(ordered, sample_count=args.samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM × Pydantic 4 スロット＋details 構造化抽出")
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
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
