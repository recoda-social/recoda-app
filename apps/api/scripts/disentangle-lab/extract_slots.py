#!/usr/bin/env python3
"""
曲投稿（packages/db/data/official-reviews.json）から
4 スロット（music_span / meta_span / context_span / mood_span）の
MECE タグ確率（noul）および原文スパン（locate）を一括抽出するスクリプト。

特徴:
- pplx_decider._backbone をパッチし、最初から device_map={"": "cuda"} で GPU に直接ロード
- バッチごとの詳細ログ（進捗・経過時間・処理速度・ETA・VRAM）をリアルタイム出力
- JSONL へ追記保存するため中断しても自動レジューム可能
- 完了時にタグ出現率（3% / 5% 未満の剪定候補）とスパンカバレッジを自動集計

実行例:
  # 200 件試験抽出（pplx-decider-flash-v1.3）
  python apps/api/scripts/disentangle-lab/extract_slots.py \
    --model /home/kimura/.cache/huggingface/pplx-decider-flash-v1.3 \
    --limit 200 --batch-size 16 \
    --output apps/api/scripts/disentangle-lab/out/pilot200_flash.jsonl

  # 全 3,678 件一括抽出
  python apps/api/scripts/disentangle-lab/extract_slots.py \
    --model /home/kimura/.cache/huggingface/pplx-decider-flash-v1.3 \
    --limit 0 --batch-size 16 \
    --output apps/api/scripts/disentangle-lab/out/all_flash.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

SLOT_TAGS: dict[str, list[str]] = {
    "music_span": ["lyrics", "vocal", "sound"],
    "meta_span": ["tieup_visual", "artist_live"],
    "context_span": [
        "scene_time",
        "scene_move",
        "scene_daily",
        "mem_youth",
        "mem_love",
        "mem_hardship",
        "mem_people_turn",
        "mem_era",
    ],
    "mood_span": [
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

TAG_QUESTIONS: dict[str, dict[str, Any]] = {
    # music_span (3)
    "lyrics": {
        "type": "noul",
        "instructions": "この曲の感想は、歌詞・言葉選び・フレーズ・メッセージの内容に言及していますか？",
    },
    "vocal": {
        "type": "noul",
        "instructions": "この曲の感想は、歌声・歌唱力・声質・コーラス・ハモリ・歌い方に言及していますか？",
    },
    "sound": {
        "type": "noul",
        "instructions": "この曲の感想は、メロディ・リズム・楽器・アレンジ・イントロ・音作り・テンポなどの音楽的サウンドに言及していますか？",
    },
    # meta_span (2)
    "tieup_visual": {
        "type": "noul",
        "instructions": "この曲の感想は、アニメ・ドラマ・映画・CM・ゲームなどのタイアップ作品や、MV・映像・ダンス・演出に言及していますか？",
    },
    "artist_live": {
        "type": "noul",
        "instructions": "この曲の感想は、ライブ・フェス・コンサート体験、またはアーティスト・メンバー本人の魅力や人柄に言及していますか？",
    },
    # context_span - scene (3)
    "scene_time": {
        "type": "noul",
        "instructions": "この曲の感想は、朝・夜・深夜・夕方・季節（春・夏・秋・冬）や天気など、特定の時間帯や季節の情景に言及していますか？",
    },
    "scene_move": {
        "type": "noul",
        "instructions": "この曲の感想は、ドライブ・車中・電車・通勤・通学・帰り道・散歩・旅行など、移動中のシーンに言及していますか？",
    },
    "scene_daily": {
        "type": "noul",
        "instructions": "この曲の感想は、勉強・仕事・家事・作業中・カフェ・部屋・日常の習慣など、普段の生活シーンに言及していますか？",
    },
    # context_span - memory (5)
    "mem_youth": {
        "type": "noul",
        "instructions": "この曲の感想は、学生時代・部活・学校・受験・青春・子どもの頃の思い出に言及していますか？",
    },
    "mem_love": {
        "type": "noul",
        "instructions": "この曲の感想は、恋愛・恋人・片思い・失恋・結婚・好きな人との思い出に言及していますか？",
    },
    "mem_hardship": {
        "type": "noul",
        "instructions": "この曲の感想は、辛い時期・病気・死別・コロナ禍・挫折・悩み・苦難を乗り越えた体験に言及していますか？",
    },
    "mem_people_turn": {
        "type": "noul",
        "instructions": "この曲の感想は、家族・友人・大切な人とのエピソードや、卒業・就職・引っ越し・留学・誕生日など人生の節目に言及していますか？",
    },
    "mem_era": {
        "type": "noul",
        "instructions": "この曲の感想は、昔・当時の流行・何十年も前・時代背景・世代の記憶に言及していますか？",
    },
    # mood_span (10)
    "hype": {
        "type": "noul",
        "instructions": "この曲の感想は、テンションが上がる・盛り上がる・ノリノリ・爽快・高揚感・多幸感といった明るい高覚醒の感情を表していますか？",
    },
    "cool": {
        "type": "noul",
        "instructions": "この曲の感想は、かっこいい・お洒落・洗練された・痺れる・圧倒されるといったクールな感性を表していますか？",
    },
    "calm_warm": {
        "type": "noul",
        "instructions": "この曲の感想は、癒やされる・落ち着く・優しい・温かい・安心する・心地よいといった穏やかな感情を表していますか？",
    },
    "cute": {
        "type": "noul",
        "instructions": "この曲の感想は、かわいい・愛らしい・キュンとする・キラキラした魅力を表していますか？",
    },
    "encouraged": {
        "type": "noul",
        "instructions": "この曲の感想は、励まされる・元気が出る・背中を押される・勇気をもらう・救われる・前向きになれるといった感情を表していますか？",
    },
    "moved": {
        "type": "noul",
        "instructions": "この曲の感想は、感動する・泣ける・胸が熱くなる・心が震える・心に刺さるといった深い感動を表していますか？",
    },
    "bittersweet": {
        "type": "noul",
        "instructions": "この曲の感想は、切ない・儚い・寂しい・胸が締め付けられる・甘酸っぱいといった切なさを表していますか？",
    },
    "nostalgic": {
        "type": "noul",
        "instructions": "この曲の感想は、懐かしい・エモい・当時を思い出す・ノスタルジックな余韻を表していますか？",
    },
    "funny": {
        "type": "noul",
        "instructions": "この曲の感想は、面白い・笑える・ユーモラス・お茶目・ツッコミどころがあるといった楽しさを表していますか？",
    },
    "dark_addictive": {
        "type": "noul",
        "instructions": "この曲の感想は、ダーク・怪しい・病みつきになる・中毒性がある・不穏・狂気的・抜け出せない魅力を表していますか？",
    },
}

LOCATE_QUESTIONS: dict[str, dict[str, Any]] = {
    "music_span": {
        "type": "locate",
        "instructions": "歌詞・歌声・メロディ・サウンド・楽器など、曲そのものの聴覚的・言語的特徴について述べている部分",
    },
    "meta_span": {
        "type": "locate",
        "instructions": "ドラマ・アニメ・映画・MV・映像・ライブ・アーティストやメンバーなど、曲の外側の要素について述べている部分",
    },
    "context_span": {
        "type": "locate",
        "instructions": "聴いていた時間帯・季節・場所・移動シーン・日常、または学生時代・恋愛・家族・苦労・昔などの個人的な記憶や体験を述べている部分",
    },
    "mood_span": {
        "type": "locate",
        "instructions": "聴いたときの感情・気持ち・雰囲気・余韻（泣ける、切ない、癒やされる、盛り上がる、かっこいい、懐かしいなど）を表している部分",
    },
}


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_decider_direct_gpu(
    checkpoint: Path, mode: str, batch_tokens: int, fast: bool = False
):
    sys.path.insert(0, str(checkpoint / "source" / "src"))
    import pplx_decider.model as pm

    def gpu_backbone(ckpt: Path, config: dict[str, Any], dtype: torch.dtype) -> torch.nn.Module:
        model_type = pm.load_json(ckpt / "config.json").get("model_type")
        t0 = time.time()
        log(f"backbone ({model_type}, dtype={dtype}) を device_map='cuda' で直接ロード開始...")
        if model_type == "qwen3_5_moe":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

            model = Qwen3_5MoeModel.from_pretrained(
                str(ckpt),
                dtype=dtype,
                attn_implementation="sdpa",
                experts_implementation=config.get("experts_implementation") or "grouped_mm",
                device_map={"": "cuda"},
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

            model = Qwen3_5Model.from_pretrained(
                str(ckpt),
                dtype=dtype,
                attn_implementation="sdpa",
                device_map={"": "cuda"},
            )
        log(
            f"backbone ロード完了 ({time.time() - t0:.1f}s) | "
            f"VRAM allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB"
        )
        return model

    pm._backbone = gpu_backbone
    t0 = time.time()
    fast_opt: bool | Any = False
    if fast:
        fast_opt = pm.FastOptions(
            compile="layer",
            cuda_graphs=False,
            token_budget=batch_tokens,
        )
    decider = pm.Decider.from_pretrained(
        checkpoint,
        device="cuda",
        mode=mode,
        fast=fast_opt,
        batch_tokens=batch_tokens,
    )
    log(
        f"Decider 準備完了 ({time.time() - t0:.1f}s) | mode={mode} fast={fast} | "
        f"VRAM allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB / "
        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB"
    )
    return decider


def summarize(records: list[dict[str, Any]]) -> None:
    n = len(records)
    if n == 0:
        log("レコードが 0 件のため集計をスキップします。")
        return

    print("\n" + "=" * 88)
    print(f"抽出結果サマリ（N = {n} 件）")
    print("=" * 88)
    print(f"{'スロット':<14} {'タグ':<18} {'p>=0.5':>9} {'p>=0.3':>9} {'平均p':>8}  判定")
    print("-" * 88)
    prune_3: list[str] = []
    prune_5: list[str] = []
    for slot, tags in SLOT_TAGS.items():
        for tag in tags:
            probs = [r["tags"][tag] for r in records]
            r50 = sum(1 for p in probs if p >= 0.5) / n
            r30 = sum(1 for p in probs if p >= 0.3) / n
            mean_p = sum(probs) / n
            flag = ""
            if r50 < 0.03:
                flag = "< 3% (剪定候補)"
                prune_3.append(tag)
            elif r50 < 0.05:
                flag = "< 5% (統合検討)"
                prune_5.append(tag)
            print(
                f"{slot:<14} {tag:<18} {r50 * 100:8.1f}% {r30 * 100:8.1f}% {mean_p:8.3f}  {flag}"
            )
    print("-" * 88)
    print(f"剪定候補 (<3%): {prune_3 or 'なし'}")
    print(f"統合検討 (3-5%): {prune_5 or 'なし'}")

    print("\n--- Locate スパン カバレッジ ---")
    for slot in SLOT_TAGS:
        non_empty = [r["spans"][slot] for r in records if r["spans"].get(slot)]
        cov = len(non_empty) / n * 100
        avg_len = (
            sum(sum(len(s["text"]) for s in spans) for spans in non_empty) / len(non_empty)
            if non_empty
            else 0.0
        )
        print(f"  {slot:<14}: カバレッジ {cov:5.1f}% ({len(non_empty)}/{n}) | 平均抽出文字数 {avg_len:4.1f}ch")

    print("\n--- サンプル 5 件 ---")
    for r in records[:5]:
        print("-" * 88)
        print(f"KEY: {r['key']} ({len(r['content'])}ch) | {r['content']}")
        top_tags = sorted(
            [(k, v) for k, v in r["tags"].items() if v >= 0.3],
            key=lambda x: -x[1],
        )
        print("  TAGS (>=0.3): " + ", ".join(f"{k}:{v:.2f}" for k, v in top_tags))
        for slot in SLOT_TAGS:
            spans = r["spans"].get(slot, [])
            if spans:
                joined = " | ".join(f'"{s["text"]}"({s["score"]:.2f})' for s in spans)
                print(f"  {slot:<14}: {joined}")
    print("=" * 88 + "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="pplx-decider による 4 スロット・MECE タグ抽出")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/kimura/.cache/huggingface/pplx-decider-flash-v1.3"),
        help="pplx-decider チェックポイントのディレクトリパス",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("packages/db/data/official-reviews.json"),
        help="入力の official-reviews.json パス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="出力 JSONL パス（既存ファイルがあれば自動レジューム）",
    )
    parser.add_argument("--limit", type=int, default=200, help="処理件数上限（0 で全件）")
    parser.add_argument("--seed", type=int, default=42, help="limit > 0 のときのサンプリングシード")
    parser.add_argument("--batch-size", type=int, default=16, help="1 バッチの投稿数")
    parser.add_argument("--batch-tokens", type=int, default=32768, help="Decider の batch_tokens")
    parser.add_argument(
        "--mode",
        choices=["isolated", "separate"],
        default="isolated",
        help="Decider の推論モード（既定: isolated）",
    )
    parser.add_argument("--skip-locate", action="store_true", help="Locate スパン抽出をスキップ")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="pplx_decider の packed fast path (torch.compile) を有効化（Locate は自動無効）",
    )
    args = parser.parse_args()
    if args.fast:
        args.skip_locate = True

    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"モデル: {args.model}")
    log(f"入力: {args.input} -> 出力: {args.output}")

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
    log(f"残り処理対象: {len(pending)} 件 / 合計 {len(target_posts)} 件")

    if pending:
        decider = load_decider_direct_gpu(args.model, args.mode, args.batch_tokens, fast=args.fast)
        total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        t_start = time.time()
        processed = 0

        with args.output.open("a", encoding="utf-8") as out_f:
            for bi in range(total_batches):
                batch = pending[bi * args.batch_size : (bi + 1) * args.batch_size]
                t_b0 = time.time()

                tag_reqs = [{"state": p["content"], "questions": TAG_QUESTIONS} for p in batch]
                tag_resps = decider.decide(tag_reqs)
                t_tags = time.time() - t_b0

                t_l0 = time.time()
                if args.skip_locate:
                    loc_resps = None
                else:
                    loc_reqs = [
                        {"state": p["content"], "questions": LOCATE_QUESTIONS} for p in batch
                    ]
                    loc_resps = decider.decide(loc_reqs)
                t_loc = time.time() - t_l0

                for i, p in enumerate(batch):
                    tags = {
                        k: round(float(tag_resps[i]["answers"][k]["noul"]), 5)
                        for k in TAG_QUESTIONS
                    }
                    spans: dict[str, list[dict[str, Any]]] = {}
                    if loc_resps is not None:
                        for sk in LOCATE_QUESTIONS:
                            spans[sk] = [
                                {
                                    "start": int(s["start"]),
                                    "end": int(s["end"]),
                                    "text": s["text"],
                                    "score": round(float(s["score"]), 4),
                                }
                                for s in loc_resps[i]["answers"][sk]["spans"]
                            ]
                    rec = {
                        "key": p["key"],
                        "isrc": p["isrc"],
                        "content": p["content"],
                        "tags": tags,
                        "spans": spans,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done_records[p["key"]] = rec
                out_f.flush()

                processed += len(batch)
                elapsed = time.time() - t_start
                speed = processed / elapsed if elapsed > 0 else 0.0
                eta = (len(pending) - processed) / speed if speed > 0 else 0.0
                pct = processed / len(pending) * 100
                vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                log(
                    f"[batch {bi + 1}/{total_batches}] {processed}/{len(pending)} ({pct:5.1f}%) | "
                    f"tags={t_tags:.2f}s locate={t_loc:.2f}s | "
                    f"{speed:.2f} post/s | elapsed={fmt_duration(elapsed)} ETA={fmt_duration(eta)} | "
                    f"VRAM={vram:.2f}GB"
                )

    ordered = [done_records[p["key"]] for p in target_posts if p["key"] in done_records]
    summarize(ordered)


if __name__ == "__main__":
    main()
