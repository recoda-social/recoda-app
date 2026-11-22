#!/usr/bin/env python3
"""
書き換え後の公式レビュー（3,675件）に合わせて、以下を差分更新するスクリプト:
  1. out/x_cache.npz:
     - 未変更 1,992 件の x と 53 クエリの query_xs はそのまま再利用
     - 書き換えられた 1,683 件の x だけを D1 (x-store-staging-01) から再取得
     - v1 との差分ノルムを確認し、D1 側が本当に更新されているか検証
  2. out/tag_details_cache.npz:
     - 既存の details テキスト埋め込みを再利用
     - 新しく抽出された all_qwen36.jsonl に登場する未キャッシュの details テキストだけを
       Cloudflare Workers AI (@cf/pfnet/plamo-embedding-1b) で埋め込み追加
"""

from __future__ import annotations

import json
import shutil
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np

from train_eval_007 import (
    cf_api,
    embed_texts_cf,
    extract_tag_entries,
    log,
    official_post_id,
)


def main() -> None:
    out_dir = Path("apps/api/scripts/disentangle-lab/out")
    store_name = "x-store-staging-01"

    official = json.loads(Path("packages/db/data/official-reviews.json").read_text(encoding="utf-8"))
    post_content = {p["key"]: p["content"] for p in official["posts"]}
    all_keys = [p["key"] for p in official["posts"]]

    v1_content: dict[str, str] = {}
    v1_jsonl = out_dir / "all_qwen36_v1.jsonl"
    if v1_jsonl.exists():
        with v1_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    v1_content[r["key"]] = r["content"]

    changed_keys = [k for k in all_keys if v1_content.get(k) != post_content[k]]
    unchanged_keys = [k for k in all_keys if v1_content.get(k) == post_content[k]]
    log(f"対象 {len(all_keys)} 件: 未変更={len(unchanged_keys)}, 書き換え={len(changed_keys)}")

    # --- 1. x_cache.npz の差分更新 ---
    x_cache_path = out_dir / "x_cache.npz"
    x_bak_path = out_dir / "x_cache_v1.npz"
    if x_cache_path.exists() and not x_bak_path.exists():
        shutil.copy2(x_cache_path, x_bak_path)
        log(f"バックアップ作成: {x_bak_path.name}")

    old_x_npz = np.load(x_bak_path if x_bak_path.exists() else x_cache_path, allow_pickle=False)
    old_x_keys = old_x_npz["keys"].tolist()
    old_x_arr = old_x_npz["xs"].astype(np.float32)
    old_x_map = {k: old_x_arr[i] for i, k in enumerate(old_x_keys)}
    query_xs = old_x_npz["query_xs"].astype(np.float32)

    need_fetch_keys = [k for k in all_keys if k in set(changed_keys) or k not in old_x_map]
    log(f"D1 ({store_name}) から {len(need_fetch_keys)} 件の新しい x を取得中...")

    dbs = cf_api(f"/d1/database?name={urllib.parse.quote(store_name)}")
    db = next((d for d in dbs if d["name"] == store_name), None)
    if db is None:
        raise RuntimeError(f"D1 database {store_name} not found")
    db_uuid = db["uuid"]

    fetched: dict[str, np.ndarray] = {}
    post_ids = [official_post_id(k) for k in need_fetch_keys]
    id_to_key = dict(zip(post_ids, need_fetch_keys, strict=True))
    for i in range(0, len(post_ids), 100):
        part = post_ids[i : i + 100]
        placeholders = ", ".join(["?"] * len(part))
        res = cf_api(
            f"/d1/database/{db_uuid}/query",
            method="POST",
            body={
                "sql": f"SELECT post_id, hex(embedding) AS embedding FROM x_vectors WHERE post_id IN ({placeholders})",
                "params": part,
            },
        )
        for row in res[0].get("results", []):
            raw_bytes = bytes.fromhex(str(row["embedding"]))
            vec = np.frombuffer(raw_bytes, dtype="<f2").astype(np.float32)
            fetched[id_to_key[str(row["post_id"])]] = vec
        if (i // 100) % 5 == 0 or i + 100 >= len(post_ids):
            log(f"  D1 fetch: {min(i + 100, len(post_ids))}/{len(post_ids)} (found={len(fetched)})")

    # D1 側が本当に書き換わっているか sanity check
    actually_updated = 0
    identical_to_v1 = 0
    for k, vec in fetched.items():
        if k in old_x_map:
            if float(np.max(np.abs(vec - old_x_map[k]))) > 1e-4:
                actually_updated += 1
            else:
                identical_to_v1 += 1
    log(f"D1 差分チェック: 値が変化={actually_updated} 件, v1 と同一={identical_to_v1} 件")

    new_x_map: dict[str, np.ndarray] = {}
    missing_keys: list[str] = []
    for k in all_keys:
        if k in fetched:
            new_x_map[k] = fetched[k]
        elif k in old_x_map:
            new_x_map[k] = old_x_map[k]
        else:
            missing_keys.append(k)
    if missing_keys:
        log(f"[WARN] D1 に見つからなかったキーが {len(missing_keys)} 件あります (先頭5件: {missing_keys[:5]})")

    final_keys = [k for k in all_keys if k in new_x_map]
    final_xs = np.stack([new_x_map[k] for k in final_keys], axis=0).astype(np.float32)
    np.savez_compressed(
        x_cache_path,
        keys=np.array(final_keys),
        xs=final_xs,
        query_xs=query_xs,
    )
    log(f"x_cache.npz 更新完了: {len(final_keys)} 件")

    # --- 2. tag_details_cache.npz の差分更新 ---
    d_cache_path = out_dir / "tag_details_cache.npz"
    d_bak_path = out_dir / "tag_details_cache_v1.npz"
    if d_cache_path.exists() and not d_bak_path.exists():
        shutil.copy2(d_cache_path, d_bak_path)
        log(f"バックアップ作成: {d_bak_path.name}")

    old_d_npz = np.load(d_bak_path if d_bak_path.exists() else d_cache_path, allow_pickle=False)
    old_d_texts = old_d_npz["texts"].tolist()
    old_d_vecs = old_d_npz["vecs"].astype(np.float16)
    text_vecs: dict[str, np.ndarray] = {t: old_d_vecs[i] for i, t in enumerate(old_d_texts)}

    records_by_key: dict[str, dict[str, Any]] = {}
    with (out_dir / "all_qwen36.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records_by_key[rec["key"]] = rec

    needed_texts: set[str] = set()
    for k in final_keys:
        rec = records_by_key.get(k)
        if not rec:
            continue
        for text, _ in extract_tag_entries(rec).values():
            if text not in text_vecs:
                needed_texts.add(text)

    missing_texts = sorted(needed_texts)
    log(f"tag_details_cache.npz: 既存={len(text_vecs)}, 新規埋め込み対象={len(missing_texts)} テキスト")
    if missing_texts:
        new_vecs = embed_texts_cf(missing_texts, "new_details")
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        new_vecs_u = (new_vecs / norms).astype(np.float16)
        for t, v in zip(missing_texts, new_vecs_u, strict=True):
            text_vecs[t] = v

    all_texts = list(text_vecs.keys())
    all_vecs = np.stack([text_vecs[t] for t in all_texts], axis=0).astype(np.float16)
    np.savez_compressed(
        d_cache_path,
        texts=np.array(all_texts),
        vecs=all_vecs,
    )
    log(f"tag_details_cache.npz 更新完了: 合計 {len(all_texts)} テキスト")


if __name__ == "__main__":
    main()
