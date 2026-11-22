# 006 — 曲投稿の類似検索における埋め込みの Disentanglement 計画

- **Status**: IN PROGRESS（H1・H2 検証完了／次フェーズ：4 スロット抽出と本番 `A` 構築）
- **Target**: `apps/api/scripts/`（オフライン学習・診断・実験）、`embedding_transforms`（`z = Ax + b`）
- **前提制約**:
  - 本番推論は現行の `z = Ax + b`（2048 → 256 次元、Turso 単一 `vector_top_k` インデックス、F16、Workers CPU 10ms）を変更しない（ADR 0019 / 0020 / 0021）。
  - 検索経路は「投稿→投稿（@3、同一曲除外）」と「短い言葉→投稿（@20）」の 2 系統を両立させる。

---

## 1. D1 実測で確定した事実（仮説 H1・H2 の結果）

対象：`D1:x-store-staging-01`（非評価学習 2,978 件 / 評価 700 件 / 言葉検索 53 クエリ）。

### H1 診断（`apps/api/scripts/diagnose-embeddings.ts`）

| 指標                             |      ランダム | 2048 次元そのまま | 現行 PCA 256 次元 |
| :------------------------------- | ------------: | ----------------: | ----------------: |
| `aspect@3` / `mood@3`            | 13.2% / 11.3% |     39.2% / 36.8% |     41.0% / 36.6% |
| 同一 Artist（非 artist 観点）@10 |          0.6% |              1.7% |              1.8% |
| 同一年代（非 era/nostalgic）@10  |         48.8% |             63.6% |             63.2% |
| **文字数相関@3**                 |          0.00 |          **0.58** |          **0.64** |

- 5-fold CV 線形予測 $R^2$：**文字数 `0.535`**（最大）、`aspect` 平均 `0.331`、`mood` 平均 `0.219`、リリース年（内容残差後）`-0.088`、頻出アーティスト（内容残差後）`-0.410`。
- **結論**:
  - 現行 PCA は文字数（文量）バイアスをむしろ増幅（`0.58 → 0.64`）していた。
  - リリース年の線形除去は不要、アーティスト重心部分空間の線形除去は内容と重なり大幅悪化（`41.0% → 27.5%`）するため不採用（必要なら SQL 条件で同一アーティストを制限する）。

### H2 実験（`apps/api/scripts/experiment-disentangle.ts`）

1. **Step 1（文字数方向の除去 → PCA 256）**:
   - `Cov-Ortho (len, λ=1.0)` で文字数相関が `0.64 → 0.44` へ低下し、**言葉検索@10 が `60.0% → 63.0%`（+3.0pt）** 改善。
   - `Cov-Ortho`（$\alpha = \infty$）と `Shrinkage-LEACE`（$\alpha = 10..1000$）の差はほぼゼロのため、逆行列不要の `Cov-Ortho` を採用。
2. **Step 2（`[z_aspect(10) | z_mood(11) | z_residual(235)]`、5-fold CV）**:
   - リークゼロの未見 fold 内（140 vs 139）で、Baseline `33.6% / 31.5%` → `Ridge 連結 (α=2000, asp35/mood35/res30)` で **`38.7% (+5.1pt) / 38.8% (+7.3pt)`**。
   - `Step 1 + Ridge 連結 (α=2000)` で全 700 件プールは **`aspect@3 = 49.5% (+8.5pt)`、`mood@3 = 44.3% (+7.7pt)`**、文字数相関 **`0.40`**。
   - 単独ブロック検索（未見）：`z_aspect` 単独 `42.6% / 18.7%`、`z_mood` 単独 `19.9% / 40.4%` と Disentanglement が明確に成立。
   - PLS-SVD は背景分散を割り戻さないため伸びず不採用。Löwdin 強制直交は `mood` が `-1.9pt` 落ちるため、強正則化多出力予測＋$S_{\text{sup}}$ の直交補空間上の残余 PCA を基準とする。

---

## 2. 次フェーズの設計：4 スロット × MECE タグ ＋ 中身スパン抽出

現行の `embedding-eval.json` の 10 aspect / 11 mood は、`trivia`（ゴミ箱）・`dark`（標本不足）・`scene`（空間・時間帯の欠落）や、`memory` の中身（例：部活の帰り道 vs 失恋の夜）が 1 ビットに潰れる課題がある。`draft-assist` の 5 要素（`space/time/sound/memory/emotion`）とも整合させ、以下へ発展させる。

### 4 スロット構造（`Locate` 原文スパン × MECE タグ）

| スロット           | 役割（6 次元）                   | 初期タグ案（200 件試験抽出後に 3〜5% 未満を剪定）                                                                                        |
| :----------------- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **`music_span`**   | 曲の中身（聴覚・言語）           | `lyrics`, `vocal`, `sound`                                                                                                               |
| **`meta_span`**    | 曲の外側（視覚・場）             | `tieup_visual`, `artist_live`                                                                                                            |
| **`context_span`** | 聴き手の空間・時間・記憶（本丸） | **シーン**: `scene_time`, `scene_move`, `scene_daily`<br>**記憶**: `mem_youth`, `mem_love`, `mem_hardship`, `mem_people_turn`, `mem_era` |
| **`mood_span`**    | 感情・余韻（快不快×覚醒度）      | `hype`, `cool`, `calm_warm`, `cute`, `encouraged`, `moved`, `bittersweet`, `nostalgic`, `funny`, `dark_addictive`                        |

### 低コストでの教師データ作成と学習手法の比較

1. **データ抽出（オフライン）**:
   - ※ `perplexity-ai/pplx-decider-v1.3` の `Locate` を検証したが、日本語の音楽感想は 1 文の中に `context` と `mood` が絡み合う構造が多く原文スパン切り出しが安定しなかったため、**汎用 LLM の Structured Outputs（JSON Schema）による `{ tag, details: Array<string>, primary }` 同時抽出**へ方針転換する。
   - コスト目安（全 3,678 件 ≒ 約 100 万トークン）：Gemini 2.5 Flash-Lite / Flash や OpenAI `gpt-4.1-mini` / `nano`（Batch API で数十円〜150 円）、または Workers AI。
   - まず 200 件で試験抽出し、① `details` への曲名・アーティスト名・メンバー名混入チェック、② 出現率 3〜5% 未満のタグの統合・剪定、③ 目視 20 件チェックを行ってから全 3,678 件を JSONL レジューム保存で一括抽出する。
2. **`z = Ax + b` の学習手法比較（リッジ回帰に固定しない）**:
   - **B1（タグ確率）**: L2 正則化ロジスティック回帰 / KL 蒸留（`primary` 重み付きソフトターゲットも比較）。
   - **B2（抽出 `details` 埋め込み $y_i$）**:
     - スロット単位（4 ブロック：`music` / `meta` / `context` / `mood`、または `scene_*` と `mem_*` を分けた 5 ブロック）で `details` を連結して `PLaMo-Embedding-1B` で埋め込む。
     - 正則化 CCA（rCCA）：$X$ と $Y$ の共有角度方向を閉形式で抽出。
     - **Relational Cosine Distillation（本命）**: $\text{KL}(\text{softmax}(\cos(y_i, y_j)/\tau) \,\|\, \text{softmax}(\cos(Ax_i, Ax_j)/\tau))$ で、`details` 同士のコサイン順位分布を直接線形行列 `A` に蒸留。
   - **空スロットの扱い**: 該当エントリが無いスロットはゼロベクトル $\mathbf{0}$ とし、「有無＋中身の近さ」を 1 ブロックで表現。
3. **最終検証と本番反映**:
   - 完全ホールドアウト（非評価 2,978 件学習 → 完全未見 700 件＋53 言葉クエリ）で最終比較。
   - 最良の線形変換を `apps/api/scripts/build-transform.ts` に組み込み、`plamo-1b:disentangle256-v1` として `embedding_transforms` に登録 → `reembed-posts` → `EMBEDDING_MODEL` 切替。

---

## 3. LLM 抽出プロンプト・スキーマ仕様

1 リクエストあたり 10〜15 件をバッチ投入し、Structured Outputs（JSON Schema）で以下を抽出する。

### System Prompt

```markdown
あなたは音楽レビューの「書かれている内容」を構造化抽出する専門のアノテーターです。
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
}
```

### User Prompt テンプレート（10〜15 件バッチ）

````markdown
以下のレビュー配列（全 N 件）について、入力と同じ順番・同じ `key` で JSON を出力してください。

```json
[{ "key": "...", "title": "...", "artistName": "...", "content": "..." }]
```
````

### Structured Outputs JSON Schema（エントリ定義）

ルートは `{ "items": Array<{ "key": string, "entries": { music, meta, context, mood } }> }` とし、各スロット配列の要素は以下のスキーマで縛る：

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["tag", "details", "primary"],
  "properties": {
    "tag": { "type": "string", "enum": ["...スロット固有の enum..."] },
    "details": {
      "type": "array",
      "minItems": 1,
      "maxItems": 4,
      "items": { "type": "string" }
    },
    "primary": { "type": "boolean" }
  }
}
```

---

## 4. 本 PR に含まれる成果物

- [apps/api/scripts/lib/diagnose.ts](../apps/api/scripts/lib/diagnose.ts) / [apps/api/scripts/diagnose-embeddings.ts](../apps/api/scripts/diagnose-embeddings.ts)
- [apps/api/scripts/lib/disentangle.ts](../apps/api/scripts/lib/disentangle.ts) / [apps/api/scripts/experiment-disentangle.ts](../apps/api/scripts/experiment-disentangle.ts)
- [apps/api/tests/embeddings/diagnose.test.ts](../apps/api/tests/embeddings/diagnose.test.ts) / [apps/api/tests/embeddings/disentangle.test.ts](../apps/api/tests/embeddings/disentangle.test.ts)
- [.github/workflows/diagnose-embeddings.yml](../.github/workflows/diagnose-embeddings.yml) / [.github/workflows/experiment-disentangle.yml](../.github/workflows/experiment-disentangle.yml)
- [plans/006-embedding-disentangle-plan.md](./006-embedding-disentangle-plan.md)
