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
   - `perplexity-ai/pplx-decider-v1.3`（または `pplx-decider-flash-v1.3`）の `Choice`/`Noul`（校正済み確率）と `Locate`（原文スパン抽出）を利用し、スポット GPU で全 3,678 件を一括処理（アクセス不可の場合は軽量 LLM の Batch API で代替）。
   - まず 200 件で試験抽出し、出現率 3〜5% 未満のタグを統合して最終スキーマを確定する。
2. **`z = Ax + b` の学習手法比較（リッジ回帰に固定しない）**:
   - **B1（タグ確率）**: L2 正則化ロジスティック回帰 / KL 蒸留（校正済み確率 $p$ をそのまま活用）。
   - **B2（`Locate` スパン埋め込み $y_i$）**:
     - 正則化 CCA（rCCA）：$X$ と $Y$ の共有角度方向を閉形式で抽出。
     - **Relational Cosine Distillation（本命）**: $\text{KL}(\text{softmax}(\cos(y_i, y_j)/\tau) \,\|\, \text{softmax}(\cos(Ax_i, Ax_j)/\tau))$ で、スパン同士のコサイン順位分布を直接線形行列 `A` に蒸留。
   - **空スロットの扱い**: 該当スパンが無い投稿はゼロベクトル $\mathbf{0}$ とし、「有無＋中身の近さ」を 1 ブロックで表現。
3. **最終検証と本番反映**:
   - 完全ホールドアウト（非評価 2,978 件学習 → 完全未見 700 件＋53 言葉クエリ）で最終比較。
   - 最良の線形変換を `apps/api/scripts/build-transform.ts` に組み込み、`plamo-1b:disentangle256-v1` として `embedding_transforms` に登録 → `reembed-posts` → `EMBEDDING_MODEL` 切替。

---

## 3. 本 PR に含まれる成果物

- [apps/api/scripts/lib/diagnose.ts](../apps/api/scripts/lib/diagnose.ts) / [apps/api/scripts/diagnose-embeddings.ts](../apps/api/scripts/diagnose-embeddings.ts)
- [apps/api/scripts/lib/disentangle.ts](../apps/api/scripts/lib/disentangle.ts) / [apps/api/scripts/experiment-disentangle.ts](../apps/api/scripts/experiment-disentangle.ts)
- [apps/api/tests/embeddings/diagnose.test.ts](../apps/api/tests/embeddings/diagnose.test.ts) / [apps/api/tests/embeddings/disentangle.test.ts](../apps/api/tests/embeddings/disentangle.test.ts)
- [.github/workflows/diagnose-embeddings.yml](../.github/workflows/diagnose-embeddings.yml) / [.github/workflows/experiment-disentangle.yml](../.github/workflows/experiment-disentangle.yml)
- [plans/006-embedding-disentangle-plan.md](./006-embedding-disentangle-plan.md)
