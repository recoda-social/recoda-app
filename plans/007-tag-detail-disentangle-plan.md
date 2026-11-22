# 007 — タグの根拠・具体（`details`）まで一致させる Disentanglement 実装計画

- **Status**: DONE（007 検証および `StrictResonanceJudge` 評価完了／最終採用：`rcca_reg8`＝Step 1 文字数除去＋ノイズ除去済み `details` 強収縮 rCCA）
- **Depends on**: [006-embedding-disentangle-plan.md](./006-embedding-disentangle-plan.md)（4 スロット × `{ tag, details, primary }` の LLM 抽出データ）
- **Target**: オフライン学習スクリプト（`apps/api/scripts/`）、`embedding_transforms`（256 次元 `z`）
- **目的**: タグの有無（粗い一致）にとどまらず、「同じタグの根拠・具体（`details`）同士」が一致し、かつ他スロットと混ざらない 256 次元表現を構築する。

---

## 1. 課題と設計方針

スロット内の `details` を単純連結して 1 本に埋め込むと、`scene_move`（夜のドライブ）と `mem_love`（別れの記憶）のように異質な中身が混ざる。一方、23 タグすべてに個別ブロックを割くと 256 次元に収まらない。

そこで **① タグ単位で突き合わせる教師類似度行列** と **② 6 ブロックへの分割＋独立性ペナルティ** の 2 つを組み合わせる。

---

## 2. 教師信号：Tag-Matched Detail Similarity

ブロック $b \in \{\text{scene}, \text{memory}, \text{music}, \text{meta}, \text{mood}\}$ ごとに、学習データ上の投稿ペア $(i, j)$ の教師類似度行列 $S_{ij}^{(b)}$ を構築する：

$$S_{ij}^{(b)} = \sum_{t \in b} \mathbb{I}[t \in i \land t \in j] \cdot w_{i,t} w_{j,t} \cdot \max\!\Big(0,\; \cos\big(e_{i,t},\; e_{j,t}\big)\Big)$$

- $e_{i,t} = \text{PLaMo-Embedding-1B}(\text{join}(\text{details}_{i,t}, \text{"。"}))$（タグ $t$ の `details` 配列を連結して埋め込んだ 2048 次元ベクトル）
- $w_{i,t} = 1.0$（`primary: true`）/ $0.5$（`primary: false`）
- **性質**:
  - タグが異なるペア（例：`mem_love` vs `mem_youth`）は寄与が **0**。
  - 同じタグでも具体が異なるペア（例：初恋 vs 失恋）は $\cos(e_{i,t}, e_{j,t})$ が低く、**同じタグ かつ 具体・根拠まで近いペア**だけが上位に並ぶ。

---

## 3. 256 次元の 6 ブロック構成

Step 1（`Cov-Ortho` による文字数方向の除去）を経た $\tilde{x} \in \mathbb{R}^{2048}$ を、以下の 6 ブロック（計 256 次元）に写像する：

| ブロック     | 次元（初期案） | 担当タグ群                                                                |
| :----------- | -------------: | :------------------------------------------------------------------------ |
| `z_scene`    |             32 | `scene_time`, `scene_move`, `scene_daily`                                 |
| `z_memory`   |             48 | `mem_youth`, `mem_love`, `mem_hardship`, `mem_people_turn`, `mem_era`     |
| `z_music`    |             48 | `lyrics`, `vocal`, `sound`                                                |
| `z_meta`     |             16 | `tieup_visual`, `artist_live`                                             |
| `z_mood`     |             32 | `mood` 10 タグ                                                            |
| `z_residual` |             80 | 上記 5 ブロックの直交補空間上の PCA（言葉検索・タグ外細部のガードレール） |

各ブロックは学習データ上の RMS ノルム $s_b = \sqrt{\mathbb{E}[\|z_b\|^2]}$ で正規化し、寄与比率 $w_b$（$\sum w_b^2 = 1$）を掛けて 1 本の 256 次元ベクトルに連結する。

---

## 4. 学習目的関数（順位蒸留 ＋ ブロック間 Disentangle）

各ブロックの線形射影 $A_b \in \mathbb{R}^{d_b \times 2048}$ を以下の合成損失で学習する（`z_residual` は 5 ブロック確定後に直交補空間の PCA で決定）：

$$\mathcal{L} = \mathcal{L}_{\text{rank}} + \beta \, \mathcal{L}_{\text{disentangle}} + \alpha \sum_{b} \|A_b\|_F^2$$

1. **順位蒸留損失（Relational Cosine Distillation）**:
   $$\mathcal{L}_{\text{rank}} = \sum_{b} \sum_{i \in \mathcal{I}_b} \text{KL}\!\Big(\text{softmax}_j\big(S_{ij}^{(b)} / \tau\big) \;\Big\|\; \text{softmax}_j\big(\cos(A_b \tilde{x}_i, A_b \tilde{x}_j) / \tau\big)\Big)$$
   ※ $\mathcal{I}_b$ はブロック $b$ に 1 つ以上タグを持つ投稿集合。
2. **ブロック間独立ペナルティ（Disentangle）**:
   $$\mathcal{L}_{\text{disentangle}} = \sum_{a \neq b} \big\|\text{Cov}(A_a \tilde{X},\; A_b \tilde{X})\big\|_F^2$$
   異なるブロック間の線形相関を罰し、スロット間の漏れを抑える。
3. **空スロット対策（Phase 2 拡張：軽量ゲート付き射影）**:
   - 純粋な線形変換 $z_b = A_b \tilde{x}$ では「ブロック $b$ に言及がない投稿を厳密に $\mathbf{0}$ に落とす」ことに限界がある。
   - Phase 1（純線形 `z = Ax + b`）で空スロット由来の誤ヒットがボトルネックと判明した場合のみ、推論側に軽量ゲートを追加する：
     $$z_b = \sigma(g_b^\top \tilde{x} + c_b) \cdot (A_b \tilde{x})$$
   - ゲートは「ブロック $b$ にタグが 1 つ以上あるか」の二値分類（BCE 損失）で学習。**Turso に保存する 256 次元 F16 ベクトルと `vector_top_k` インデックスは一切変更せず**、Workers の計算増も 5 回の内積（0.1ms 未満）にとどまる。

---

## 5. 検証ステップと採用基準

1. **Phase 0（データ準備）**: [006-embedding-disentangle-plan.md](./006-embedding-disentangle-plan.md) のプロンプトで 200 件試験抽出 → タグ剪定 → 全 3,678 件抽出 → 各 `{ tag, details }` を `PLaMo-Embedding-1B` で埋め込み。
2. **Phase 1（純線形 `z = Ax + b` の完全ホールドアウト評価）**:
   - 非評価 2,978 件で学習、完全未見 700 件＋53 言葉クエリで以下を比較：
     - Baseline（現行 PCA 256）
     - タグのみ蒸留（B1：KL 蒸留）
     - 本提案（B2：Tag-Matched Detail Similarity ＋ Relational Cosine Distillation ＋ 独立ペナルティ）
   - **評価指標**:
     - タグ一致 `@3` / `@10`
     - **具体一致度（新規）**: 検索上位 3 件の平均 $S_{ij}^{(b)}$（Tag-Matched Detail Similarity）および nDCG@3
     - **Disentanglement 度**: 各ブロック単独検索での他ブロック相関・独立性
     - **ガードレール**: 言葉検索 53 クエリ `@10` が Baseline（60.0%）を下回らないこと
3. **Phase 2（必要に応じて軽量ゲート検証）**:
   - 「言及がないスロットを持つ投稿」での誤ヒット率を確認し、ゲート $\sigma(g_b^\top \tilde{x} + c_b)$ の有無を比較。
4. **本番反映**:
   - 現行 `z = Ax + b` で十分なら `build-transform.ts` に組み込んで `plamo-1b:disentangle256-v1` へ切替。
   - ゲートが明確に勝る場合は、`embedding_transforms` にゲートパラメータ列（5 × 2049 float32 ≒ 40KB）を追加し `transform.ts` の `project()` のみ更新。

---

## 6. 実測結果と最終採用モデルの決定

対象：ローカル vLLM（`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` + ReplaySSM）で全 3,678 件を Pydantic 構造化抽出（失敗 0・所要 10 分 16 秒、アーティスト名混入 1.2%・曲名混入 2.4%）し、Cloudflare D1 の 2048 次元 `x` と Workers AI（`@cf/pfnet/plamo-embedding-1b`）の `details` 埋め込みを用いて完全ホールドアウト（学習 2,978 件 → 完全未見 700 件＋53 言葉クエリ）で比較した。

### 重要な発見（Goodhart の法則の回避）

- 23 タグへの過教師（`p1_linear`: 教師 60%）や Phase 2 軽量ゲート（`p2_soft`）は、旧タグ指標（`asp@3 / mod@3`）を大きく伸ばした一方、**ブラインド LLM-as-a-Judge（正逆両方向・位置バイアス除去）では Baseline に明確に負けた**（`p1_linear` 42.2%、`p2_soft` 33.0%）。粗い 23 タグにエネルギーを集中させると、`PLaMo-Embedding-1B` が元々持つ微細な感情・文脈ニュアンスが潰れるため。
- 逆に、23 タグのボトルネックを通さず、**曲名・アーティスト名・挨拶ノイズ・文量を取り除いた連続 `details` 埋め込み $y_i^{\text{clean}}$ と $\tilde{x}\_i$ の強収縮正則化 CCA（`rcca_reg8`: `Cov-Ortho(len, λ=1.0)` ＋ `rCCA(reg_x=8.0, reg_y=4.0, power=1.5)`）** が、縦・横 MECE 4 軸（`situation` / `memory` / `musical_focus` / `emotion_shift`）に基づく **`StrictResonanceJudge` 絶対評価で全軸 1 位**となった。

### `StrictResonanceJudge`（完全未見 150 クエリ × Top-3 ＝ 各 450 ペア）＋定量指標

| 手法（純線形 `z = Ax + b`）       | 平均点 (/3) | 妥当率 (≥2) |   **強共鳴 (3点)** | **ハズレ (0点)** |      **#1 妥当率** |       **`言葉@10`** |  `len@3` |
| :-------------------------------- | ----------: | ----------: | -----------------: | ---------------: | -----------------: | ------------------: | -------: |
| `baseline`（現行 PCA 256）        |       2.418 |       76.4% |        66.4% (299) |         1.1% (5) |              76.0% |               60.4% |     0.64 |
| **`rcca_reg8`（最終採用）**       |   **2.487** |   **78.9%** | **70.4%** (+4.0pt) |   **0.7%** (3件) | **82.7%** (+6.7pt) | **73.8%** (+13.4pt) | **0.34** |
| `metric_mlp`（8,000 ペア学習）    |       2.402 |       76.9% |        65.1% (293) |         1.8% (8) |              79.3% |               66.8% |     0.30 |
| `metric_linear`（8,000 ペア学習） |       2.240 |       68.4% |        58.4% (263) |        2.9% (13) |              72.7% |               58.3% |     0.32 |

### 最終採用と本番反映手順

- **採用モデル**: `plamo-1b:disentangle256-v1`（全 3,678 件で学習した `rcca_reg8` を単一アフィン変換 `z = Ax + b` へ合成、推論コード・Turso スキーマ変更ゼロ）
  - ※本モデルが行う分離（Disentanglement）は、**「文量・表層ノイズ（曲名・アーティスト名・挨拶等）と純粋な体験・感性の中身の分離（`Cov-Ortho` ＋ ノイズ除去済み `details` との強収縮 `rCCA`）」および「出力 256 軸同士の無相関化」** であり、当初案の「5 スロットを明示的な別ブロックへ分割する方式」は体験×感情の微細な掛け算を損なうため不採用とした。
- **生成＆登録コマンド**:
  1. バイナリ生成：`apps/api/scripts/disentangle-lab/export_production_transform.py` → `out/plamo-1b-disentangle256-v1.bin`（2,098,176 bytes）
  2. Turso 登録：`vpx apps/api/scripts/register-transform.ts apps/api/scripts/disentangle-lab/out/plamo-1b-disentangle256-v1.bin plamo-1b:disentangle256-v1`
  3. 再射影＆切替：`EMBEDDING_MODEL=plamo-1b:disentangle256-v1 vpx apps/api/scripts/embed-posts.ts` で新モデルの 256 次元 `z` を生成 → `packages/db/src/constants.ts` の `EMBEDDING_MODEL` を `plamo-1b:disentangle256-v1` へ切替。
