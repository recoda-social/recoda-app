# iOS Safari のキーボード表示中は visualViewport でシェルを書き換えない

チャット画面（`ChatWorkspace`）では、`viewport` メタタグに `interactive-widget=resizes-content` を指定したうえで、`html` / `body` / チャットシェルに `position: fixed` を付けず、**キーボード表示中に `window.visualViewport` でシェルの `height` や `top`、あるいは入力欄の `transform` を書き換えない**。代わりにキーボード開閉の挙動はブラウザネイティブに任せ、キーボードが閉じた後に `window.scrollY > 0` が残った場合だけ `window.scrollTo(0, 0)` でリセットする。iOS Safari 上でページ全体が一度スクロールしてから JS 側でサイズ調整される 2 段階のガクつきを解消するためである。

## 前提

- チャット画面は上部バー・カレンダーストリップ・スクロール可能なスレッド領域（`<main>`）・下部の入力欄（`ComposerDock`）で構成される全画面シェル（`ChatWorkspace.tsx`）である。
- 従来の実装では、`html` / `body` / `[data-chat-shell]` をすべて `position: fixed` で固定し、`window.visualViewport` の `resize` / `scroll` イベントごとに `shellEl.style.height = "${vv.height}px"`、`shellEl.style.top = "${vv.offsetTop}px"`、`window.scrollTo(0, 0)` を実行していた。
- 実機 iPhone（iOS Safari）での検証において、以下の挙動を確認した：
  - 画面下部の入力欄にフォーカスすると、iOS Safari は `html` / `body` が `position: fixed; overflow: hidden` であっても、ネイティブ側で visualViewport 全体を上へパンするアニメーションを実行する。
  - `visualViewport` の `resize` / `scroll` イベントはキーボード出現アニメーションの最中に毎フレーム発火せず遅延して届くため、「ネイティブのスクロール → JS による `height` / `top` の強制書き換え」という 2 段階のガクつきが発生する。
- 同種のモバイルチャット UI である Gemini Web（`https://gemini.google.com/app`）の実装を調査したところ、`interactive-widget=resizes-content` と `position: fixed` を使わない `height: 100%` Flex 構成を採用し、`visualViewport` ではキーボードが閉じた後の `window.scrollTo(0, 0)` による後片付けと開閉状態の判定のみを行っていた。

## 理由

### 1. ブラウザネイティブのアニメーションと JS が競合しない

キーボードが開く瞬間に JS が `height`・`top`・`transform`・`scrollTo(0, 0)` を介入させないことで、iOS Safari のネイティブなパン挙動だけで 1 回で完結する。結果として、従来の「一度ページ全体が上にスクロールしてからガクッとサイズ調整される」違和感がなくなる。

### 2. キーボード終了後のスクロールずれだけを確実に防げる

iOS Safari ではキーボードを閉じた後に `window.scrollY > 0` が残ることがある。`visualViewport` の `resize` と `window` の `scroll` で、キーボードが閉じた状態（`!(vv.height < html.clientHeight - 150)`）かつ `window.scrollY > 0 || window.scrollX > 0` のときだけ `window.scrollTo(0, 0)` を呼ぶことで、開くときの動きを邪魔せずに閉じた後のレイアウト崩れを防げる。

### 3. ポップオーバー高さ計算と Android Chrome 対応がそのまま噛み合う

- 既存の `useComposerPopoverHeight` は `visualViewport.offsetTop` を考慮して上部境界を計算しているため、シェルを固定し直さなくても曲検索ポップオーバーが画面上端からはみ出さない。
- `viewport` メタタグに `interactive-widget=resizes-content` を追加しているため、Android Chrome や将来この仕様をサポートする iOS Safari では、レイアウトビューポート自体がキーボードに合わせてリサイズされる。

## 検討した他の案

- **従来の実装（`html` / `body` / `shell` を `position: fixed` にし、`visualViewport` で `shell.style.height` と `shell.style.top` を即時同期）**：iOS Safari ネイティブのパン後に JS のサイズ調整が走るため、2 段階のガクつきが顕著であり不採用。
- **初回タップ時のオフスクリーン・フォーカストリック ＋ `ComposerDock` の `transform` トランジション**：未フォーカス時のタップを `touchend` で横取りし、一時的に `translateY(-2000px)` に逃がして `focus({ preventScroll: true })` することで自動パンを抑止し、入力欄だけを CSS トランジションで持ち上げる案を試作した。実機 iPhone で操作感が不自然になり、IME やフォーカス挙動との干渉リスクも高いため不採用。
- **フォーカス横取りなしで `ComposerDock` だけ `transform` + CSS トランジション（ダブル rAF による coalescing 含む）**：iOS Safari 側のパン（`offsetTop` の変化）と `ComposerDock` の `transform` トランジションが噛み合わず、フォーカス時に入力欄が必要以上に上へ飛び出してからキーボード上へ戻るオーバーシュートが発生したため不採用。

## この決定による影響

- `apps/website/index.html` の `viewport` メタタグに `interactive-widget=resizes-content` を追加する。
- `apps/website/src/chat/ChatWorkspace.tsx` では `html` / `body` / `[data-chat-shell]` から `position: fixed` を外し、`html` / `body` / `#app` を `height: 100%`、`shell` を `position: relative; height: 100dvh` として管理する（アンマウント時に元のインラインスタイルへ戻す）。
- `ChatWorkspace.tsx` の `visualViewport` リスナーは、`keyboardOpen` 状態の更新とキーボード終了後の `window.scrollTo(0, 0)` のみを担う。
- 将来 iOS Safari が `interactive-widget=resizes-content` や `VirtualKeyboard` API をネイティブに完全サポートした場合も、そのまま恩恵を受けられる構成になっている。
