# iOS Safari の入力 blur 時チラつきを防ぐため、チャット画面の非入力クロームを合成レイヤーへ集約する

チャット画面（`ChatWorkspace`）では、`html` / `body` / `#app` の初期レイアウト設定を `useLayoutEffect` で初回ペイント前に揃えたうえで、`styles.shell` を `height: "100%"` / `overflow: "clip"`、`styles.workspace` を `overflow: "hidden"` なし＋`minHeight: 0` にする。あわせて、入力要素を含まない上部クローム（`AppShell.Topbar`・`MiniPlayerBar`・`ListeningCalendarStrip`）を単一の GPU 合成レイヤー（`styles.topChrome`：`transform: "translateZ(0)"`・`backfaceVisibility: "hidden"`・`willChange: "transform"`）へ集約して `styles.topbar` からチャット画面では不要な半透明・`backdrop-filter`・`position: relative; z-index: 10` を外し、下部のモードスイッチ行（`ComposerDock.Header`）も入力欄（`ComposerDock.Surface`）から切り離して同様に GPU 合成レイヤーへ固定し、`ChatComposer.Input` の自動高さ調整 `useLayoutEffect` から未使用の `keyboardOpen` 依存を取り除く。iOS Safari で入力欄を `blur` した際にヘッダーやカレンダーのボタン類・モードスイッチが一瞬消える描画抜け（#114 / PR #118）を、[ADR-0023](./0023-ios-keyboard-layout-without-visual-viewport-resize.md)（キーボード出現時のネイティブ挙動維持）および [ADR-0024](./0024-mobile-dialog-scroll-and-input-zoom-on-ios.md)（入力要素の祖先に `transform` を付けないカーソルズレ回避）と両立して解消するためである。

## 前提

- チャット画面は [ADR-0023](./0023-ios-keyboard-layout-without-visual-viewport-resize.md) の方針に従い、`html` / `body` / シェルに `position: fixed` や `visualViewport` によるリサイズを行わず、iOS Safari ネイティブのキーボードパン挙動に任せている。
- `@ui-sdk/core/dist/web.css` は `@media (pointer: coarse)` 環境で `button[data-slot]` および `[data-slot="segmented-control-item"]` に `position: relative` と `::after` 拡張ヒット領域を付与するため、タッチ端末ではカレンダーの月送りボタン・「月全体」・7 日間セル・モードスイッチがそれぞれ独立した `RenderLayer` を形成する。
- 実機 iPhone（iOS Safari）での検証において、以下の挙動を確認した：
  - 入力欄にフォーカスした後 `blur` すると、キーボードが閉じるタイミングでヘッダーの中身・カレンダーの月送りボタン／「月全体」／7 日間セル・モードスイッチが一瞬描画されなくなり、静的テキスト（「2026年11月」見出し）と入力欄本体だけが残る（#114）。
  - `Topbar` から `backdrop-filter: blur(10px)` を外すだけ、あるいは `focusin` / `focusout` での即時 `sync` 停止と `window.scrollTo(0, 0)` の 180ms デバウンス＋`requestAnimationFrame` 遅延だけでは解消しない。
  - `ListeningCalendarStrip`・`ComposerDock.Header`・`Topbar` をそれぞれ個別に `transform: translateZ(0)` で合成レイヤー化するとカレンダーとモードスイッチは安定したが、最上端（`y = 0` 付近・高さ約 46px）の `Topbar` だけが単独の小さなオフスクリーンレイヤーとして破棄・遅延され、一瞬消える挙動が残った。

## 理由

### 1. 上部クロームを単一の合成レイヤーへまとめることで最上端の破棄を防げる

`AppShell.Topbar`・`MiniPlayerBar`・`ListeningCalendarStrip` を `styles.topChrome` で囲み、`styles.topChrome` だけに `transform: "translateZ(0)"`・`backfaceVisibility: "hidden"`・`willChange: "transform"` を付与し、`styles.topbar` は `position: "static"`・`zIndex: "auto"`・`backgroundColor: t.bg`・`backdropFilter: "none"` とする。チャット画面ではトップバーの背後に本文がスクロールしないため半透明ブラーは不要であり、上部 UI 全体が 1 つのバッキングストアを共有することで、キーボードパン時に最上端のトップバーだけ独立して破棄・再描画遅延されるのを防げる。

### 2. 入力要素の祖先に `transform` を付けず、非入力行だけを分離して固定できる

[ADR-0024](./0024-mobile-dialog-scroll-and-input-zoom-on-ios.md) で確認した通り、`<textarea>` や `<input>` の祖先に `transform` が存在すると WebKit 上で UIKit 描画の青いカーソル位置がズレる。そこで `ComposerDock.Root` や `ComposerDock.Surface` には `transform` を付けず、入力要素を内包しない `ComposerDock.Header`（`ModeSwitch` と `TargetDatePicker` トリガー行）と `topChrome` のみを合成レイヤーへ固定することで、カーソル追従を損なわずにボタン類の描画抜けを防げる。

### 3. シェル高さの `100%` 統一と多重 `overflow: hidden` 解消でキーボード閉じ時の再計算を抑えられる

`html` / `body` / `#app` に `height: 100%` を `useLayoutEffect` で初回ペイント前から揃えたうえで、`styles.shell` を `height: "100dvh"` から `height: "100%"`・`overflow: "clip"` に揃え、`styles.workspace` から重複する `overflow: "hidden"` を外して `minHeight: 0` を指定する。これにより、iOS Safari のツールバーやキーボード終了に伴う `100dvh` 対 `100%` の高さ不整合と、不要なスクロールクリッピング用 `RenderLayer` の生成を抑えられる。

### 4. キーボード閉じ時の不要な同期リフローを取り除ける

`ChatComposer.Input` の自動高さ調整 `useLayoutEffect` は本文・選択曲・コンパクト幅のみに依存すれば十分であり、未使用の `keyboardOpen` を依存配列から外すことで、キーボード閉じ直後のフレームに `el.style.height = "0px"` と `getComputedStyle` / `scrollHeight` の強制同期リフローが走るのを防げる。

## 検討した他の案

- **`Topbar` の `backdrop-filter: blur(10px)` 削除のみ／`window.scrollTo(0, 0)` の 180ms デバウンス＋`requestAnimationFrame` 遅延**：実機 iPhone で再現が解消しなかったため不採用。
- **`Topbar`・`ListeningCalendarStrip`・`ComposerDock.Header` をそれぞれ個別の GPU レイヤーにする**：カレンダーとモードスイッチは改善したが、最上端の `Topbar` だけ単独の小さなオフスクリーンレイヤーとして破棄される挙動が残ったため不採用。
- **`shell`・`workspace`・`ComposerDock.Root` ごと `transform: translateZ(0)` でレイヤー化する**：入力欄（`<textarea>` / `<input>`）の祖先に `transform` が乗り、[ADR-0024](./0024-mobile-dialog-scroll-and-input-zoom-on-ios.md) で対処した WebKit のカーソルズレが再発するため不採用。
- **`@ui-sdk/core/web.css` の `@media (pointer: coarse)` による `position: relative` / `::after` 拡張ヒット領域を打ち消す**：モバイルでの 44px タップ領域を損なううえ、`Topbar` 内の非ボタン要素（「Recoda」ロゴ等）の消失までは防げないため不採用。
- **`blur` 時に `document.body.style.display = "none"` 等で強制リフローを起こす**：画面全体のチラつきやフォーカス・スクロール位置への副作用が大きいため不採用。

## この決定による影響

- `apps/website/src/chat/ChatWorkspace.tsx` では `useLayoutEffect` でルート高さを初期化し、`styles.shell`（`height: "100%"` / `overflow: "clip"`）・`styles.workspace`（`minHeight: 0`）・`styles.topChrome`・`styles.topbar` を用いる（[ADR-0023](./0023-ios-keyboard-layout-without-visual-viewport-resize.md) の `shell` 高さ `100dvh` 指定は本決定により `100%` へ更新された）。
- `apps/website/src/shell/AppShell.tsx` 自体は変更せず、`ProfileScreen` や `PostDetailScreen` の sticky 半透明＋`backdrop-filter` ヘッダーを維持する。
- `apps/website/src/chat/ComposerDock.tsx` は非入力行である `styles.header` のみを合成レイヤー化する。今後もチャット画面で合成レイヤーを追加する際は、入力要素（`<textarea>` / `<input>`）の祖先に `transform` を付けないこと。
- `apps/website/src/chat/ChatComposer.tsx` の高さ調整 `useLayoutEffect` に `keyboardOpen` を含めない。
- 本決定に基づく実装は PR #118（Issue #114）で反映済みである。
