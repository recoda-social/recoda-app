import { create, props } from "@stylexjs/stylex";
import { useEffect, useLayoutEffect } from "react";
import type { Me } from "#/lib/auth";
import { usePreviewPlayer } from "#/songs/preview-player";
import { ChatComposer } from "./ChatComposer";
import { useChat } from "./ChatContext";
import { ChatHistoryProvider } from "./ChatHistoryProvider";
import { ChatHistorySidebar } from "./ChatHistorySidebar";
import { ChatProvider } from "./ChatProvider";
import { ChatThread } from "./ChatThread";
import { ChatTopChrome } from "./ChatTopChrome";
import { ComposerDock } from "./ComposerDock";
import { DraftAssistProvider } from "./DraftAssistProvider";
import { ListeningCalendarStrip } from "./listening/ListeningCalendarStrip";
import { ListeningDayShelf } from "./listening/ListeningDayShelf";
import { ListeningComposer } from "./listening/ListeningLogComposer";
import { ListeningLogProvider } from "./listening/ListeningLogProvider";
import { chatMedia, chatTokens as t } from "./tokens.stylex";

function ChatShell({ user }: { user: Me }) {
  const { setKeyboardOpen } = useChat().actions;
  const { preloadAppleMusic } = usePreviewPlayer().actions;

  useEffect(() => {
    preloadAppleMusic();
  }, [preloadAppleMusic]);

  // Gemini Web と同じく、キーボード出現時は iOS Safari ネイティブの挙動に任せて
  // JS でシェルの高さ・位置を書き換えない。キーボードが閉じた後にスクロールずれが
  // 残った場合だけ window.scrollTo(0, 0) で戻す。
  useLayoutEffect(() => {
    const html = document.documentElement;
    const { body } = document;
    const appRoot = document.getElementById("app");
    const prevHtml = {
      width: html.style.width,
      height: html.style.height,
      overflow: html.style.overflow,
      overscrollBehavior: html.style.overscrollBehavior,
      backgroundColor: html.style.backgroundColor,
    };
    const prevBody = {
      width: body.style.width,
      height: body.style.height,
      overflow: body.style.overflow,
      overscrollBehavior: body.style.overscrollBehavior,
      backgroundColor: body.style.backgroundColor,
    };
    const prevAppHeight = appRoot?.style.height ?? "";

    const colorSchemeMedia = window.matchMedia("(prefers-color-scheme: dark)");
    const syncBackground = () => {
      const bg = colorSchemeMedia.matches ? "#171717" : "#ffffff";
      html.style.backgroundColor = bg;
      body.style.backgroundColor = bg;
    };

    html.style.width = "100%";
    html.style.height = "100%";
    html.style.overflow = "hidden";
    html.style.overscrollBehavior = "none";
    body.style.width = "100%";
    body.style.height = "100%";
    body.style.overflow = "hidden";
    body.style.overscrollBehavior = "none";
    if (appRoot) {
      appRoot.style.height = "100%";
    }
    syncBackground();
    colorSchemeMedia.addEventListener("change", syncBackground);
    window.scrollTo(0, 0);

    const vv = window.visualViewport;
    let lastWidth = vv ? vv.width : window.innerWidth;
    let baseHeight = vv ? vv.height : window.innerHeight;

    const sync = () => {
      const currentWidth = vv ? vv.width : window.innerWidth;
      const currentHeight = vv ? vv.height : window.innerHeight;
      const hasInputFocus =
        document.activeElement instanceof HTMLTextAreaElement ||
        document.activeElement instanceof HTMLInputElement;

      if (Math.abs(currentWidth - lastWidth) > 20) {
        lastWidth = currentWidth;
        baseHeight = currentHeight;
      } else if (!hasInputFocus) {
        baseHeight = Math.max(baseHeight, currentHeight);
      }

      if (vv) {
        if (Math.abs(vv.scale - 1) > 0.01) {
          return;
        }
        const keyboardClosed = !(vv.height < html.clientHeight - 150);
        if (keyboardClosed && (window.scrollY > 0 || window.scrollX > 0)) {
          window.scrollTo(0, 0);
        }
        setKeyboardOpen(
          window.innerHeight - vv.height > 100 || (hasInputFocus && baseHeight - vv.height > 100),
        );
        return;
      }

      setKeyboardOpen(hasInputFocus && baseHeight - window.innerHeight > 100);
    };

    sync();

    if (vv) {
      vv.addEventListener("resize", sync);
    }
    window.addEventListener("scroll", sync, { passive: true });
    window.addEventListener("resize", sync);
    document.addEventListener("focusin", sync);
    document.addEventListener("focusout", sync);

    return () => {
      colorSchemeMedia.removeEventListener("change", syncBackground);
      html.style.width = prevHtml.width;
      html.style.height = prevHtml.height;
      html.style.overflow = prevHtml.overflow;
      html.style.overscrollBehavior = prevHtml.overscrollBehavior;
      html.style.backgroundColor = prevHtml.backgroundColor;
      body.style.width = prevBody.width;
      body.style.height = prevBody.height;
      body.style.overflow = prevBody.overflow;
      body.style.overscrollBehavior = prevBody.overscrollBehavior;
      body.style.backgroundColor = prevBody.backgroundColor;
      if (appRoot) {
        appRoot.style.height = prevAppHeight;
      }
      if (vv) {
        vv.removeEventListener("resize", sync);
      }
      window.removeEventListener("scroll", sync);
      window.removeEventListener("resize", sync);
      document.removeEventListener("focusin", sync);
      document.removeEventListener("focusout", sync);
    };
  }, [setKeyboardOpen]);

  return (
    <div data-chat-shell="" {...props(styles.shell)}>
      <ChatHistorySidebar.Desktop />
      <ChatHistorySidebar.MobileDrawer />

      <div {...props(styles.workspace)}>
        <div {...props(styles.upperStage)}>
          <ChatTopChrome.Root>
            <ChatTopChrome.Bar>
              <ChatTopChrome.Topbar>
                <ChatTopChrome.Brand />
                <ChatTopChrome.Account />
              </ChatTopChrome.Topbar>
              <ChatTopChrome.Player />
            </ChatTopChrome.Bar>

            <ListeningCalendarStrip />
          </ChatTopChrome.Root>

          <main {...props(styles.main)}>
            <div {...props(styles.mainInner)}>
              <ListeningDayShelf />
              <ChatThread user={user} />
            </div>
          </main>
        </div>

        <ComposerDock.Root>
          <ComposerDock.Header>
            <ListeningComposer.ModeSwitch />
            <ComposerDock.Panel mode="log">
              <ListeningComposer.TargetDatePicker />
            </ComposerDock.Panel>
          </ComposerDock.Header>

          <ComposerDock.Surface>
            <ComposerDock.Panel mode="review">
              <ChatComposer.Form>
                <ChatComposer.DraftAssist />
                <ChatComposer.SongPicker />
                <ChatComposer.Attachment />
                <ChatComposer.Input />
                <ChatComposer.EmptyHints />
                <ChatComposer.Toolbar>
                  <ChatComposer.ToolGroup>
                    <ChatComposer.PickSongButton />
                    <ChatComposer.AssistButton />
                  </ChatComposer.ToolGroup>
                  <ChatComposer.Submit />
                </ChatComposer.Toolbar>
              </ChatComposer.Form>
            </ComposerDock.Panel>

            <ComposerDock.Panel mode="log">
              <ListeningComposer.SearchCombobox />
            </ComposerDock.Panel>
          </ComposerDock.Surface>
        </ComposerDock.Root>
      </div>
    </div>
  );
}

export function ChatWorkspace({ user }: { user: Me }) {
  return (
    <ChatHistoryProvider publicId={user.publicId}>
      <ChatProvider userPublicId={user.publicId}>
        <ListeningLogProvider>
          <DraftAssistProvider>
            <ChatTopChrome.Provider>
              <ChatShell user={user} />
            </ChatTopChrome.Provider>
          </DraftAssistProvider>
        </ListeningLogProvider>
      </ChatProvider>
    </ChatHistoryProvider>
  );
}

const styles = create({
  shell: {
    position: "relative",
    display: "flex",
    flexDirection: "row",
    width: "100%",
    height: "100%",
    overflow: "clip",
    color: t.text,
    colorScheme: "light dark",
    backgroundColor: t.bg,
  },
  workspace: {
    display: "flex",
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: "0%",
    flexDirection: "column",
    minWidth: 0,
    height: "100%",
    minHeight: 0,
  },
  // Topbar・MiniPlayerBar・カレンダーストリップとメイン領域を単一の合成レイヤーへまとめることで、
  // 感想入力中に Topbar を非表示にしても上部レイヤーがオフスクリーン破棄されず、
  // blur 時にカレンダー全体がチラつく WebKit の挙動を防ぐ（#114 / #115）。
  // ComposerDock（入力欄）は兄弟要素に置くため、入力要素の祖先に transform は付かない（ADR 0024）。
  upperStage: {
    display: "flex",
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: "0%",
    flexDirection: "column",
    minHeight: 0,
    backgroundColor: t.bg,
    transform: "translateZ(0)",
    backfaceVisibility: "hidden",
    willChange: "transform",
  },
  main: {
    flexGrow: "1",
    flexShrink: "1",
    flexBasis: "0%",
    width: "100%",
    minHeight: 0,
    overflowY: "auto",
    overscrollBehaviorY: "contain",
  },
  mainInner: {
    display: "flex",
    flexDirection: "column",
    width: "min(100%, 760px)",
    minHeight: "100%",
    paddingTop: {
      default: 24,
      [chatMedia.compact]: 12,
    },
    paddingRight: {
      default: 16,
      [chatMedia.compact]: 12,
    },
    paddingBottom: {
      default: 32,
      [chatMedia.compact]: 16,
    },
    paddingLeft: {
      default: 16,
      [chatMedia.compact]: 12,
    },
    marginBlock: 0,
    marginInline: "auto",
  },
});
