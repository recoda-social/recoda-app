import { create, props } from "@stylexjs/stylex";
import { Link } from "@tanstack/react-router";
import { Avatar, AvatarFallback, AvatarImage } from "@ui-sdk/react/avatar";
import { useEffect, useLayoutEffect } from "react";
import type { Me } from "#/lib/auth";
import { MiniPlayerBar } from "#/songs/MiniPlayerBar";
import { usePreviewPlayer } from "#/songs/preview-player";
import { ChatComposer } from "./ChatComposer";
import { ComposerDock } from "./ComposerDock";
import { AppShell } from "#/shell/AppShell";
import { useChat } from "./ChatContext";
import { useChatHistory } from "./ChatHistoryContext";
import { ChatHistoryProvider } from "./ChatHistoryProvider";
import { ChatHistorySidebar } from "./ChatHistorySidebar";
import { ChatProvider } from "./ChatProvider";
import { ChatThread } from "./ChatThread";
import { DraftAssistProvider } from "./DraftAssistProvider";
import { ListeningCalendarStrip } from "./listening/ListeningCalendarStrip";
import { ListeningDayShelf } from "./listening/ListeningDayShelf";
import { ListeningComposer } from "./listening/ListeningLogComposer";
import { ListeningLogProvider } from "./listening/ListeningLogProvider";
import { chatMedia, chatTokens as t } from "./tokens.stylex";

function initials(name: string) {
  return Array.from(name.trim()).slice(0, 2).join("") || "U";
}

function ChatShell({ user }: { user: Me }) {
  const { setKeyboardOpen } = useChat().actions;
  const { state: historyState } = useChatHistory();
  const { preloadAppleMusic } = usePreviewPlayer().actions;
  const showTopbarBrand = !(historyState.isDesktop && historyState.desktopOpen);

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
        <div {...props(styles.topChrome)}>
          <AppShell.Topbar style={styles.topbar}>
            <div {...props(styles.brand)}>
              <ChatHistorySidebar.OpenTrigger />
              {showTopbarBrand ? (
                <>
                  <AppShell.BrandIcon />
                  <AppShell.BrandName />
                </>
              ) : null}
            </div>
            <div {...props(styles.account)}>
              <Link to="/profile" aria-label="プロフィールを開く" {...props(styles.profileLink)}>
                <Avatar {...props(styles.avatarSm)}>
                  {user.image ? (
                    <AvatarImage
                      src={user.image}
                      alt={user.name}
                      referrerPolicy="no-referrer"
                      {...props(styles.avatarImg)}
                    />
                  ) : null}
                  <AvatarFallback>
                    <span aria-hidden="true">{initials(user.name)}</span>
                    <span data-slot="visually-hidden">{user.name}</span>
                  </AvatarFallback>
                </Avatar>
                <span {...props(styles.accountName)}>{user.name}</span>
              </Link>
              <AppShell.SignOutButton />
            </div>
          </AppShell.Topbar>

          <MiniPlayerBar />

          <ListeningCalendarStrip />
        </div>

        <main {...props(styles.main)}>
          <div {...props(styles.mainInner)}>
            <ListeningDayShelf />
            <ChatThread user={user} />
          </div>
        </main>

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
            <ChatShell user={user} />
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
  // Topbar とカレンダーストリップを単一の合成レイヤーへまとめることで、
  // キーボード表示中に最上端のトップバー（y = 0 付近）だけが単独でオフスクリーン破棄され
  // blur 時に一瞬消える WebKit の挙動を防ぐ。
  topChrome: {
    display: "flex",
    flexShrink: 0,
    flexDirection: "column",
    backgroundColor: t.bg,
    transform: "translateZ(0)",
    backfaceVisibility: "hidden",
    willChange: "transform",
  },
  // ChatWorkspace ではトップバー背後にコンテンツがスクロールしないため、
  // 半透明＋backdrop-filter を外して topChrome と同一レイヤーで描画する。
  topbar: {
    position: "static",
    zIndex: "auto",
    flexShrink: 0,
    backgroundColor: t.bg,
    backdropFilter: "none",
  },
  brand: {
    display: "flex",
    gap: 8,
    alignItems: "center",
  },
  account: {
    display: "flex",
    gap: 10,
    alignItems: "center",
  },
  profileLink: {
    display: "inline-flex",
    gap: 8,
    alignItems: "center",
    paddingBlock: 4,
    paddingInline: 8,
    color: "inherit",
    textDecoration: "none",
    backgroundColor: {
      default: "transparent",
      ":hover": {
        default: "transparent",
        [chatMedia.hover]: t.hover,
      },
    },
    borderRadius: 9999,
    transition: "background-color 150ms ease",
  },
  avatarSm: {
    display: "inline-flex",
    flexShrink: 0,
    alignItems: "center",
    justifyContent: "center",
    width: 24,
    height: 24,
    overflow: "hidden",
    fontSize: 11,
    borderRadius: 9999,
  },
  avatarImg: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  accountName: {
    display: {
      default: "inline",
      [chatMedia.compact]: "none",
    },
    fontSize: 13,
    color: t.muted,
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
