import { create, props } from "@stylexjs/stylex";
import { Link } from "@tanstack/react-router";
import { Avatar, AvatarFallback, AvatarImage } from "@ui-sdk/react/avatar";
import { useEffect, useRef } from "react";
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
  const shellRef = useRef<HTMLDivElement | null>(null);
  const { setKeyboardOpen } = useChat().actions;
  const { state: historyState } = useChatHistory();
  const { preloadAppleMusic } = usePreviewPlayer().actions;
  const showTopbarBrand = !(historyState.isDesktop && historyState.desktopOpen);

  useEffect(() => {
    preloadAppleMusic();
  }, [preloadAppleMusic]);

  // iOS Safari などでソフトウェアキーボード表示中に画面をスクロールしても
  // 入力欄がキーボード裏へ隠れないよう、シェルを visualViewport に追従させる。
  useEffect(() => {
    const shellEl = shellRef.current;
    if (!shellEl) {
      return;
    }

    const html = document.documentElement;
    const { body } = document;
    const prevHtml = {
      position: html.style.position,
      inset: html.style.inset,
      width: html.style.width,
      height: html.style.height,
      overflow: html.style.overflow,
      overscrollBehavior: html.style.overscrollBehavior,
      backgroundColor: html.style.backgroundColor,
    };
    const prevBody = {
      position: body.style.position,
      inset: body.style.inset,
      width: body.style.width,
      height: body.style.height,
      overflow: body.style.overflow,
      overscrollBehavior: body.style.overscrollBehavior,
      backgroundColor: body.style.backgroundColor,
    };

    const colorSchemeMedia = window.matchMedia("(prefers-color-scheme: dark)");
    const syncBackground = () => {
      const bg = colorSchemeMedia.matches ? "#171717" : "#ffffff";
      html.style.backgroundColor = bg;
      body.style.backgroundColor = bg;
    };

    html.style.position = "fixed";
    html.style.inset = "0";
    html.style.width = "100%";
    html.style.height = "100%";
    html.style.overflow = "hidden";
    html.style.overscrollBehavior = "none";
    body.style.position = "fixed";
    body.style.inset = "0";
    body.style.width = "100%";
    body.style.height = "100%";
    body.style.overflow = "hidden";
    body.style.overscrollBehavior = "none";
    syncBackground();
    colorSchemeMedia.addEventListener("change", syncBackground);
    window.scrollTo(0, 0);

    const vv = window.visualViewport;
    let lastWidth = vv ? vv.width : window.innerWidth;
    let baseHeight = vv ? vv.height : window.innerHeight;

    const sync = () => {
      if (window.scrollY !== 0 || window.scrollX !== 0) {
        window.scrollTo(0, 0);
      }
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
        shellEl.style.height = `${vv.height}px`;
        shellEl.style.top = `${vv.offsetTop}px`;
        setKeyboardOpen(
          window.innerHeight - vv.height > 100 || (hasInputFocus && baseHeight - vv.height > 100),
        );
        return;
      }

      shellEl.style.height = `${window.innerHeight}px`;
      shellEl.style.top = "0px";
      setKeyboardOpen(hasInputFocus && baseHeight - window.innerHeight > 100);
    };

    sync();

    if (vv) {
      vv.addEventListener("resize", sync);
      vv.addEventListener("scroll", sync);
    }
    window.addEventListener("resize", sync);
    document.addEventListener("focusin", sync);
    document.addEventListener("focusout", sync);

    return () => {
      colorSchemeMedia.removeEventListener("change", syncBackground);
      html.style.position = prevHtml.position;
      html.style.inset = prevHtml.inset;
      html.style.width = prevHtml.width;
      html.style.height = prevHtml.height;
      html.style.overflow = prevHtml.overflow;
      html.style.overscrollBehavior = prevHtml.overscrollBehavior;
      html.style.backgroundColor = prevHtml.backgroundColor;
      body.style.position = prevBody.position;
      body.style.inset = prevBody.inset;
      body.style.width = prevBody.width;
      body.style.height = prevBody.height;
      body.style.overflow = prevBody.overflow;
      body.style.overscrollBehavior = prevBody.overscrollBehavior;
      body.style.backgroundColor = prevBody.backgroundColor;
      if (vv) {
        vv.removeEventListener("resize", sync);
        vv.removeEventListener("scroll", sync);
      }
      window.removeEventListener("resize", sync);
      document.removeEventListener("focusin", sync);
      document.removeEventListener("focusout", sync);
    };
  }, [setKeyboardOpen]);

  return (
    <div ref={shellRef} data-chat-shell="" {...props(styles.shell)}>
      <ChatHistorySidebar.Desktop />
      <ChatHistorySidebar.MobileDrawer />

      <div {...props(styles.workspace)}>
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
              <ListeningComposer.SearchPopover />
              <ListeningComposer.Input />
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
    position: "fixed",
    top: 0,
    right: 0,
    left: 0,
    display: "flex",
    flexDirection: "row",
    height: "100dvh",
    overflow: "hidden",
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
    overflow: "hidden",
  },
  topbar: {
    position: "relative",
    flexShrink: 0,
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
