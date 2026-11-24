import { create, props } from "@stylexjs/stylex";
import { Link } from "@tanstack/react-router";
import { Avatar, AvatarFallback, AvatarImage } from "@ui-sdk/react/avatar";
import { createContext, use } from "react";
import type { ReactNode } from "react";
import { useAuth } from "#/auth/AuthContext";
import { AppShell } from "#/shell/AppShell";
import { MiniPlayerBar } from "#/songs/MiniPlayerBar";
import { useChat } from "./ChatContext";
import { useChatHistory } from "./ChatHistoryContext";
import { ChatHistorySidebar } from "./ChatHistorySidebar";
import { useListeningLog } from "./listening/ListeningLogContext";
import { chatMedia, chatTokens as t } from "./tokens.stylex";

type ChatTopChromeState = {
  isHeaderHidden: boolean;
};

type ChatTopChromeContextValue = {
  state: ChatTopChromeState;
  actions: Record<string, never>;
  meta: Record<string, never>;
};

const EMPTY_RECORD: Record<string, never> = {};

const ChatTopChromeContext = createContext<ChatTopChromeContextValue | null>(null);

function useChatTopChrome() {
  const value = use(ChatTopChromeContext);
  if (!value) {
    throw new Error("ChatTopChrome の部品は ChatTopChrome.Provider の中で使う");
  }
  return value;
}

function initials(name: string) {
  return Array.from(name.trim()).slice(0, 2).join("") || "U";
}

function Provider({ children }: { children: ReactNode }) {
  const { state: chatState } = useChat();
  const { state: listeningState } = useListeningLog();
  const isHeaderHidden =
    listeningState.composerMode === "review" &&
    chatState.selectedSong !== null &&
    chatState.keyboardOpen;

  return (
    <ChatTopChromeContext
      value={{
        state: { isHeaderHidden },
        actions: EMPTY_RECORD,
        meta: EMPTY_RECORD,
      }}
    >
      {children}
    </ChatTopChromeContext>
  );
}

function Root({ children }: { children: ReactNode }) {
  return <div {...props(styles.topChrome)}>{children}</div>;
}

function Bar({ children }: { children: ReactNode }) {
  const { state } = useChatTopChrome();

  return (
    <div
      data-chat-top-chrome=""
      {...props(styles.bar, state.isHeaderHidden && styles.barHeaderHidden)}
    >
      {children}
    </div>
  );
}

function Topbar({ children }: { children: ReactNode }) {
  const { state } = useChatTopChrome();
  if (state.isHeaderHidden) {
    return null;
  }

  return <AppShell.Topbar style={styles.topbar}>{children}</AppShell.Topbar>;
}

function Brand() {
  const { state } = useChatHistory();
  const showBrand = !(state.isDesktop && state.desktopOpen);

  return (
    <div {...props(styles.brand)}>
      <ChatHistorySidebar.OpenTrigger />
      {showBrand ? (
        <>
          <AppShell.BrandIcon />
          <AppShell.BrandName />
        </>
      ) : null}
    </div>
  );
}

function Account() {
  const { state } = useAuth();
  const { user } = state;
  if (!user) {
    return null;
  }

  return (
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
  );
}

function Player() {
  return <MiniPlayerBar style={styles.miniPlayer} />;
}

export const ChatTopChrome = {
  Provider,
  Root,
  Bar,
  Topbar,
  Brand,
  Account,
  Player,
};

const styles = create({
  // ChatWorkspace の upperStage 合成レイヤー内で描画され、複数行入力時は
  // 子要素の ListeningCalendarStrip だけが縮むように flexShrink: 1 を維持する（#114 / #115）。
  topChrome: {
    display: "flex",
    flexGrow: 0,
    flexShrink: 1,
    flexBasis: "auto",
    flexDirection: "column",
    minHeight: 0,
    backgroundColor: t.bg,
  },
  bar: {
    display: "flex",
    flexShrink: 0,
    flexDirection: "column",
    backgroundColor: t.bg,
  },
  // 感想入力中に Topbar を非表示にしたときも、MiniPlayerBar やその下の要素が
  // ノッチ／ステータスバーと重ならないよう上部セーフエリアを確保する。
  barHeaderHidden: {
    paddingTop: "env(safe-area-inset-top, 0px)",
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
  // MiniPlayerBar も topChrome 内で背景スクロールを伴わないため、
  // 半透明＋backdrop-filter を外して topChrome と同一レイヤーで描画する。
  miniPlayer: {
    zIndex: "auto",
    backgroundColor: t.surface,
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
});
