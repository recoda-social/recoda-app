import { create, props } from "@stylexjs/stylex";
import type { ReactNode } from "react";
import { useChat } from "./ChatContext";
import { useListeningLog } from "./listening/ListeningLogContext";
import type { ComposerMode } from "./listening/ListeningLogContext";
import { chatMedia, chatTokens as t } from "./tokens.stylex";

function Root({ children }: { children: ReactNode }) {
  const { state } = useChat();

  return (
    <div {...props(styles.dock, state.keyboardOpen && styles.dockKeyboardOpen)}>{children}</div>
  );
}

function Header({ children }: { children: ReactNode }) {
  return <div {...props(styles.header)}>{children}</div>;
}

function Surface({ children }: { children: ReactNode }) {
  const { actions } = useChat();
  const { registerComposer } = actions;

  return (
    <div ref={registerComposer} data-chat-composer="" {...props(styles.surface)}>
      {children}
    </div>
  );
}

function Panel({ mode, children }: { mode: ComposerMode; children: ReactNode }) {
  const { state } = useListeningLog();
  if (state.composerMode !== mode) {
    return null;
  }
  return children;
}

export const ComposerDock = {
  Root,
  Header,
  Surface,
  Panel,
};

const styles = create({
  dock: {
    position: "relative",
    zIndex: 20,
    display: "flex",
    flexShrink: 0,
    flexDirection: "column",
    gap: {
      default: 8,
      [chatMedia.compact]: 6,
    },
    width: "min(100%, 760px)",
    paddingTop: {
      default: 8,
      [chatMedia.compact]: 4,
    },
    paddingRight: {
      default: "max(16px, env(safe-area-inset-right, 0px))",
      [chatMedia.compact]: "max(12px, env(safe-area-inset-right, 0px))",
    },
    paddingBottom: {
      default: "max(20px, env(safe-area-inset-bottom, 0px))",
      [chatMedia.compact]: "max(12px, env(safe-area-inset-bottom, 0px))",
    },
    paddingLeft: {
      default: "max(16px, env(safe-area-inset-left, 0px))",
      [chatMedia.compact]: "max(12px, env(safe-area-inset-left, 0px))",
    },
    marginBlock: 0,
    marginInline: "auto",
    backgroundImage: `linear-gradient(to bottom, color-mix(in srgb, ${t.bg} 0%, transparent), ${t.bg} 28%)`,
  },
  dockKeyboardOpen: {
    paddingBottom: {
      default: 12,
      [chatMedia.compact]: 8,
    },
  },
  header: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    alignItems: "center",
    justifyContent: "space-between",
    paddingInline: 2,
  },
  surface: {
    position: "relative",
    display: "flex",
    flexDirection: "column",
    gap: {
      default: 8,
      [chatMedia.compact]: 4,
    },
    paddingTop: {
      default: 12,
      [chatMedia.compact]: 8,
    },
    paddingRight: {
      default: 14,
      [chatMedia.compact]: 10,
    },
    paddingBottom: {
      default: 10,
      [chatMedia.compact]: 8,
    },
    paddingLeft: {
      default: 14,
      [chatMedia.compact]: 12,
    },
    backgroundColor: t.surface,
    borderColor: {
      default: t.border,
      ":focus-within": t.borderStrong,
    },
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: {
      default: 22,
      [chatMedia.compact]: 18,
    },
    boxShadow: t.shadow,
    transition: "border-color 0.15s ease",
  },
});
