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
    width: "100%",
    paddingTop: {
      default: 16,
      [chatMedia.compact]: 12,
    },
    paddingRight: {
      default: "max(16px, env(safe-area-inset-right, 0px), calc((100% - 760px) / 2 + 16px))",
      [chatMedia.compact]: "max(12px, env(safe-area-inset-right, 0px))",
    },
    paddingBottom: {
      default: "max(20px, env(safe-area-inset-bottom, 0px))",
      [chatMedia.compact]: "max(12px, env(safe-area-inset-bottom, 0px))",
    },
    paddingLeft: {
      default: "max(16px, env(safe-area-inset-left, 0px), calc((100% - 760px) / 2 + 16px))",
      [chatMedia.compact]: "max(12px, env(safe-area-inset-left, 0px))",
    },
    marginTop: {
      default: -56,
      [chatMedia.compact]: -50,
    },
    marginRight: 0,
    marginBottom: 0,
    marginLeft: 0,
    pointerEvents: "none",
    backgroundImage: {
      default: `linear-gradient(to bottom, color-mix(in srgb, ${t.bg} 0%, transparent) 0px, color-mix(in srgb, ${t.bg} 2.4%, transparent) 5.6px, color-mix(in srgb, ${t.bg} 9.5%, transparent) 11.2px, color-mix(in srgb, ${t.bg} 20.6%, transparent) 16.8px, color-mix(in srgb, ${t.bg} 34.5%, transparent) 22.4px, color-mix(in srgb, ${t.bg} 50%, transparent) 28px, color-mix(in srgb, ${t.bg} 65.5%, transparent) 33.6px, color-mix(in srgb, ${t.bg} 79.4%, transparent) 39.2px, color-mix(in srgb, ${t.bg} 90.5%, transparent) 44.8px, color-mix(in srgb, ${t.bg} 97.6%, transparent) 50.4px, ${t.bg} 56px)`,
      [chatMedia.compact]: `linear-gradient(to bottom, color-mix(in srgb, ${t.bg} 0%, transparent) 0px, color-mix(in srgb, ${t.bg} 2.4%, transparent) 5px, color-mix(in srgb, ${t.bg} 9.5%, transparent) 10px, color-mix(in srgb, ${t.bg} 20.6%, transparent) 15px, color-mix(in srgb, ${t.bg} 34.5%, transparent) 20px, color-mix(in srgb, ${t.bg} 50%, transparent) 25px, color-mix(in srgb, ${t.bg} 65.5%, transparent) 30px, color-mix(in srgb, ${t.bg} 79.4%, transparent) 35px, color-mix(in srgb, ${t.bg} 90.5%, transparent) 40px, color-mix(in srgb, ${t.bg} 97.6%, transparent) 45px, ${t.bg} 50px)`,
    },
  },
  dockKeyboardOpen: {
    paddingBottom: {
      default: 12,
      [chatMedia.compact]: 8,
    },
  },
  // ModeSwitch や TargetDatePicker を含むヘッダー行は入力要素を内包しないため、
  // 合成レイヤーへ固定して iOS Safari の blur 時チラつきを防ぐ（入力欄の祖先には付けない）。
  header: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    alignItems: "center",
    justifyContent: "space-between",
    paddingInline: 2,
    pointerEvents: "none",
    transform: "translateZ(0)",
    backfaceVisibility: "hidden",
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
    pointerEvents: "auto",
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
    transition: "border-color 0.15s ease",
  },
});
