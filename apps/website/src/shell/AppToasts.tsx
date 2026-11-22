import { create, props } from "@stylexjs/stylex";
import type { StyleXStyles } from "@stylexjs/stylex";
import type { Toaster } from "@ui-sdk/core/toast";
import {
  createToaster,
  Toast,
  ToastAction,
  ToastDescription,
  ToastRegion,
  ToastTitle,
} from "@ui-sdk/react/toast";
import { createContext, use, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { chatMedia, chatTokens as t } from "#/chat/tokens.stylex";
import { motion } from "#/motion.stylex";

type AppToastState = Record<string, never>;

type AppToastActions = {
  success: (title: string) => void;
  error: (title: string) => void;
  dismiss: () => void;
};

type AppToastMeta = {
  toaster: Toaster<ReactNode>;
};

export type AppToastContextValue = {
  state: AppToastState;
  actions: AppToastActions;
  meta: AppToastMeta;
};

const EMPTY_STATE: AppToastState = {};

const AppToastContext = createContext<AppToastContextValue | null>(null);

export function useAppToast() {
  const value = use(AppToastContext);
  if (!value) {
    throw new Error("useAppToast は AppToasts.Provider の中で使う");
  }
  return value;
}

function Provider({ duration = 4000, children }: { duration?: number; children: ReactNode }) {
  const [toaster] = useState(() => createToaster({ duration }));

  useEffect(() => () => toaster.dismiss(), [toaster]);

  const actions = useMemo<AppToastActions>(
    () => ({
      success: (title) => toaster.success(title),
      error: (title) => toaster.error(title),
      dismiss: () => toaster.dismiss(),
    }),
    [toaster],
  );

  const meta = useMemo<AppToastMeta>(() => ({ toaster }), [toaster]);

  return (
    <AppToastContext value={{ state: EMPTY_STATE, actions, meta }}>{children}</AppToastContext>
  );
}

type ViewportProps = {
  side?: "top" | "bottom";
  align?: "start" | "center" | "end";
  regionStyle?: StyleXStyles;
};

function Viewport({ side = "bottom", align = "end", regionStyle }: ViewportProps) {
  const { meta } = useAppToast();

  return (
    <ToastRegion
      toaster={meta.toaster}
      data-agent-hidden=""
      side={side}
      align={align}
      {...props(regionStyle)}
    >
      {() => (
        <Toast {...props(styles.toast)}>
          <div {...props(styles.toastBody)}>
            <ToastTitle {...props(styles.toastTitle)} />
            <ToastDescription {...props(styles.toastDesc)} />
          </div>
          <ToastAction {...props(styles.toastAction)} />
        </Toast>
      )}
    </ToastRegion>
  );
}

export const AppToasts = {
  Provider,
  Viewport,
};

const styles = create({
  toast: {
    display: "flex",
    gap: 12,
    alignItems: "center",
    justifyContent: "space-between",
    paddingBlock: 12,
    paddingInline: 14,
    color: t.text,
    backgroundColor: t.surfaceElevated,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 12,
    boxShadow: t.shadow,
  },
  toastBody: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
    minWidth: 0,
  },
  toastTitle: {
    fontSize: 13,
    fontWeight: 600,
  },
  toastDesc: {
    fontSize: 12,
    color: t.muted,
  },
  toastAction: {
    flexShrink: 0,
    paddingBlock: 6,
    paddingInline: 10,
    fontSize: 12,
    fontWeight: 600,
    color: t.primaryFg,
    cursor: "pointer",
    backgroundColor: {
      default: t.primary,
      ":hover": {
        default: t.primary,
        [chatMedia.hover]: t.borderStrong,
      },
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 8,
    transition: `background-color 150ms ${motion.easeOut}`,
  },
});
