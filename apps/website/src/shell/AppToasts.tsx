import { create, props } from "@stylexjs/stylex";
import type { StyleXStyles } from "@stylexjs/stylex";
import type { Toaster } from "@ui-sdk/core/toast";
import { Toast, ToastAction, ToastDescription, ToastRegion, ToastTitle } from "@ui-sdk/react/toast";
import type { ReactNode } from "react";
import { chatMedia, chatTokens as t } from "#/chat/tokens.stylex";
import { motion } from "#/motion.stylex";

type AppToastsProps = {
  toaster: Toaster<ReactNode>;
  side?: "top" | "bottom";
  align?: "start" | "center" | "end";
  regionStyle?: StyleXStyles;
};

export function AppToasts({
  toaster,
  side = "bottom",
  align = "end",
  regionStyle,
}: AppToastsProps) {
  return (
    <ToastRegion
      toaster={toaster}
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
