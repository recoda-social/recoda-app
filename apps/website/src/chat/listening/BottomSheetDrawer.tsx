import { create, props } from "@stylexjs/stylex";
import {
  Drawer,
  DrawerBackdrop,
  DrawerClose,
  DrawerContent,
  DrawerTitle,
  DrawerTrigger,
} from "@ui-sdk/react/drawer";
import { createContext, use, useCallback, useLayoutEffect, useRef } from "react";
import type { ReactNode } from "react";
import { motion } from "#/motion.stylex";
import { chatTokens as t } from "../tokens.stylex";

type BottomSheetContextValue = {
  state: {
    open: boolean;
  };
  actions: {
    registerBackdrop: (el: HTMLDivElement | null) => void;
    registerContent: (el: HTMLDivElement | null) => void;
    cancelOpenAnimations: () => void;
  };
  meta: Record<string, never>;
};

const EMPTY_META: Record<string, never> = {};

const BottomSheetContext = createContext<BottomSheetContextValue | null>(null);

function useBottomSheet() {
  const value = use(BottomSheetContext);
  if (!value) {
    throw new Error("BottomSheetDrawer の部品は BottomSheetDrawer.Root の中で使う");
  }
  return value;
}

// @ui-sdk/react/drawer の iOS Safari で初回マウント時の開きアニメーションが効かないため、
// 開く瞬間の 1 回だけ WAAPI でシート上昇と背景の濃さを同一カーブで同期させる（fill なしなのでドラッグ・閉じは ui-sdk に完全に委ねる）。
function Root({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}) {
  const backdropRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const animationsRef = useRef<Array<Animation>>([]);

  const registerBackdrop = useCallback((el: HTMLDivElement | null) => {
    backdropRef.current = el;
  }, []);

  const registerContent = useCallback((el: HTMLDivElement | null) => {
    contentRef.current = el;
  }, []);

  const cancelOpenAnimations = useCallback(() => {
    for (const anim of animationsRef.current) {
      anim.cancel();
    }
    animationsRef.current = [];
  }, []);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }
    const backdropEl = backdropRef.current;
    const contentEl = contentRef.current;
    if (
      !backdropEl ||
      !contentEl ||
      typeof backdropEl.animate !== "function" ||
      typeof contentEl.animate !== "function"
    ) {
      return;
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const duration = reducedMotion ? 150 : 300;
    const easing = reducedMotion ? "ease" : motion.easeDrawer;

    const backdropAnim = backdropEl.animate([{ opacity: 0 }, { opacity: 1 }], {
      duration,
      easing,
    });
    const contentAnim = reducedMotion
      ? contentEl.animate([{ opacity: 0 }, { opacity: 1 }], { duration, easing })
      : contentEl.animate([{ transform: "translateY(100%)" }, { transform: "translateY(0)" }], {
          duration,
          easing,
        });

    animationsRef.current = [backdropAnim, contentAnim];

    return () => {
      backdropAnim.cancel();
      contentAnim.cancel();
      animationsRef.current = [];
    };
  }, [open]);

  return (
    <BottomSheetContext
      value={{
        state: { open },
        actions: { registerBackdrop, registerContent, cancelOpenAnimations },
        meta: EMPTY_META,
      }}
    >
      <Drawer side="bottom" open={open} onOpenChange={onOpenChange}>
        {children}
      </Drawer>
    </BottomSheetContext>
  );
}

function Content({ children }: { children: ReactNode }) {
  const { actions } = useBottomSheet();
  const { registerBackdrop, registerContent, cancelOpenAnimations } = actions;

  return (
    <>
      <DrawerBackdrop ref={registerBackdrop} {...props(styles.backdrop)} />
      <DrawerContent
        ref={registerContent}
        initialFocus="content"
        onPointerDown={cancelOpenAnimations}
        {...props(styles.content)}
      >
        <div {...props(styles.grip)} aria-hidden="true" />
        {children}
      </DrawerContent>
    </>
  );
}

export const BottomSheetDrawer = {
  Root,
  Trigger: DrawerTrigger,
  Content,
  Title: DrawerTitle,
  Close: DrawerClose,
};

const styles = create({
  backdrop: {
    backgroundColor: {
      default: "rgba(0, 0, 0, 0.48)",
      "@media (prefers-color-scheme: dark)": "rgba(0, 0, 0, 0.68)",
    },
  },
  content: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
    width: "100%",
    paddingTop: 10,
    paddingRight: "max(16px, env(safe-area-inset-right, 0px))",
    paddingBottom: "max(20px, env(safe-area-inset-bottom, 0px))",
    paddingLeft: "max(16px, env(safe-area-inset-left, 0px))",
    color: t.text,
    backgroundColor: t.surfaceElevated,
    borderTopColor: t.border,
    borderTopStyle: "solid",
    borderTopWidth: 1,
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    borderBottomRightRadius: 0,
    borderBottomLeftRadius: 0,
    boxShadow: t.shadow,
  },
  grip: {
    alignSelf: "center",
    width: 36,
    height: 4,
    backgroundColor: t.borderStrong,
    borderRadius: 9999,
  },
});
