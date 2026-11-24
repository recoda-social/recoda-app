import { useLayoutEffect, useState } from "react";
import type { RefObject } from "react";
import { chatPopover } from "./tokens.stylex";

export function useComposerPopoverHeight({
  open,
  keyboardOpen,
  composerEl,
  composerRef,
}: {
  open: boolean;
  keyboardOpen: boolean;
  composerEl: HTMLElement | null;
  composerRef: RefObject<HTMLElement | null>;
}) {
  const [maxHeight, setMaxHeight] = useState<number>(chatPopover.maxHeight);
  const collisionPadding = keyboardOpen
    ? chatPopover.collisionPaddingCompact
    : chatPopover.collisionPadding;

  useLayoutEffect(() => {
    const targetComposer = composerEl ?? composerRef.current;
    if (!open || !targetComposer) {
      return;
    }

    let rafId = 0;
    const shellEl = targetComposer.closest("[data-chat-shell]");
    const topChromeEl =
      shellEl?.querySelector("[data-chat-top-chrome]") ?? shellEl?.querySelector("header") ?? null;
    const compactMedia = window.matchMedia("(max-width: 540px)");

    const updateMaxHeight = () => {
      const rect = targetComposer.getBoundingClientRect();
      const vv = window.visualViewport;
      const shellTop = shellEl ? shellEl.getBoundingClientRect().top : 0;
      // カレンダーストリップの上まで広げて使えるよう、上限基準は上部クローム（ヘッダー／再生バー）下端にする。
      const topChromeBottom = topChromeEl ? topChromeEl.getBoundingClientRect().bottom : shellTop;
      const vvTop = vv ? vv.offsetTop : 0;
      const topBoundary = Math.max(topChromeBottom, shellTop, vvTop);
      const available = Math.floor(
        rect.top - topBoundary - chatPopover.sideOffset - collisionPadding,
      );
      const maxCap = compactMedia.matches ? available : chatPopover.maxHeight;
      const next = Math.max(chatPopover.minHeight, Math.min(maxCap, available));
      setMaxHeight((prev) => (prev === next ? prev : next));
    };

    const scheduleUpdate = () => {
      updateMaxHeight();
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(updateMaxHeight);
    };

    updateMaxHeight();

    const vv = window.visualViewport;
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(scheduleUpdate) : null;
    ro?.observe(targetComposer);
    if (shellEl) {
      ro?.observe(shellEl);
    }
    if (topChromeEl) {
      ro?.observe(topChromeEl);
    }
    vv?.addEventListener("resize", scheduleUpdate);
    vv?.addEventListener("scroll", scheduleUpdate);
    window.addEventListener("resize", scheduleUpdate);

    return () => {
      cancelAnimationFrame(rafId);
      ro?.disconnect();
      vv?.removeEventListener("resize", scheduleUpdate);
      vv?.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [collisionPadding, composerEl, composerRef, open]);

  return maxHeight;
}
