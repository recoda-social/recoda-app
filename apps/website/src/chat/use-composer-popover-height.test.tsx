// @vitest-environment jsdom
import "#/test/setup";
import { render, screen } from "@testing-library/react";
import { useRef, useState } from "react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vite-plus/test";
import { useComposerPopoverHeight } from "./use-composer-popover-height";

function PopoverHeightHarness({ children }: { children: ReactNode }) {
  const composerRef = useRef<HTMLDivElement | null>(null);
  const [composerEl, setComposerEl] = useState<HTMLDivElement | null>(null);
  const maxHeight = useComposerPopoverHeight({
    open: true,
    keyboardOpen: false,
    composerEl,
    composerRef,
  });

  return (
    <div
      data-chat-shell=""
      ref={(el) => {
        if (!el) {
          return;
        }
        el.getBoundingClientRect = () => new DOMRect(0, 0, 390, 844);
      }}
    >
      {children}
      <div
        data-testid="composer"
        ref={(el) => {
          composerRef.current = el;
          if (el) {
            el.getBoundingClientRect = () => new DOMRect(0, 360, 390, 120);
          }
          setComposerEl((prev) => (prev === el ? prev : el));
        }}
      />
      <output data-testid="max-height">{maxHeight}</output>
    </div>
  );
}

describe("useComposerPopoverHeight", () => {
  it("uses [data-chat-top-chrome] bottom even when header is hidden", () => {
    render(
      <PopoverHeightHarness>
        <div
          data-chat-top-chrome=""
          ref={(el) => {
            if (!el) {
              return;
            }
            el.getBoundingClientRect = () => new DOMRect(0, 0, 390, 96);
          }}
        />
      </PopoverHeightHarness>,
    );

    // 360 (composer top) - 96 (topChrome bottom) - 10 (sideOffset) - 16 (collisionPadding) = 238
    expect(screen.getByTestId("max-height").textContent).toBe("238");
  });

  it("falls back to <header> bottom when [data-chat-top-chrome] is absent", () => {
    render(
      <PopoverHeightHarness>
        <header
          ref={(el) => {
            if (!el) {
              return;
            }
            el.getBoundingClientRect = () => new DOMRect(0, 0, 390, 56);
          }}
        />
      </PopoverHeightHarness>,
    );

    // 360 (composer top) - 56 (header bottom) - 10 (sideOffset) - 16 (collisionPadding) = 278
    expect(screen.getByTestId("max-height").textContent).toBe("278");
  });
});
