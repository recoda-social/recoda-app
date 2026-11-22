import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vite-plus/test";

vi.mock("@stylexjs/stylex", () => ({
  create: <T extends Record<string, Record<string, string | number | null | object>>>(styles: T) =>
    styles,
  defineVars: <T extends Record<string, string | number | object>>(vars: T) => vars,
  defineConsts: <T extends Record<string, string | number>>(consts: T) => consts,
  keyframes: () => "mock-keyframes",
  props: () => ({}),
}));

declare global {
  var __UI_SDK_STRICT__: boolean | undefined;
}
globalThis.__UI_SDK_STRICT__ = true;

if (typeof HTMLElement !== "undefined") {
  HTMLElement.prototype.showPopover = function () {};
  HTMLElement.prototype.hidePopover = function () {};
  Element.prototype.getAnimations = () => [];
  Element.prototype.scrollIntoView = function () {};
  Element.prototype.checkVisibility = () => true;
  const emptyRects: Array<DOMRect> = [];
  const emptyRectList: DOMRectList = {
    length: 0,
    item: () => null,
    [Symbol.iterator]: () => emptyRects[Symbol.iterator](),
  };
  Element.prototype.getClientRects = () => emptyRectList;
  document.elementFromPoint = () => null;
  document.elementsFromPoint = () => [];

  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  if (typeof window.matchMedia !== "function") {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener() {},
        removeListener() {},
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent: () => false,
      }),
    });
  }

  document.head.append(
    Object.assign(document.createElement("style"), {
      textContent: "[popover] { display: block !important }",
    }),
  );
}

afterEach(cleanup);
