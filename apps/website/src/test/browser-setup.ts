import "@ui-sdk/core/web.css";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vite-plus/test";

const BASE_RESET_ATTR = "data-recoda-browser-reset";

if (!document.head.querySelector(`style[${BASE_RESET_ATTR}]`)) {
  const style = document.createElement("style");
  style.setAttribute(BASE_RESET_ATTR, "");
  style.textContent = `
    @layer base-reset, ui-sdk;

    @layer base-reset {
      *,
      *::before,
      *::after {
        box-sizing: border-box;
      }

      html,
      body,
      #app {
        min-height: 100%;
      }

      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
    }
  `;
  document.head.prepend(style);
}

afterEach(cleanup);
