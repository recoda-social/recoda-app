import { fileURLToPath } from "node:url";
import type { PluginOption } from "vite-plus";
import { defineConfig, lazyPlugins, loadEnv } from "vite-plus";
import { playwright } from "vite-plus/test/browser-playwright";
import { devtools } from "@tanstack/devtools-vite";

import { tanstackRouter } from "@tanstack/router-plugin/vite";

import viteReact from "@vitejs/plugin-react";
import stylex from "@stylexjs/unplugin";

const apiEnvDir = fileURLToPath(new URL("../api", import.meta.url));

const config = defineConfig(({ mode }) => {
  const rawAllowedHost =
    process.env.DEV_ALLOWED_HOST ??
    loadEnv(mode, process.cwd(), "").DEV_ALLOWED_HOST ??
    loadEnv(mode, apiEnvDir, "").DEV_ALLOWED_HOST ??
    "";
  const devAllowedHosts = rawAllowedHost
    .split(",")
    .map((host) => host.trim())
    .filter(Boolean);

  return {
    resolve: { tsconfigPaths: true },
    plugins: lazyPlugins((): Array<PluginOption> => [
      ...(mode === "test" ? [] : [devtools()]),
      tanstackRouter({ target: "react", autoCodeSplitting: true }),
      // StyleX のドキュメントに従い、Fast Refresh を保つため @vitejs/plugin-react より前に置く。
      // useCSSLayers：StyleX が出力する CSS を @layer に入れ、後から書く CSS の詳細度と競合させない。
      // aliases：package.json の imports（#/*）を StyleX にも教える。StyleX は defineVars のファイルを自分で解決するため。
      // @stylexjs/unplugin の型定義が any を返すため PluginOption として扱う。
      // oxlint-disable-next-line typescript/no-unsafe-type-assertion
      stylex.vite({
        useCSSLayers: true,
        aliases: { "#/*": [fileURLToPath(new URL("./src/*", import.meta.url))] },
      }) as PluginOption,
      viteReact(),
    ]),
    test: {
      include: ["src/**/*.browser.test.{ts,tsx}"],
      setupFiles: ["./src/test/browser-setup.ts"],
      browser: {
        enabled: true,
        headless: true,
        provider: playwright({
          contextOptions: {
            hasTouch: true,
            isMobile: true,
          },
        }),
        instances: [{ browser: "webkit" }],
      },
    },
    server: {
      allowedHosts: devAllowedHosts.length > 0 ? devAllowedHosts : undefined,
      proxy: {
        // Host ヘッダーを書き換えない（localhost:3000 のまま API に渡す）。
        // API の Better Auth は、Host からアプリの URL を決めるため（apps/api/src/auth/auth.ts の baseURL）。
        "/api": { target: "http://localhost:8787", changeOrigin: false },
      },
    },
  };
});

export default config;
