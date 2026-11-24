import "#/test/browser-setup";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vite-plus/test";
import { page } from "vite-plus/test/browser";
import { AuthContext } from "#/auth/AuthContext";
import type { Me } from "#/lib/auth";
import type { AppleSong } from "#/lib/musickit";
import { PreviewPlayerProvider } from "#/songs/preview-player";
import { ChatWorkspace } from "./ChatWorkspace";

const sampleSong: AppleSong = {
  id: "song-1",
  attributes: {
    name: "群青",
    artistName: "YOASOBI",
    isrc: "JPXX02000001",
    previews: [{ url: "https://example.com/preview.m4a" }],
  },
};

vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, to, ...rest }: { children: ReactNode; to?: string }) => (
    <a href={to ?? "#"} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock("#/lib/musickit", () => ({
  artworkUrl: (template: string, size: number) =>
    template.replace("{w}", String(size)).replace("{h}", String(size)),
  catalogSongsQuery: (ids: Array<string>) => ({
    queryKey: ["musickit", "songs", ids],
    queryFn: async () => [],
    enabled: false,
  }),
  findSongIdByIsrc: vi.fn(async () => null),
  getLoadedMusicKit: vi.fn(() => null),
  getMusicKit: vi.fn(() => new Promise(() => {})),
  searchSongs: vi.fn(async () => [sampleSong]),
}));

vi.mock("#/lib/listening-logs", () => ({
  listeningLogsQuery: (from: string, to: string) => ({
    queryKey: ["listening-logs", from, to],
    queryFn: async () => ({ logs: [] }),
  }),
  createListeningLog: vi.fn(async () => ({ id: "log-1" })),
  deleteListeningLog: vi.fn(async () => undefined),
}));

vi.mock("#/lib/profile", () => ({
  authorPostsInfiniteQuery: (publicId: string) => ({
    queryKey: ["users", "posts", publicId],
    initialPageParam: null as string | null,
    queryFn: async () => ({ posts: [], nextCursor: null }),
    getNextPageParam: () => undefined,
  }),
}));

vi.mock("#/lib/posts", () => ({
  createPost: vi.fn(async (input: { songId: string; isrc: string; content: string }) => ({
    id: "post-1",
    content: input.content,
    song: {
      appleSongId: input.songId,
      isrc: input.isrc,
      title: sampleSong.attributes.name,
      artistName: sampleSong.attributes.artistName,
    },
  })),
  similarPostsQuery: (postId: string, includeSameSong: boolean) => ({
    queryKey: ["posts", "similar", postId, includeSameSong],
    queryFn: async () => ({ posts: [] }),
  }),
}));

vi.mock("#/lib/draft-assist", () => ({
  requestDraftAssist: vi.fn(async () => null),
  sendDraftAssistEvent: vi.fn(),
}));

const testUser: Me = {
  publicId: "pub-1",
  name: "テストユーザー",
  email: "test@example.com",
  image: null,
  bio: null,
  isAdmin: false,
};

const mountedContainers: Array<HTMLElement> = [];

afterEach(() => {
  while (mountedContainers.length > 0) {
    mountedContainers.pop()?.remove();
  }
});

function renderIntoApp() {
  Object.defineProperty(HTMLMediaElement.prototype, "play", {
    configurable: true,
    value: vi.fn(() => Promise.resolve()),
  });
  Object.defineProperty(HTMLMediaElement.prototype, "pause", {
    configurable: true,
    value: vi.fn(),
  });

  const appRoot = document.createElement("div");
  appRoot.id = "app";
  document.body.append(appRoot);
  mountedContainers.push(appRoot);

  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <PreviewPlayerProvider>
        <AuthContext
          value={{
            state: { user: testUser, isPending: false },
            actions: {
              signInWithGoogle: vi.fn(),
              signOut: vi.fn(async () => {}),
            },
            meta: {},
          }}
        >
          <ChatWorkspace user={testUser} />
        </AuthContext>
      </PreviewPlayerProvider>
    </QueryClientProvider>,
    { container: appRoot },
  );

  return appRoot;
}

function getAncestors(el: Element): Array<Element> {
  const list: Array<Element> = [];
  let current: Element | null = el.parentElement;
  while (current) {
    list.push(current);
    current = current.parentElement;
  }
  return list;
}

function getBackdropFilter(style: CSSStyleDeclaration): string {
  const webkit = style.getPropertyValue("-webkit-backdrop-filter");
  if (webkit && webkit !== "none") {
    return webkit;
  }
  return style.backdropFilter || "none";
}

async function selectSampleSong() {
  const textarea = screen.getByLabelText("曲への感想を入力");
  fireEvent.input(textarea, { target: { value: "@群青" } });

  const picker = await screen.findByLabelText("曲の検索候補");
  const songItem = await within(picker).findByText("群青", {}, { timeout: 3000 });
  const button = songItem.closest("button");
  expect(button).not.toBeNull();
  if (!button) {
    throw new Error("曲ボタンが見つかりません");
  }
  fireEvent.click(button);
}

describe("ChatWorkspace WebKit layout and compositing invariants", () => {
  it("preserves single upper compositing layer, touch font-size, and normal-flow composer without ancestor transforms on inputs", async () => {
    await page.viewport(390, 664);
    const container = renderIntoApp();

    const shell = container.querySelector<HTMLElement>("[data-chat-shell]");
    const header = container.querySelector<HTMLElement>("header");
    const topChromeBar = container.querySelector<HTMLElement>("[data-chat-top-chrome]");
    const calendarStrip = container.querySelector<HTMLElement>(
      'section[aria-label="聴いた曲カレンダー"]',
    );
    const main = container.querySelector<HTMLElement>("main");
    const composerSurface = container.querySelector<HTMLElement>("[data-chat-composer]");
    const composerDock = composerSurface?.parentElement;
    const modeSwitch = container.querySelector<HTMLElement>('[role="radiogroup"]');
    const dockHeader = modeSwitch?.parentElement;
    const textarea = screen.getByLabelText("曲への感想を入力");

    expect(shell).not.toBeNull();
    expect(header).not.toBeNull();
    expect(topChromeBar).not.toBeNull();
    expect(calendarStrip).not.toBeNull();
    expect(main).not.toBeNull();
    expect(composerSurface).not.toBeNull();
    expect(composerDock).not.toBeNull();
    expect(dockHeader).not.toBeNull();

    if (
      !shell ||
      !header ||
      !topChromeBar ||
      !calendarStrip ||
      !main ||
      !composerSurface ||
      !composerDock ||
      !dockHeader
    ) {
      return;
    }

    // ADR 0023: html / body / shell は position: fixed にしない
    expect(getComputedStyle(document.documentElement).position).not.toBe("fixed");
    expect(getComputedStyle(document.body).position).not.toBe("fixed");
    const shellStyle = getComputedStyle(shell);
    expect(shellStyle.position).toBe("relative");
    expect(shellStyle.overflow).toBe("clip");
    expect(Math.round(shell.getBoundingClientRect().height)).toBe(664);

    // #115: ComposerDock は通常フロー（position: relative; flexShrink: 0）で、
    // 上部のモードスイッチ背面のみ <main> と重ねて easing gradient でフェードする
    const dockStyle = getComputedStyle(composerDock);
    expect(dockStyle.position).toBe("relative");
    expect(dockStyle.flexShrink).toBe("0");
    expect(Number.parseFloat(dockStyle.marginTop)).toBeLessThan(0);
    expect(dockStyle.backgroundImage).toContain("linear-gradient");
    expect(dockStyle.pointerEvents).toBe("none");
    expect(getComputedStyle(composerSurface).pointerEvents).toBe("auto");
    expect(getComputedStyle(composerSurface).boxShadow).toBe("none");
    if (modeSwitch) {
      expect(getComputedStyle(modeSwitch).pointerEvents).toBe("auto");
    }

    // 入力欄周りの背景色と iOS Safari のアドレスバー・キーボード背後の背景色を一致させる
    expect(getComputedStyle(document.documentElement).backgroundColor).toBe(
      shellStyle.backgroundColor,
    );
    expect(getComputedStyle(document.body).backgroundColor).toBe(shellStyle.backgroundColor);
    expect(getComputedStyle(container).backgroundColor).toBe(shellStyle.backgroundColor);

    // ADR 0024: タッチ端末では自動ズーム防止のため入力文字サイズ >= 16px
    expect(Number.parseFloat(getComputedStyle(textarea).fontSize)).toBeGreaterThanOrEqual(16);

    // ADR 0024: <textarea> のどの祖先にも transform が付かないこと
    for (const ancestor of getAncestors(textarea)) {
      expect(getComputedStyle(ancestor).transform).toBe("none");
    }

    // #114 / ADR 0025: Header・Calendar・Main が単一の upperStage 合成レイヤーを共有し、
    // その内側では個別の transform / backdrop-filter レイヤーを作らない
    const upperStage = main.parentElement;
    expect(upperStage).not.toBeNull();
    if (!upperStage) {
      return;
    }
    expect(getComputedStyle(upperStage).transform).not.toBe("none");
    expect(upperStage.contains(header)).toBe(true);
    expect(upperStage.contains(topChromeBar)).toBe(true);
    expect(upperStage.contains(calendarStrip)).toBe(true);
    expect(upperStage.contains(composerDock)).toBe(false);

    for (const innerChrome of [topChromeBar.parentElement, topChromeBar, header, calendarStrip]) {
      expect(innerChrome).not.toBeNull();
      if (innerChrome && innerChrome !== upperStage) {
        expect(getComputedStyle(innerChrome).transform).toBe("none");
        expect(getBackdropFilter(getComputedStyle(innerChrome))).toBe("none");
      }
    }

    // ADR 0025: モードスイッチ行（ComposerDock.Header）は入力欄を含まず単独の合成レイヤーを持つ
    expect(getComputedStyle(dockHeader).transform).not.toBe("none");
    expect(dockHeader.contains(textarea)).toBe(false);
  });

  it("hides Topbar only while writing with keyboard open, keeps MiniPlayerBar unshrunk, shrinks calendar on multi-line input, and keeps SongPicker below MiniPlayerBar", async () => {
    await page.viewport(390, 664);
    const container = renderIntoApp();

    const textarea = screen.getByLabelText("曲への感想を入力");
    const calendarStrip = container.querySelector<HTMLElement>(
      'section[aria-label="聴いた曲カレンダー"]',
    );
    const composerDock =
      container.querySelector<HTMLElement>("[data-chat-composer]")?.parentElement;

    expect(calendarStrip).not.toBeNull();
    expect(composerDock).not.toBeNull();
    if (!calendarStrip || !composerDock) {
      return;
    }

    const fullCalendarHeight = calendarStrip.getBoundingClientRect().height;
    expect(fullCalendarHeight).toBeGreaterThan(70);

    // 1. フォーカスしてビューポートを縮める（仮想キーボード表示を再現）
    textarea.focus();
    await page.viewport(390, 360);

    // 曲未選択（検索中）のあいだは Topbar が残る
    await waitFor(() => {
      expect(container.querySelector("header")).not.toBeNull();
    });

    // 2. 曲を選んで試聴再生を開始する → Topbar は非表示になり、MiniPlayerBar は見えたまま
    await selectSampleSong();
    await waitFor(() => {
      expect(container.querySelector("header")).toBeNull();
    });

    fireEvent.click(screen.getByRole("button", { name: "群青 を試聴" }));
    const miniPlayer = container.querySelector<HTMLElement>("[data-mini-player]");
    expect(miniPlayer).not.toBeNull();
    if (!miniPlayer) {
      return;
    }

    // MiniPlayerBar の @starting-style 進入トランジション（180ms, translateY(-6px) -> 0）完了を待つ
    await waitFor(() => {
      const rect = miniPlayer.getBoundingClientRect();
      expect(rect.top).toBeGreaterThanOrEqual(0);
      expect(rect.height).toBeGreaterThan(40);
    });
    const miniPlayerRect = miniPlayer.getBoundingClientRect();
    expect(getBackdropFilter(getComputedStyle(miniPlayer))).toBe("none");

    // 3. 1 行の状態で「曲を選ぶ」を開くと、検索ポップオーバーが MiniPlayerBar の下端に重ならない
    fireEvent.click(screen.getByRole("button", { name: "曲を検索して添付" }));
    const picker = await screen.findByLabelText("曲の検索候補");
    await waitFor(() => {
      const pickerRect = picker.getBoundingClientRect();
      expect(pickerRect.top).toBeGreaterThanOrEqual(miniPlayer.getBoundingClientRect().bottom - 1);
    });
    // ポップオーバーを閉じる
    fireEvent.keyDown(textarea, { key: "Escape" });

    // 4. 複数行入力すると、通常フローのまま入力欄は画面内に収まり、MiniPlayerBar は縮まず、カレンダーだけが縮む
    fireEvent.input(textarea, {
      target: { value: "1行目\n2行目\n3行目\n4行目\n5行目\n6行目" },
    });

    await waitFor(() => {
      expect(calendarStrip.getBoundingClientRect().height).toBeLessThan(fullCalendarHeight - 10);
    });

    expect(composerDock.getBoundingClientRect().bottom).toBeLessThanOrEqual(361);
    expect(
      Math.abs(miniPlayer.getBoundingClientRect().height - miniPlayerRect.height),
    ).toBeLessThan(1);
    expect(getComputedStyle(calendarStrip).overflow).toBe("clip");

    // 5. 曲選択中でもキーボードを閉じる（blur + viewport 復帰）と Topbar とカレンダーが復帰する
    textarea.blur();
    await page.viewport(390, 664);

    await waitFor(() => {
      expect(container.querySelector("header")).not.toBeNull();
      expect(calendarStrip.getBoundingClientRect().height).toBeGreaterThanOrEqual(
        fullCalendarHeight - 1,
      );
    });
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();
  });
});
