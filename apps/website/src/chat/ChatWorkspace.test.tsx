// @vitest-environment jsdom
import "#/test/setup";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vite-plus/test";
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

class FakeVisualViewport extends EventTarget {
  width = 390;
  height = 768;
  offsetTop = 0;
  scale = 1;

  setKeyboardOpen(open: boolean) {
    this.height = open ? 420 : 768;
    this.dispatchEvent(new Event("resize"));
  }
}

function renderWorkspace() {
  window.scrollTo = vi.fn();
  const fakeVv = new FakeVisualViewport();
  Object.defineProperty(window, "innerHeight", {
    configurable: true,
    value: 768,
  });
  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: fakeVv,
  });
  Object.defineProperty(HTMLMediaElement.prototype, "play", {
    configurable: true,
    value: vi.fn(() => Promise.resolve()),
  });
  Object.defineProperty(HTMLMediaElement.prototype, "pause", {
    configurable: true,
    value: vi.fn(),
  });

  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const result = render(
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
  );

  return { ...result, fakeVv };
}

async function selectSampleSong() {
  const textarea = screen.getByLabelText("曲への感想を入力（@ または / で曲を検索）");
  fireEvent.input(textarea, { target: { value: "@群青" } });

  const picker = await screen.findByLabelText("曲の検索候補");
  const songItem = await within(picker).findByText("群青", {}, { timeout: 2000 });
  const button = songItem.closest("button");
  expect(button).not.toBeNull();
  if (!button) {
    throw new Error("曲ボタンが見つかりません");
  }
  fireEvent.click(button);
}

describe("ChatWorkspace header visibility during review writing", () => {
  it("hides Topbar only while keyboard is open with a song selected in review mode, keeping MiniPlayerBar visible and restoring Topbar on keyboard close, mode switch, clear, and submit", async () => {
    const { container, fakeVv } = renderWorkspace();

    // 1. 初期状態（キーボード閉・曲未選択）: ヘッダーが表示されている
    expect(container.querySelector("header")).not.toBeNull();
    expect(container.querySelector("[data-chat-top-chrome]")).not.toBeNull();

    // 2. 曲未選択のままキーボードが開いても（検索中）ヘッダーは表示されたまま
    act(() => fakeVv.setKeyboardOpen(true));
    expect(container.querySelector("header")).not.toBeNull();

    // 3. キーボード表示中に曲を選ぶとヘッダーが非表示になり、試聴再生バーは見える
    await selectSampleSong();
    expect(container.querySelector("header")).toBeNull();
    expect(container.querySelector("[data-chat-top-chrome]")).not.toBeNull();

    const previewBtn = screen.getByRole("button", { name: "群青 を試聴" });
    fireEvent.click(previewBtn);
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();

    // 4. 曲を選択したままキーボードを閉じるとヘッダーが復帰する
    act(() => fakeVv.setKeyboardOpen(false));
    expect(container.querySelector("header")).not.toBeNull();
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();

    // 5. 曲選択中のまま再びキーボードを開くとヘッダーが非表示になる
    act(() => fakeVv.setKeyboardOpen(true));
    expect(container.querySelector("header")).toBeNull();
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();

    // 6. 「聴いた曲を記録」モードへ切り替えるとヘッダーが復帰し、「感想を残す」へ戻すと再び隠れる
    fireEvent.click(screen.getByRole("radio", { name: "聴いた曲を記録" }));
    expect(container.querySelector("header")).not.toBeNull();
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();

    fireEvent.click(screen.getByRole("radio", { name: "感想を残す" }));
    expect(container.querySelector("header")).toBeNull();
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();

    // 7. 曲選択を解除するとキーボードが開いたままでもヘッダーが復帰する
    fireEvent.click(screen.getByRole("button", { name: "群青 の選択を解除" }));
    expect(container.querySelector("header")).not.toBeNull();
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();

    // 8. 再度曲を選んで感想を送信するとヘッダーが復帰する
    await selectSampleSong();
    expect(container.querySelector("header")).toBeNull();

    const textarea = screen.getByLabelText("曲への感想を入力（@ または / で曲を検索）");
    fireEvent.input(textarea, { target: { value: "サビの広がりがとても気持ちいい" } });
    fireEvent.click(screen.getByRole("button", { name: "感想を送信" }));

    await waitFor(() => {
      expect(container.querySelector("header")).not.toBeNull();
    });
    expect(container.querySelector("[data-mini-player]")).not.toBeNull();
  });
});
