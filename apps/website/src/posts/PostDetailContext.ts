import { createContext, use } from "react";
import type { AppleSong } from "#/lib/musickit";
import type { PostView } from "#/lib/posts";

type PostDetailState = {
  post: PostView | null;
  isPostPending: boolean;
  postError: string | null;
  similarPosts: Array<PostView>;
  isSimilarPending: boolean;
  similarError: string | null;
  similarExpired: boolean;
  similarExpiresAt: string | null;
  includeSameSong: boolean;
};

type PostDetailActions = {
  setIncludeSameSong: (include: boolean) => void;
  lookupCatalogSong: (post: PostView) => AppleSong | undefined;
  recordPlayedSong: (song: { songId: string; isrc: string }) => void;
};

type PostDetailMeta = {
  postId: string;
};

export type PostDetailContextValue = {
  state: PostDetailState;
  actions: PostDetailActions;
  meta: PostDetailMeta;
};

export const PostDetailContext = createContext<PostDetailContextValue | null>(null);

export function usePostDetail() {
  const value = use(PostDetailContext);
  if (!value) {
    throw new Error("usePostDetail は PostDetailProvider の中で使う");
  }
  return value;
}
