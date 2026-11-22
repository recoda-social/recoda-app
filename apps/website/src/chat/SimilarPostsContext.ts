import { createContext, use } from "react";
import type { AppleSong } from "#/lib/musickit";
import type { PostView } from "#/lib/posts";

type SimilarPostsState = {
  posts: Array<PostView>;
  isPending: boolean;
  error: string | null;
  includeSameSong: boolean;
};

type SimilarPostsActions = {
  setIncludeSameSong: (include: boolean) => void;
  lookupCatalogSong: (post: PostView) => AppleSong | undefined;
};

type SimilarPostsMeta = {
  postId: string;
  songTitle: string;
};

export type SimilarPostsContextValue = {
  state: SimilarPostsState;
  actions: SimilarPostsActions;
  meta: SimilarPostsMeta;
};

export const SimilarPostsContext = createContext<SimilarPostsContextValue | null>(null);

export function useSimilarPosts() {
  const value = use(SimilarPostsContext);
  if (!value) {
    throw new Error("useSimilarPosts は SimilarPostsProvider の中で使う");
  }
  return value;
}
