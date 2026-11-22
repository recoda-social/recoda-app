import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { catalogSongsQuery } from "#/lib/musickit";
import type { AppleSong } from "#/lib/musickit";
import { postQuery, similarPostsQuery } from "#/lib/posts";
import type { PostView } from "#/lib/posts";
import { usePreviewPlayer } from "#/songs/preview-player";
import { PostDetailContext } from "./PostDetailContext";

export function PostDetailProvider({ postId, children }: { postId: string; children: ReactNode }) {
  const [includeSameSong, setIncludeSameSong] = useState(false);

  const postResult = useQuery(postQuery(postId));
  const similarResult = useQuery({
    ...similarPostsQuery(postId, includeSameSong),
    enabled: postResult.isSuccess,
  });

  const similarPosts = useMemo(() => similarResult.data?.posts ?? [], [similarResult.data?.posts]);
  const postAppleSongId = postResult.data?.song?.appleSongId;
  const songIds = useMemo(() => {
    const ids: Array<string> = [];
    if (postAppleSongId) {
      ids.push(postAppleSongId);
    }
    for (const post of similarPosts) {
      if (post.song?.appleSongId) {
        ids.push(post.song.appleSongId);
      }
    }
    return ids;
  }, [postAppleSongId, similarPosts]);

  const catalogQuery = useQuery(catalogSongsQuery(songIds));
  const catalogSongs = useMemo(() => {
    const byId = new Map<string, AppleSong>();
    const byIsrc = new Map<string, AppleSong>();
    for (const song of catalogQuery.data ?? []) {
      byId.set(song.id, song);
      if (song.attributes.isrc) {
        byIsrc.set(song.attributes.isrc.toUpperCase(), song);
      }
    }
    return { byId, byIsrc };
  }, [catalogQuery.data]);

  const lookupCatalogSong = useCallback(
    (post: PostView) =>
      post.song
        ? (catalogSongs.byId.get(post.song.appleSongId) ??
          catalogSongs.byIsrc.get(post.song.isrc.toUpperCase()))
        : undefined,
    [catalogSongs],
  );

  const { preloadAppleMusic } = usePreviewPlayer().actions;
  useEffect(() => {
    preloadAppleMusic();
  }, [preloadAppleMusic]);

  const similarData = similarResult.data;

  return (
    <PostDetailContext
      value={{
        state: {
          post: postResult.data ?? null,
          isPostPending: postResult.isPending,
          postError: postResult.isError ? postResult.error.message : null,
          similarPosts,
          isSimilarPending: similarResult.isPending,
          similarError: similarResult.isError ? similarResult.error.message : null,
          similarExpired: Boolean(similarData?.expired),
          similarExpiresAt: similarData && !similarData.expired ? similarData.expiresAt : null,
          includeSameSong,
        },
        actions: {
          setIncludeSameSong,
          lookupCatalogSong,
        },
        meta: {
          postId,
        },
      }}
    >
      {children}
    </PostDetailContext>
  );
}
