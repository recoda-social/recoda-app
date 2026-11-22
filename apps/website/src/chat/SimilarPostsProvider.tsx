import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { catalogSongsQuery } from "#/lib/musickit";
import type { AppleSong } from "#/lib/musickit";
import { similarPostsQuery } from "#/lib/posts";
import type { PostView } from "#/lib/posts";
import { useSearchingNotes } from "./SearchingNotes";
import { SimilarPostsContext } from "./SimilarPostsContext";

export function SimilarPostsProvider({
  postId,
  songTitle,
  children,
}: {
  postId: string;
  songTitle: string;
  children: ReactNode;
}) {
  const [includeSameSong, setIncludeSameSong] = useState(false);
  const query = useQuery(similarPostsQuery(postId, includeSameSong));
  const searchingNotes = useSearchingNotes();
  const setNotesActive = searchingNotes?.actions.setActive;

  useEffect(() => {
    setNotesActive?.(query.isPending);
  }, [query.isPending, setNotesActive]);

  const posts = useMemo(() => query.data?.posts ?? [], [query.data?.posts]);
  const songIds = useMemo(
    () => posts.map((post) => post.song?.appleSongId).filter((id): id is string => Boolean(id)),
    [posts],
  );
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

  return (
    <SimilarPostsContext
      value={{
        state: {
          posts,
          isPending: query.isPending,
          error: query.isError ? query.error.message : null,
          includeSameSong,
        },
        actions: {
          setIncludeSameSong,
          lookupCatalogSong,
        },
        meta: {
          postId,
          songTitle,
        },
      }}
    >
      {children}
    </SimilarPostsContext>
  );
}
