import type { z } from "zod";
import type { SessionUser } from "../../auth/session.ts";
import type { CatalogSong } from "../../lib/apple-music/catalog.ts";
import type { Database } from "../../lib/db.ts";
import { createPublicId } from "../../lib/id.ts";
import { getOrCreateSong } from "../../songs/repository.ts";
import { ListeningLogOutputError } from "../errors.ts";
import { findListeningLogBySongAndDate, insertListeningLog } from "../repository.ts";
import { listeningLogOutput } from "../schemas.ts";
import type { createListeningLogInput } from "../schemas.ts";

export async function createListeningLog<FetchError extends Error>({
  db,
  fetchSong,
  user,
  input,
}: {
  db: Database;
  fetchSong: (songId: string) => Promise<FetchError | CatalogSong>;
  user: SessionUser;
  input: z.output<typeof createListeningLogInput>;
}) {
  const song = await getOrCreateSong({ db, songId: input.songId, isrc: input.isrc, fetchSong });
  if (song instanceof Error) {
    return song;
  }

  const existing = input.dedupe
    ? await findListeningLogBySongAndDate({
        db,
        userId: user.id,
        songIsrc: song.isrc,
        listenedOn: input.listenedOn,
      })
    : null;
  if (existing instanceof Error) {
    return existing;
  }

  const row =
    existing ??
    (await insertListeningLog({
      db,
      log: {
        id: createPublicId(),
        userId: user.id,
        songIsrc: song.isrc,
        listenedOn: input.listenedOn,
      },
    }));
  if (row instanceof Error) {
    return row;
  }

  const output = listeningLogOutput.safeParse({ ...row, song });
  if (!output.success) {
    return new ListeningLogOutputError({ logId: row.id, cause: output.error });
  }
  return output.data;
}
