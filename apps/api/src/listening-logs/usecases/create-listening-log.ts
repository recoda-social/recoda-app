import type { z } from "zod";
import type { SessionUser } from "../../auth/session.ts";
import type { CatalogSong } from "../../lib/apple-music/catalog.ts";
import type { Database } from "../../lib/db.ts";
import { createPublicId } from "../../lib/id.ts";
import { getOrCreateSong } from "../../songs/repository.ts";
import { ListeningLogOutputError } from "../errors.ts";
import { insertListeningLog } from "../repository.ts";
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

  const inserted = await insertListeningLog({
    db,
    log: {
      id: createPublicId(),
      userId: user.id,
      songIsrc: song.isrc,
      listenedOn: input.listenedOn,
    },
  });
  if (inserted instanceof Error) {
    return inserted;
  }

  const output = listeningLogOutput.safeParse({ ...inserted, song });
  if (!output.success) {
    return new ListeningLogOutputError({ logId: inserted.id, cause: output.error });
  }
  return output.data;
}
