import { listeningLogs, users } from "db";
import { beforeEach, describe, expect, test } from "vite-plus/test";
import type { SessionUser } from "../../../src/auth/session.ts";
import type { CatalogSong } from "../../../src/lib/apple-music/catalog.ts";
import type { Database } from "../../../src/lib/db.ts";
import { createListeningLog } from "../../../src/listening-logs/usecases/create-listening-log.ts";
import { createTestDb } from "../../helpers/db.ts";

const song: CatalogSong = {
  isrc: "JPP302300157",
  appleSongId: "1679278167",
  title: "アイドル",
  artistName: "YOASOBI",
  albumName: "アイドル - Single",
  durationMs: 213_000,
  artworkUrl: "https://example.com/{w}x{h}bb.jpg",
  url: "https://music.apple.com/jp/song/1679278167",
  artistIds: ["1490256993"],
};

const user: SessionUser = {
  id: "u1",
  name: "テスト",
  email: "a@example.com",
  emailVerified: false,
  createdAt: new Date(),
  updatedAt: new Date(),
  publicId: "p1",
  isAdmin: false,
};

let db: Database;

beforeEach(async () => {
  db = await createTestDb();
  await db
    .insert(users)
    .values({ id: "u1", name: "テスト", email: "a@example.com", publicId: "p1" });
});

describe("createListeningLog", () => {
  test("dedupe: true のとき、同じ日に同じ曲が既にあれば重複登録せず既存の記録を返す", async () => {
    const first = await createListeningLog({
      db,
      fetchSong: async () => song,
      user,
      input: { songId: song.appleSongId, isrc: song.isrc, listenedOn: "2026-11-24", dedupe: true },
    });
    if (first instanceof Error) {
      throw first;
    }

    const second = await createListeningLog({
      db,
      fetchSong: async () => song,
      user,
      input: { songId: song.appleSongId, isrc: song.isrc, listenedOn: "2026-11-24", dedupe: true },
    });
    if (second instanceof Error) {
      throw second;
    }

    expect(second.id).toBe(first.id);
    expect(await db.select().from(listeningLogs)).toHaveLength(1);
  });

  test("dedupe を指定しないときは、同じ日でも追加できる", async () => {
    const first = await createListeningLog({
      db,
      fetchSong: async () => song,
      user,
      input: { songId: song.appleSongId, isrc: song.isrc, listenedOn: "2026-11-24" },
    });
    const second = await createListeningLog({
      db,
      fetchSong: async () => song,
      user,
      input: { songId: song.appleSongId, isrc: song.isrc, listenedOn: "2026-11-24" },
    });

    if (first instanceof Error || second instanceof Error) {
      throw new Error("expected success");
    }
    expect(second.id).not.toBe(first.id);
    expect(await db.select().from(listeningLogs)).toHaveLength(2);
  });
});
