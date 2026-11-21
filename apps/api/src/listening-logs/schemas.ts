import { ISRC_LENGTH, listeningLogs, songs } from "db";
import { createInsertSchema, createSelectSchema } from "drizzle-orm/zod";
import { z } from "zod";
import { songOutput } from "../songs/schemas.ts";

const DATE_KEY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

const dateKeySchema = z
  .string()
  .regex(DATE_KEY_PATTERN, "日付は YYYY-MM-DD 形式で指定してください")
  .refine((value) => {
    const [y, m, d] = value.split("-").map(Number);
    if (!y || !m || !d) {
      return false;
    }
    const date = new Date(Date.UTC(y, m - 1, d));
    return date.getUTCFullYear() === y && date.getUTCMonth() === m - 1 && date.getUTCDate() === d;
  }, "存在する日付を指定してください");

const songInsert = createInsertSchema(songs, {
  isrc: (schema) => schema.length(ISRC_LENGTH).toUpperCase(),
});

export const createListeningLogInput = z.object({
  songId: songInsert.shape.appleSongId,
  isrc: songInsert.shape.isrc,
  listenedOn: dateKeySchema,
});

export const listListeningLogsQuery = z
  .object({
    from: dateKeySchema,
    to: dateKeySchema,
  })
  .refine(({ from, to }) => from <= to, "from は to 以前の日付にしてください")
  .refine(({ from, to }) => {
    const diffDays =
      (Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / (1000 * 60 * 60 * 24);
    return diffDays <= 62;
  }, "取得範囲は最大 62 日間までです");

export const listeningLogOutput = createSelectSchema(listeningLogs)
  .pick({ id: true, listenedOn: true, createdAt: true })
  .extend({ song: songOutput });
