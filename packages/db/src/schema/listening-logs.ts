import { sql } from "drizzle-orm";
import { check, index, snakeCase, text } from "drizzle-orm/sqlite-core";
import { users } from "./auth.ts";
import { createdAt } from "./columns.ts";
import { songs } from "./music.ts";

export const listeningLogs = snakeCase.table(
  "listening_logs",
  {
    id: text().primaryKey(),
    userId: text()
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    songIsrc: text()
      .notNull()
      .references(() => songs.isrc, { onDelete: "restrict" }),
    listenedOn: text().notNull(),
    createdAt: createdAt(),
  },
  (t) => [
    index("listening_logs_user_date_idx").on(t.userId, t.listenedOn, t.createdAt),
    check(
      "listening_logs_listened_on_format",
      sql`length(${t.listenedOn}) = 10 and ${t.listenedOn} glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'`,
    ),
  ],
);

export type ListeningLog = typeof listeningLogs.$inferSelect;
export type NewListeningLog = typeof listeningLogs.$inferInsert;
