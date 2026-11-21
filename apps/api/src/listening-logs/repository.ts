import { listeningLogs } from "db";
import type { NewListeningLog } from "db";
import { and, eq } from "drizzle-orm";
import type { Database } from "../lib/db.ts";
import {
  ListeningLogInsertError,
  ListeningLogNotFoundError,
  ListeningLogQueryError,
} from "./errors.ts";

export async function insertListeningLog({ db, log }: { db: Database; log: NewListeningLog }) {
  const inserted = await db
    .insert(listeningLogs)
    .values(log)
    .returning()
    .catch((error: unknown) => new ListeningLogInsertError({ cause: error }));
  if (inserted instanceof Error) {
    return inserted;
  }
  const row = inserted[0];
  if (!row) {
    return new ListeningLogInsertError();
  }
  return row;
}

export async function listListeningLogsByRange({
  db,
  userId,
  from,
  to,
}: {
  db: Database;
  userId: string;
  from: string;
  to: string;
}) {
  return db.query.listeningLogs
    .findMany({
      where: {
        userId,
        listenedOn: { gte: from, lte: to },
      },
      with: { song: true },
      orderBy: { listenedOn: "asc", createdAt: "asc" },
    })
    .catch((error: unknown) => new ListeningLogQueryError({ cause: error }));
}

export async function deleteListeningLogById({
  db,
  logId,
  userId,
}: {
  db: Database;
  logId: string;
  userId: string;
}) {
  const deleted = await db
    .delete(listeningLogs)
    .where(and(eq(listeningLogs.id, logId), eq(listeningLogs.userId, userId)))
    .returning({ id: listeningLogs.id })
    .catch((error: unknown) => new ListeningLogQueryError({ cause: error }));
  if (deleted instanceof Error) {
    return deleted;
  }
  if (deleted.length === 0) {
    return new ListeningLogNotFoundError({ logId });
  }
  return deleted[0];
}
