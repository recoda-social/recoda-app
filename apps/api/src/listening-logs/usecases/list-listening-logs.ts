import type { Database } from "../../lib/db.ts";
import { ListeningLogOutputError } from "../errors.ts";
import { listListeningLogsByRange } from "../repository.ts";
import { listeningLogOutput } from "../schemas.ts";

export async function listListeningLogs({
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
  const rows = await listListeningLogsByRange({ db, userId, from, to });
  if (rows instanceof Error) {
    return rows;
  }

  const logs = [];
  for (const row of rows) {
    const parsed = listeningLogOutput.safeParse(row);
    if (!parsed.success) {
      return new ListeningLogOutputError({ logId: row.id, cause: parsed.error });
    }
    logs.push(parsed.data);
  }
  return { logs };
}
