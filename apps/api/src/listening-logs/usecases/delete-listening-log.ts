import type { Database } from "../../lib/db.ts";
import { deleteListeningLogById } from "../repository.ts";

export async function deleteListeningLog({
  db,
  logId,
  userId,
}: {
  db: Database;
  logId: string;
  userId: string;
}) {
  return deleteListeningLogById({ db, logId, userId });
}
