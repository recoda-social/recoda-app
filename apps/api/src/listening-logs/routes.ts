import { zValidator } from "@hono/zod-validator";
import { matchErrorPartial } from "errore";
import { Hono } from "hono";
import { requireAuth } from "../auth/session.ts";
import type { SessionEnv } from "../auth/session.ts";
import { fetchSong } from "../lib/apple-music/client.ts";
import { db } from "../lib/db.ts";
import { logError } from "../lib/log.ts";
import { rateLimit } from "../lib/rate-limit/middleware.ts";
import { ListeningLogNotFoundError } from "./errors.ts";
import { createListeningLogInput, listListeningLogsQuery } from "./schemas.ts";
import { createListeningLog } from "./usecases/create-listening-log.ts";
import { deleteListeningLog } from "./usecases/delete-listening-log.ts";
import { listListeningLogs } from "./usecases/list-listening-logs.ts";

const INTERNAL_ERROR = "処理に失敗しました。時間をおいて、もう一度試してください";

export const listeningLogsRoutes = new Hono<SessionEnv>()
  .get(
    "/",
    requireAuth,
    rateLimit("READ_RATE_LIMIT"),
    zValidator("query", listListeningLogsQuery, (result, c) => {
      if (!result.success) {
        return c.json({ error: "日付の範囲が正しくありません", issues: result.error.issues }, 400);
      }
    }),
    async (c) => {
      const { from, to } = c.req.valid("query");
      const result = await listListeningLogs({
        db,
        userId: c.get("user").id,
        from,
        to,
      });
      if (result instanceof Error) {
        logError(result);
        return c.json({ error: INTERNAL_ERROR }, 500);
      }
      return c.json(result);
    },
  )
  .post(
    "/",
    requireAuth,
    rateLimit("WRITE_RATE_LIMIT"),
    zValidator("json", createListeningLogInput, (result, c) => {
      if (!result.success) {
        return c.json(
          {
            error: "入力が正しくありません。曲と日付を確認してください",
            issues: result.error.issues,
          },
          400,
        );
      }
    }),
    async (c) => {
      const created = await createListeningLog({
        db,
        fetchSong,
        user: c.get("user"),
        input: c.req.valid("json"),
      });
      if (created instanceof Error) {
        const { status, error } = matchErrorPartial<
          typeof created,
          { status: 422 | 500 | 503; error: string }
        >(
          created,
          {
            AppleSongNotFoundError: (e) => ({ status: 422, error: e.message }),
            SongWithoutIsrcError: (e) => ({ status: 422, error: e.message }),
            AppleMusicRateLimitError: (e) => ({ status: 503, error: e.message }),
          },
          (e) => {
            logError(e);
            return {
              status: 500,
              error: "記録を保存できませんでした。時間をおいて、もう一度試してください",
            };
          },
        );
        return c.json({ error }, status);
      }
      return c.json(created, 201);
    },
  )
  .delete("/:id", requireAuth, rateLimit("WRITE_RATE_LIMIT"), async (c) => {
    const deleted = await deleteListeningLog({
      db,
      logId: c.req.param("id"),
      userId: c.get("user").id,
    });
    if (deleted instanceof ListeningLogNotFoundError) {
      return c.json({ error: deleted.message }, 404);
    }
    if (deleted instanceof Error) {
      logError(deleted);
      return c.json({ error: INTERNAL_ERROR }, 500);
    }
    return c.body(null, 204);
  });
