/**
 * オフライン学習した変換バイナリ（A: 256×2048 float32 + b: 256 float32）を
 * Turso の embedding_transforms テーブルへ登録するスクリプト。
 *
 * 実行例:
 *   TURSO_DATABASE_URL=... TURSO_AUTH_TOKEN=... \
 *   vpx apps/api/scripts/register-transform.ts \
 *     apps/api/scripts/disentangle-lab/out/plamo-1b-disentangle256-v1.bin \
 *     plamo-1b:disentangle256-v1
 */
import { readFile } from "node:fs/promises";
import { createClient } from "@libsql/client";
import { embeddingTransforms, relations } from "db";
import { EMBEDDING_DIMENSIONS, EMBEDDING_INPUT_DIMENSIONS } from "db/constants";
import { drizzle } from "drizzle-orm/libsql";
import { createTaggedError } from "errore";

class TransformBinaryError extends createTaggedError({
  name: "TransformBinaryError",
  message: "変換バイナリの読み込みまたは検証に失敗しました: $detail",
}) {}

const A_BYTES = EMBEDDING_DIMENSIONS * EMBEDDING_INPUT_DIMENSIONS * 4;
const B_BYTES = EMBEDDING_DIMENSIONS * 4;
const EXPECTED_BYTES = A_BYTES + B_BYTES;

function fail(error: Error): never {
  console.error(error);
  process.exit(1);
}

const [binPath, model = "plamo-1b:disentangle256-v1"] = process.argv.slice(2);
if (!binPath) {
  fail(
    new TransformBinaryError({
      detail: "引数に変換バイナリのパスを指定してください",
    }),
  );
}

const raw = await readFile(binPath).catch(
  (error: unknown) =>
    new TransformBinaryError({ detail: `${binPath} を読めませんでした`, cause: error }),
);
if (raw instanceof Error) {
  fail(raw);
}
if (raw.byteLength !== EXPECTED_BYTES) {
  fail(
    new TransformBinaryError({
      detail: `バイト数が不正です（期待=${EXPECTED_BYTES}, 実際=${raw.byteLength}）`,
    }),
  );
}

const aBytes = new Uint8Array(raw.buffer, raw.byteOffset, A_BYTES);
const bBytes = new Uint8Array(raw.buffer, raw.byteOffset + A_BYTES, B_BYTES);

const db = drizzle({
  client: createClient({
    url: process.env.TURSO_DATABASE_URL || "http://127.0.0.1:8080",
    authToken: process.env.TURSO_AUTH_TOKEN || undefined,
  }),
  relations,
});

const inserted = await db
  .insert(embeddingTransforms)
  .values({ model, a: aBytes, b: bBytes })
  .onConflictDoUpdate({
    target: embeddingTransforms.model,
    set: { a: aBytes, b: bBytes, createdAt: new Date() },
  })
  .catch(
    (error: unknown) =>
      new TransformBinaryError({ detail: "Turso への登録に失敗しました", cause: error }),
  );
if (inserted instanceof Error) {
  fail(inserted);
}

console.log(
  JSON.stringify({
    model,
    aBytes: aBytes.byteLength,
    bBytes: bBytes.byteLength,
    status: "registered",
  }),
);
