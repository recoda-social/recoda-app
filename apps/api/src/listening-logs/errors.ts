import { createTaggedError } from "errore";

export class ListeningLogInsertError extends createTaggedError({
  name: "ListeningLogInsertError",
  message: "聴いた曲の記録を保存できませんでした",
}) {}

export class ListeningLogQueryError extends createTaggedError({
  name: "ListeningLogQueryError",
  message: "聴いた曲の記録の読み書きに失敗しました",
}) {}

export class ListeningLogNotFoundError extends createTaggedError({
  name: "ListeningLogNotFoundError",
  message: "記録 $logId は見つかりません",
}) {}

export class ListeningLogOutputError extends createTaggedError({
  name: "ListeningLogOutputError",
  message: "記録 $logId を応答の形に変換できませんでした",
}) {}
