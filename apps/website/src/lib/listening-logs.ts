import { queryOptions } from "@tanstack/react-query";
import type { InferResponseType } from "hono/client";
import { client } from "#/lib/api";

export type ListeningLogsResponse = InferResponseType<
  (typeof client.api)["listening-logs"]["$get"],
  200
>;

export type ListeningLogView = ListeningLogsResponse["logs"][number];

export const listeningLogsQuery = (from: string, to: string) =>
  queryOptions({
    queryKey: ["listening-logs", from, to],
    queryFn: async () => {
      const res = await client.api["listening-logs"].$get({ query: { from, to } });
      const body = await res.json();
      if ("error" in body) {
        throw new Error(body.error);
      }
      return body;
    },
  });

export async function createListeningLog(input: {
  songId: string;
  isrc: string;
  listenedOn: string;
  dedupe?: boolean;
}) {
  const res = await client.api["listening-logs"].$post({ json: input });
  const body = await res.json();
  if ("error" in body) {
    throw new Error(body.error);
  }
  return body;
}

export async function deleteListeningLog(logId: string) {
  const res = await client.api["listening-logs"][":id"].$delete({ param: { id: logId } });
  if (res.status === 204) {
    return;
  }
  const body = await res.json();
  if ("error" in body) {
    throw new Error(body.error);
  }
}
