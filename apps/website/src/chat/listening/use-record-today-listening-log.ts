import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import { createListeningLog } from "#/lib/listening-logs";
import { getTodayDateKey } from "./calendar-utils";

/** 曲を「今日聴いた曲」に重複なしで記録する関数を返す。 */
export function useRecordTodayListeningLog() {
  const queryClient = useQueryClient();
  const { mutate } = useMutation({
    mutationFn: createListeningLog,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["listening-logs"] });
    },
  });

  return useCallback(
    (song: { songId: string; isrc: string }) => {
      mutate({
        songId: song.songId,
        isrc: song.isrc,
        listenedOn: getTodayDateKey(),
        dedupe: true,
      });
    },
    [mutate],
  );
}
