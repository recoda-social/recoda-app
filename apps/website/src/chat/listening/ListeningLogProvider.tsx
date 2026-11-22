import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { createListeningLog, deleteListeningLog, listeningLogsQuery } from "#/lib/listening-logs";
import type { ListeningLogView } from "#/lib/listening-logs";
import { searchSongs } from "#/lib/musickit";
import type { AppleSong } from "#/lib/musickit";
import { useChat } from "../ChatContext";
import {
  getQueryRangeForMonth,
  getTodayDateKey,
  parseDateKey,
  shiftDateKey,
} from "./calendar-utils";
import { ListeningLogContext } from "./ListeningLogContext";
import type { ComposerMode } from "./ListeningLogContext";

export function ListeningLogProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const chat = useChat();
  const [todayKey] = useState(() => getTodayDateKey());
  const initialParts = parseDateKey(todayKey);

  const [composerMode, setComposerMode] = useState<ComposerMode>("review");
  const [selectedDate, setSelectedDate] = useState(todayKey);
  const [visibleYear, setVisibleYear] = useState(initialParts.year);
  const [visibleMonthIndex, setVisibleMonthIndex] = useState(initialParts.monthIndex);
  const [calendarExpanded, setCalendarExpanded] = useState(false);
  const [shelfExpanded, setShelfExpanded] = useState(false);

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Array<AppleSong>>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const registerInput = useCallback((el: HTMLInputElement | null) => {
    inputRef.current = el;
  }, []);

  const range = useMemo(
    () => getQueryRangeForMonth(visibleYear, visibleMonthIndex),
    [visibleYear, visibleMonthIndex],
  );

  const query = useQuery(listeningLogsQuery(range.from, range.to));

  const logsByDate = useMemo(() => {
    const map = new Map<string, Array<ListeningLogView>>();
    for (const log of query.data?.logs ?? []) {
      const list = map.get(log.listenedOn);
      if (list) {
        list.push(log);
      } else {
        map.set(log.listenedOn, [log]);
      }
    }
    return map;
  }, [query.data?.logs]);

  const selectedEntries = useMemo(
    () => logsByDate.get(selectedDate) ?? [],
    [logsByDate, selectedDate],
  );

  const trimmedQuery = composerMode === "log" ? searchQuery.trim() : "";

  useEffect(() => {
    if (!trimmedQuery) {
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      setSearchLoading(true);
      searchSongs(trimmedQuery)
        .then((songs) => {
          if (cancelled) {
            return;
          }
          setSearchResults(songs);
          setHighlightedIndex(0);
          setSearchLoading(false);
        })
        .catch(() => {
          if (cancelled) {
            return;
          }
          setSearchResults([]);
          setSearchLoading(false);
        });
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [trimmedQuery]);

  const addMutation = useMutation({
    mutationFn: createListeningLog,
    onSuccess: () => {
      setMutationError(null);
      void queryClient.invalidateQueries({ queryKey: ["listening-logs"] });
    },
    onError: (error) => {
      setMutationError(error.message);
    },
  });

  const removeMutation = useMutation({
    mutationFn: deleteListeningLog,
    onSuccess: () => {
      setMutationError(null);
      void queryClient.invalidateQueries({ queryKey: ["listening-logs"] });
    },
    onError: (error) => {
      setMutationError(error.message);
    },
  });

  const selectDate = useCallback((dateKey: string) => {
    const parts = parseDateKey(dateKey);
    setSelectedDate(dateKey);
    setVisibleYear(parts.year);
    setVisibleMonthIndex(parts.monthIndex);
    setShelfExpanded(false);
  }, []);

  const toggleShelfExpanded = useCallback(() => {
    setShelfExpanded((prev) => !prev);
  }, []);

  const shiftWeek = useCallback(
    (deltaWeeks: number) => {
      selectDate(shiftDateKey(selectedDate, deltaWeeks * 7));
    },
    [selectDate, selectedDate],
  );

  const shiftMonth = useCallback(
    (deltaMonths: number) => {
      const next = new Date(visibleYear, visibleMonthIndex + deltaMonths, 1);
      setVisibleYear(next.getFullYear());
      setVisibleMonthIndex(next.getMonth());
    },
    [visibleMonthIndex, visibleYear],
  );

  const visibleSearchResults = trimmedQuery ? searchResults : [];
  const visibleSearchLoading = trimmedQuery ? searchLoading : false;
  const visibleHighlighted = trimmedQuery ? highlightedIndex : 0;

  const moveHighlight = useCallback(
    (delta: -1 | 1) => {
      if (visibleSearchResults.length === 0) {
        return;
      }
      setHighlightedIndex(
        (prev) => (prev + delta + visibleSearchResults.length) % visibleSearchResults.length,
      );
    },
    [visibleSearchResults.length],
  );

  const addSong = useCallback(
    (song: AppleSong) => {
      const isrc = song.attributes.isrc;
      if (!isrc) {
        return;
      }
      setSearchQuery("");
      setHighlightedIndex(0);
      addMutation.mutate({
        songId: song.id,
        isrc,
        listenedOn: selectedDate,
      });
    },
    [addMutation, selectedDate],
  );

  const removeLog = useCallback(
    (logId: string) => {
      removeMutation.mutate(logId);
    },
    [removeMutation],
  );

  const writeReviewForLog = useCallback(
    (log: ListeningLogView) => {
      const appleSong: AppleSong = {
        id: log.song.appleSongId,
        attributes: {
          name: log.song.title ?? "（曲名なし）",
          artistName: log.song.artistName ?? "",
          albumName: log.song.albumName ?? undefined,
          artwork: log.song.artworkUrl ? { url: log.song.artworkUrl } : undefined,
          durationInMillis: log.song.durationMs ?? undefined,
          isrc: log.song.isrc,
          url: log.song.url ?? undefined,
        },
      };
      setComposerMode("review");
      chat.actions.selectSong(appleSong);
    },
    [chat.actions],
  );

  return (
    <ListeningLogContext
      value={{
        state: {
          composerMode,
          todayKey,
          selectedDate,
          visibleYear,
          visibleMonthIndex,
          calendarExpanded,
          shelfExpanded,
          logsByDate,
          selectedEntries,
          isLoading: query.isPending,
          errorMessage: mutationError ?? (query.isError ? query.error.message : null),
          searchQuery,
          searchResults: visibleSearchResults,
          searchLoading: visibleSearchLoading,
          highlightedIndex: visibleHighlighted,
          isSaving: addMutation.isPending || removeMutation.isPending,
        },
        actions: {
          setComposerMode,
          selectDate,
          shiftWeek,
          shiftMonth,
          setCalendarExpanded,
          setShelfExpanded,
          toggleShelfExpanded,
          setSearchQuery,
          setHighlightedIndex,
          moveHighlight,
          addSong,
          removeLog,
          writeReviewForLog,
          registerInput,
        },
        meta: {
          inputRef,
        },
      }}
    >
      {children}
    </ListeningLogContext>
  );
}
