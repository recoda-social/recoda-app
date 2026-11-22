import { createContext, use } from "react";
import type { RefObject } from "react";
import type { ListeningLogView } from "#/lib/listening-logs";
import type { AppleSong } from "#/lib/musickit";

export type ComposerMode = "review" | "log";

type ListeningLogState = {
  composerMode: ComposerMode;
  todayKey: string;
  selectedDate: string;
  visibleYear: number;
  visibleMonthIndex: number;
  calendarExpanded: boolean;
  shelfExpanded: boolean;
  logsByDate: Map<string, Array<ListeningLogView>>;
  selectedEntries: Array<ListeningLogView>;
  isLoading: boolean;
  errorMessage: string | null;
  searchQuery: string;
  searchResults: Array<AppleSong>;
  searchLoading: boolean;
  highlightedIndex: number;
  isSaving: boolean;
};

type ListeningLogActions = {
  setComposerMode: (mode: ComposerMode) => void;
  selectDate: (dateKey: string) => void;
  shiftWeek: (deltaWeeks: number) => void;
  shiftMonth: (deltaMonths: number) => void;
  setCalendarExpanded: (expanded: boolean) => void;
  setShelfExpanded: (expanded: boolean) => void;
  toggleShelfExpanded: () => void;
  setSearchQuery: (query: string) => void;
  setHighlightedIndex: (index: number) => void;
  moveHighlight: (delta: -1 | 1) => void;
  addSong: (song: AppleSong) => void;
  removeLog: (logId: string) => void;
  writeReviewForLog: (log: ListeningLogView) => void;
  registerInput: (el: HTMLInputElement | null) => void;
};

type ListeningLogMeta = {
  inputRef: RefObject<HTMLInputElement | null>;
};

export type ListeningLogContextValue = {
  state: ListeningLogState;
  actions: ListeningLogActions;
  meta: ListeningLogMeta;
};

export const ListeningLogContext = createContext<ListeningLogContextValue | null>(null);

export function useListeningLog() {
  const value = use(ListeningLogContext);
  if (!value) {
    throw new Error("useListeningLog は ListeningLogProvider の中で使う");
  }
  return value;
}
