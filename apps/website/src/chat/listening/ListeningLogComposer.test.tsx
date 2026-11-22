// @vitest-environment jsdom
import "#/test/setup";
import { render } from "@testing-library/react";
import { createAgent } from "@ui-sdk/react/agent";
import { useRef, useState } from "react";
import { describe, expect, it, vi } from "vite-plus/test";
import type { AppleSong } from "#/lib/musickit";
import { ChatContext } from "../ChatContext";
import { ListeningComposer } from "./ListeningLogComposer";
import { ListeningLogContext } from "./ListeningLogContext";
import type { ComposerMode } from "./ListeningLogContext";

const agent = createAgent();

const sampleSong: AppleSong = {
  id: "song-1",
  attributes: {
    name: "群青",
    artistName: "YOASOBI",
    isrc: "JPXX02000001",
  },
};

function ComposerHarness({ onAddSong }: { onAddSong: (song: AppleSong) => void }) {
  const [composerMode, setComposerMode] = useState<ComposerMode>("review");
  const [selectedDate, setSelectedDate] = useState("2026-11-22");
  const [searchQuery, setSearchQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const composerRef = useRef<HTMLElement | null>(null);

  return (
    <ChatContext
      value={{
        state: {
          turns: [],
          selectedSong: null,
          draft: "",
          length: 0,
          canSubmit: false,
          isSubmitting: false,
          isComposing: false,
          lastSubmission: null,
          searchTrigger: null,
          searchQuery: "",
          menuOpen: false,
          keyboardOpen: false,
          songs: [],
          songsLoading: false,
          highlightedIndex: 0,
        },
        actions: {
          selectSong: vi.fn(),
          clearSong: vi.fn(),
          updateDraft: vi.fn(),
          setComposing: vi.fn(),
          openSongSearch: vi.fn(),
          dismissSongSearch: vi.fn(),
          submit: vi.fn(),
          setKeyboardOpen: vi.fn(),
          highlightSong: vi.fn(),
          moveHighlight: vi.fn(),
          registerTextarea: vi.fn(),
          registerComposer: vi.fn(),
        },
        meta: {
          maxLength: 500,
          textareaRef,
          composerRef,
          composerEl: null,
          pickerId: "picker-test",
        },
      }}
    >
      <ListeningLogContext
        value={{
          state: {
            composerMode,
            todayKey: "2026-11-22",
            selectedDate,
            visibleYear: 2026,
            visibleMonthIndex: 10,
            calendarExpanded: false,
            shelfExpanded: false,
            logsByDate: new Map(),
            selectedEntries: [],
            isLoading: false,
            errorMessage: null,
            searchQuery,
            searchResults: searchQuery.trim() ? [sampleSong] : [],
            searchLoading: false,
            isSaving: false,
          },
          actions: {
            setComposerMode,
            selectDate: setSelectedDate,
            shiftWeek: vi.fn(),
            shiftMonth: vi.fn(),
            setCalendarExpanded: vi.fn(),
            setShelfExpanded: vi.fn(),
            toggleShelfExpanded: vi.fn(),
            setSearchQuery,
            addSong: onAddSong,
            removeLog: vi.fn(),
            writeReviewForLog: vi.fn(),
            registerInput: (el) => {
              inputRef.current = el;
            },
          },
          meta: { inputRef },
        }}
      >
        <ListeningComposer.ModeSwitch />
        <ListeningComposer.TargetDatePicker />
        <ListeningComposer.SearchCombobox />
      </ListeningLogContext>
    </ChatContext>
  );
}

describe("ListeningLogComposer agent readiness", () => {
  it("lets an agent switch mode and search songs via SegmentedControl and Combobox", async () => {
    const onAddSong = vi.fn();
    render(<ComposerHarness onAddSong={onAddSong} />);

    const components = agent
      .snapshot()
      .filter((entry) => !entry.offPage && entry.component !== "more");
    for (const entry of components) {
      expect(entry.label, `${entry.component} ${entry.ref} has no name`).toBeTruthy();
    }

    const labels = components.map((entry) => entry.label);
    expect(labels).toContain("入力モード");
    expect(labels).toContain("聴いた曲を検索して追加");

    const modeResult = await agent.act({
      steps: [{ ref: "入力モード", action: "setValue", args: ["聴いた曲を記録"] }],
    });
    expect(modeResult.ok, modeResult.error?.message).toBe(true);
    expect(
      agent.snapshot().find((entry) => !entry.offPage && entry.label === "入力モード")?.value,
    ).toBe("log");
  });
});
