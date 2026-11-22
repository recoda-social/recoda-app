import { create, props } from "@stylexjs/stylex";
import { Button } from "@ui-sdk/react/button";
import {
  Calendar,
  CalendarDay,
  CalendarGrid,
  CalendarHeading,
  CalendarNext,
  CalendarPrevious,
  CalendarWeekday,
  CalendarWeekdays,
  CalendarWeeks,
} from "@ui-sdk/react/calendar";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxStatus,
} from "@ui-sdk/react/combobox";
import {
  Popover,
  PopoverClose,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@ui-sdk/react/popover";
import {
  SegmentedControl,
  SegmentedControlIndicator,
  SegmentedControlItem,
} from "@ui-sdk/react/segmented-control";
import { createContext, use, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { artworkUrl } from "#/lib/musickit";
import { useMediaQuery } from "#/lib/use-media-query";
import { motion } from "#/motion.stylex";
import {
  CalendarIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MessageIcon,
  MusicNoteIcon,
  PlusIcon,
  SearchIcon,
} from "../icons";
import { useChat } from "../ChatContext";
import { chatMedia, chatPopover, chatTokens as t } from "../tokens.stylex";
import { useComposerPopoverHeight } from "../use-composer-popover-height";
import { BottomSheetDrawer } from "./BottomSheetDrawer";
import { formatJapaneseDate, shiftDateKey } from "./calendar-utils";
import { useListeningLog } from "./ListeningLogContext";

type TargetDatePickerContextValue = {
  state: {
    open: boolean;
    selectedDate: string;
    todayKey: string;
    pickerMonth: string;
    quickDayItems: ReadonlyArray<{ label: string; key: string }>;
  };
  actions: {
    onOpenChange: (nextOpen: boolean) => void;
    pickDate: (dateKey: string) => void;
    setPickerMonth: (month: string) => void;
  };
  meta: {
    triggerLabel: string;
  };
};

const TargetDatePickerContext = createContext<TargetDatePickerContextValue | null>(null);

function useTargetDatePicker() {
  const value = use(TargetDatePickerContext);
  if (!value) {
    throw new Error("TargetDatePicker の部品は TargetDatePickerProvider の中で使う");
  }
  return value;
}

function TargetDatePickerProvider({ children }: { children: ReactNode }) {
  const { state, actions } = useListeningLog();
  const { selectedDate, todayKey } = state;
  const [open, setOpen] = useState(false);
  const [pickerMonth, setPickerMonth] = useState(() => selectedDate.slice(0, 7));

  const yesterdayKey = shiftDateKey(todayKey, -1);
  const twoDaysAgoKey = shiftDateKey(todayKey, -2);

  const quickDayItems = useMemo(
    () => [
      { label: "今日", key: todayKey },
      { label: "昨日", key: yesterdayKey },
      { label: "一昨日", key: twoDaysAgoKey },
    ],
    [todayKey, yesterdayKey, twoDaysAgoKey],
  );

  const pickDate = (dateKey: string) => {
    actions.selectDate(dateKey);
    setOpen(false);
  };

  const onOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen) {
      setPickerMonth(selectedDate.slice(0, 7));
    }
  };

  const triggerLabel = `登録先の日付を変更（現在: ${formatJapaneseDate(selectedDate)}）`;

  return (
    <TargetDatePickerContext
      value={{
        state: { open, selectedDate, todayKey, pickerMonth, quickDayItems },
        actions: { onOpenChange, pickDate, setPickerMonth },
        meta: { triggerLabel },
      }}
    >
      {children}
    </TargetDatePickerContext>
  );
}

function MiniMonthGrid() {
  const { state, actions } = useTargetDatePicker();

  return (
    <Calendar
      mode="single"
      locale="ja-JP"
      weekStartsOn={0}
      fixedWeeks
      showOutsideDays
      today={state.todayKey}
      value={state.selectedDate}
      onValueChange={(next) => {
        if (typeof next === "string") {
          actions.pickDate(next);
        }
      }}
      month={state.pickerMonth}
      onMonthChange={(next) => actions.setPickerMonth(next)}
      {...props(styles.miniCalendar)}
    >
      <div {...props(styles.miniMonthHead)}>
        <CalendarHeading {...props(styles.miniMonthTitle)} />
        <div {...props(styles.miniMonthNav)}>
          <CalendarPrevious aria-label="前の月" {...props(styles.miniNavBtn)}>
            <ChevronLeftIcon />
          </CalendarPrevious>
          <CalendarNext aria-label="次の月" {...props(styles.miniNavBtn)}>
            <ChevronRightIcon />
          </CalendarNext>
        </div>
      </div>

      <CalendarGrid {...props(styles.miniGrid)}>
        <CalendarWeekdays>
          {(index) => <CalendarWeekday index={index} {...props(styles.miniWeekday)} />}
        </CalendarWeekdays>
        <CalendarWeeks>
          {(cell) => {
            const dayNum = Number(cell.date.slice(8));
            return (
              <CalendarDay value={cell.date} {...props(styles.miniDayTd)}>
                {(dayState) => (
                  <span
                    {...props(
                      styles.miniDayBtn,
                      dayState.outside && styles.miniDayMuted,
                      dayState.today && styles.miniDayToday,
                      dayState.selected && styles.miniDaySelected,
                    )}
                  >
                    {dayNum}
                  </span>
                )}
              </CalendarDay>
            );
          }}
        </CalendarWeeks>
      </CalendarGrid>
    </Calendar>
  );
}

function MobileTargetDatePicker() {
  const { state, actions, meta } = useTargetDatePicker();

  return (
    <BottomSheetDrawer.Root open={state.open} onOpenChange={actions.onOpenChange}>
      <BottomSheetDrawer.Trigger aria-label={meta.triggerLabel} {...props(styles.datePillTrigger)}>
        <span>登録先: {formatJapaneseDate(state.selectedDate)}</span>
        <ChevronDownIcon />
      </BottomSheetDrawer.Trigger>
      <BottomSheetDrawer.Content>
        <div {...props(styles.datePopoverHead)}>
          <BottomSheetDrawer.Title {...props(styles.datePopoverTitle)}>
            登録先の日付を選ぶ
          </BottomSheetDrawer.Title>
        </div>
        <div {...props(styles.quickDays)}>
          {state.quickDayItems.map((item) => {
            const active = state.selectedDate === item.key;
            return (
              <BottomSheetDrawer.Close
                key={item.key}
                onClick={() => actions.pickDate(item.key)}
                {...props(styles.quickDayBtn, active && styles.quickDayBtnActive)}
              >
                {item.label}
              </BottomSheetDrawer.Close>
            );
          })}
        </div>
        <MiniMonthGrid />
      </BottomSheetDrawer.Content>
    </BottomSheetDrawer.Root>
  );
}

function DesktopTargetDatePicker() {
  const { state, actions, meta } = useTargetDatePicker();

  return (
    <Popover open={state.open} onOpenChange={actions.onOpenChange}>
      <PopoverTrigger aria-label={meta.triggerLabel} {...props(styles.datePillTrigger)}>
        <span>登録先: {formatJapaneseDate(state.selectedDate)}</span>
        <ChevronDownIcon />
      </PopoverTrigger>

      <PopoverContent side="top" align="end" sideOffset={8} {...props(styles.datePopover)}>
        <div {...props(styles.datePopoverHead)}>
          <PopoverTitle {...props(styles.datePopoverTitle)}>登録先の日付を選ぶ</PopoverTitle>
        </div>

        <div {...props(styles.quickDays)}>
          {state.quickDayItems.map((item) => {
            const active = state.selectedDate === item.key;
            return (
              <PopoverClose
                key={item.key}
                onClick={() => actions.pickDate(item.key)}
                {...props(styles.quickDayBtn, active && styles.quickDayBtnActive)}
              >
                {item.label}
              </PopoverClose>
            );
          })}
        </div>

        <MiniMonthGrid />
      </PopoverContent>
    </Popover>
  );
}

function TargetDatePickerPill() {
  const compact = useMediaQuery("(max-width: 540px)");

  return (
    <TargetDatePickerProvider>
      {compact ? <MobileTargetDatePicker /> : <DesktopTargetDatePicker />}
    </TargetDatePickerProvider>
  );
}

function SearchCombobox() {
  const { state, actions, meta } = useListeningLog();
  const chat = useChat();
  const { searchQuery, searchResults, searchLoading, isSaving } = state;
  const { registerInput } = actions;

  const showResults = searchQuery.trim().length > 0;
  const visibleSongs = useMemo(() => searchResults.slice(0, 8), [searchResults]);
  const comboboxItems = useMemo(
    () =>
      visibleSongs.map((song) => ({
        value: song.id,
        label: song.attributes.name,
        disabled: !song.attributes.isrc,
      })),
    [visibleSongs],
  );
  const maxHeight = useComposerPopoverHeight({
    open: showResults,
    keyboardOpen: chat.state.keyboardOpen,
    composerEl: chat.meta.composerEl,
    composerRef: chat.meta.composerRef,
  });

  return (
    <Combobox
      items={comboboxItems}
      filter={false}
      autoHighlight
      status={searchLoading ? "loading" : "idle"}
      query={searchQuery}
      onQueryChange={(nextQuery) => actions.setSearchQuery(nextQuery)}
      open={showResults}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          actions.setSearchQuery("");
        }
      }}
      value={null}
      onValueChange={(nextValue) => {
        if (typeof nextValue !== "string") {
          return;
        }
        const song = visibleSongs.find((item) => item.id === nextValue);
        if (!song) {
          return;
        }
        actions.addSong(song);
        if (window.matchMedia("(pointer: fine)").matches) {
          meta.inputRef.current?.focus();
        }
      }}
    >
      <div {...props(styles.logInputRow)}>
        <SearchIcon />
        <ComboboxInput
          ref={registerInput}
          enterKeyHint="search"
          aria-label="聴いた曲を検索して追加"
          placeholder={isSaving ? "記録を保存中…" : "曲名・アーティスト名で検索して追加（複数OK）…"}
          {...props(styles.logInput)}
        />
      </div>

      <ComboboxContent
        side="top"
        align="start"
        sideOffset={chatPopover.sideOffset}
        {...props(styles.searchPopover, styles.searchPopoverMaxHeight(maxHeight))}
      >
        <ComboboxStatus {...props(styles.statusText)}>
          {(statusState) => (statusState.loading ? "曲を検索中…" : null)}
        </ComboboxStatus>
        <ComboboxEmpty {...props(styles.statusText)}>
          {(emptyState) =>
            emptyState.empty && !emptyState.loading ? "一致する曲が見つかりません" : null
          }
        </ComboboxEmpty>
        <ComboboxList aria-label="聴いた曲の検索候補" {...props(styles.searchList)}>
          {visibleSongs.map((song) => {
            const { name, artistName, artwork, isrc } = song.attributes;
            const disabled = !isrc;
            return (
              <ComboboxItem
                key={song.id}
                value={song.id}
                label={name}
                disabled={disabled}
                render={(itemProps, itemState) => (
                  <div
                    {...itemProps}
                    {...props(
                      styles.resultRow,
                      itemState.highlighted && styles.resultRowActive,
                      itemState.disabled && styles.resultRowDisabled,
                    )}
                  />
                )}
              >
                {artwork ? (
                  <img
                    src={artworkUrl(artwork.url, 72)}
                    alt=""
                    width={32}
                    height={32}
                    {...props(styles.resultArt)}
                  />
                ) : (
                  <span {...props(styles.resultArt, styles.artFallback)} aria-hidden="true">
                    <MusicNoteIcon />
                  </span>
                )}
                <span {...props(styles.metaCol)}>
                  <span {...props(styles.metaTitle)}>{name}</span>
                  <span {...props(styles.metaArtist)}>{artistName}</span>
                </span>
                <span {...props(styles.addBadge)}>
                  {disabled ? (
                    "選択不可"
                  ) : (
                    <>
                      <PlusIcon />
                      <span>追加</span>
                    </>
                  )}
                </span>
              </ComboboxItem>
            );
          })}
        </ComboboxList>

        {chat.state.keyboardOpen && visibleSongs.length > 0 ? (
          <div {...props(styles.searchFooter)}>
            <Button
              type="button"
              aria-label="キーボードを閉じて検索結果を広く表示する"
              onClick={() => meta.inputRef.current?.blur()}
              {...props(styles.expandResultsBtn)}
            >
              検索結果を広く表示
            </Button>
          </div>
        ) : null}
      </ComboboxContent>
    </Combobox>
  );
}

function ModeSwitch() {
  const { state, actions, meta } = useListeningLog();
  const { composerMode } = state;

  return (
    <SegmentedControl
      aria-label="入力モード"
      value={composerMode}
      onValueChange={(next) => {
        if (next !== "review" && next !== "log") {
          return;
        }
        actions.setComposerMode(next);
        if (next === "log" && window.matchMedia("(pointer: fine)").matches) {
          requestAnimationFrame(() => meta.inputRef.current?.focus());
        }
      }}
      {...props(styles.modeSwitch)}
    >
      <SegmentedControlIndicator {...props(styles.modeIndicator)} />
      <SegmentedControlItem
        value="review"
        render={(itemProps, itemState) => (
          <button
            type="button"
            {...itemProps}
            {...props(styles.modeBtn, itemState.checked && styles.modeBtnActive)}
          />
        )}
      >
        <MessageIcon />
        <span>感想を残す</span>
      </SegmentedControlItem>
      <SegmentedControlItem
        value="log"
        render={(itemProps, itemState) => (
          <button
            type="button"
            {...itemProps}
            {...props(styles.modeBtn, itemState.checked && styles.modeBtnActive)}
          />
        )}
      >
        <CalendarIcon />
        <span>聴いた曲を記録</span>
      </SegmentedControlItem>
    </SegmentedControl>
  );
}

export const ListeningComposer = {
  ModeSwitch,
  TargetDatePicker: TargetDatePickerPill,
  SearchCombobox,
};

const styles = create({
  modeSwitch: {
    position: "relative",
    display: "inline-flex",
    gap: 2,
    padding: 2,
    backgroundColor: t.bg,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9999,
    isolation: "isolate",
  },
  modeIndicator: {
    position: "absolute",
    top: "var(--segmented-control-indicator-top)",
    left: "var(--segmented-control-indicator-left)",
    zIndex: -1,
    width: "var(--segmented-control-indicator-width)",
    height: "var(--segmented-control-indicator-height)",
    backgroundColor: t.primary,
    borderRadius: 9999,
    transition: {
      default: `left 180ms ${motion.easeOut}, width 180ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "none",
    },
  },
  modeBtn: {
    display: "inline-flex",
    gap: 5,
    alignItems: "center",
    height: 26,
    paddingBlock: 0,
    paddingInline: 10,
    fontFamily: "inherit",
    fontSize: 12,
    fontWeight: 500,
    color: t.muted,
    cursor: "pointer",
    backgroundColor: "transparent",
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 9999,
    transition: "color 150ms ease",
  },
  modeBtnActive: {
    color: t.primaryFg,
  },
  datePillTrigger: {
    display: "inline-flex",
    gap: 5,
    alignItems: "center",
    height: 28,
    paddingBlock: 0,
    paddingInline: 10,
    fontFamily: "inherit",
    fontSize: 12,
    fontWeight: 500,
    fontVariantNumeric: "tabular-nums",
    color: t.text,
    cursor: "pointer",
    backgroundColor: {
      default: t.bg,
      ":hover": {
        default: t.bg,
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9999,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.97)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 150ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "background-color 120ms ease",
    },
  },
  datePopover: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    width: "min(290px, calc(100vw - 24px))",
    padding: 12,
    color: t.text,
    backgroundColor: t.surfaceElevated,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 16,
    boxShadow: t.shadow,
    opacity: {
      "@starting-style": 0,
      default: 1,
    },
    transform: {
      "@starting-style": {
        default: "scale(0.96)",
        [chatMedia.reducedMotion]: "none",
      },
      default: "scale(1)",
    },
    transformOrigin: "var(--transform-origin)",
    transition: {
      default: `opacity 150ms ${motion.easeOut}, transform 160ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "opacity 120ms ease",
    },
  },
  datePopoverHead: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  datePopoverTitle: {
    margin: 0,
    fontSize: 13.5,
    fontWeight: 600,
  },
  quickDays: {
    display: "grid",
    gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
    gap: 6,
  },
  quickDayBtn: {
    height: 32,
    paddingBlock: 0,
    paddingInline: 8,
    fontFamily: "inherit",
    fontSize: 12.5,
    fontWeight: 500,
    color: t.text,
    cursor: "pointer",
    backgroundColor: {
      default: t.surface,
      ":hover": {
        default: t.surface,
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.97)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 150ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "background-color 120ms ease",
    },
  },
  quickDayBtnActive: {
    color: t.primaryFg,
    backgroundColor: t.primary,
    borderColor: t.primary,
  },
  miniCalendar: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
    width: "100%",
    padding: 0,
    backgroundColor: "transparent",
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 0,
  },
  miniMonthHead: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 2,
  },
  miniMonthTitle: {
    margin: 0,
    fontSize: 12.5,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
  },
  miniMonthNav: {
    display: "inline-flex",
    gap: 2,
  },
  miniNavBtn: {
    display: "grid",
    placeItems: "center",
    width: 26,
    height: 26,
    padding: 0,
    color: t.muted,
    cursor: "pointer",
    backgroundColor: {
      default: "transparent",
      ":hover": {
        default: "transparent",
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 7,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.97)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 150ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "background-color 120ms ease",
    },
  },
  miniGrid: {
    width: "100%",
    tableLayout: "fixed",
    borderSpacing: 2,
    borderCollapse: "separate",
  },
  miniWeekday: {
    width: "auto",
    height: "auto",
    paddingBlock: 2,
    fontSize: 10.5,
    fontWeight: 600,
    color: t.muted,
    textAlign: "center",
  },
  miniDayTd: {
    width: "auto",
    height: "auto",
    padding: 0,
    verticalAlign: "middle",
    backgroundColor: "transparent",
    borderRadius: 0,
  },
  miniDayBtn: {
    display: "grid",
    placeItems: "center",
    width: "100%",
    height: 30,
    padding: 0,
    fontFamily: "inherit",
    fontSize: 12,
    fontWeight: 500,
    fontVariantNumeric: "tabular-nums",
    color: t.text,
    cursor: "pointer",
    backgroundColor: {
      default: "transparent",
      ":hover": {
        default: "transparent",
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 8,
  },
  miniDayMuted: {
    opacity: 0.35,
  },
  miniDayToday: {
    fontWeight: 700,
    boxShadow: `inset 0 0 0 1px ${t.borderStrong}`,
  },
  miniDaySelected: {
    color: t.primaryFg,
    backgroundColor: t.primary,
  },
  logInputRow: {
    display: "flex",
    gap: 9,
    alignItems: "center",
    paddingBlock: 7,
    paddingInline: 10,
    color: t.muted,
    backgroundColor: t.bg,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 12,
  },
  logInput: {
    width: "100%",
    fontFamily: "inherit",
    fontSize: {
      default: 14,
      [chatMedia.touch]: 16,
    },
    color: t.text,
    outline: "none",
    backgroundColor: "transparent",
    borderStyle: "none",
    borderWidth: 0,
  },
  searchPopover: {
    display: "flex",
    flexDirection: "column",
    gap: {
      default: 4,
      [chatMedia.compact]: 2,
    },
    width: "var(--anchor-width)",
    maxHeight: chatPopover.maxHeight,
    padding: 6,
    overflow: "hidden",
    backgroundColor: t.surfaceElevated,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 14,
    boxShadow: t.shadow,
    opacity: {
      "@starting-style": 0,
      default: 1,
    },
    transform: {
      "@starting-style": {
        default: "translateY(4px)",
        [chatMedia.reducedMotion]: "none",
      },
      default: "translateY(0)",
    },
    transformOrigin: "var(--transform-origin)",
    transition: {
      default: `opacity 140ms ${motion.easeOut}, transform 140ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "opacity 120ms ease",
    },
  },
  searchPopoverMaxHeight: (maxHeight: number) => ({
    maxHeight,
  }),
  searchList: {
    display: "flex",
    flexGrow: 1,
    flexShrink: 1,
    flexDirection: "column",
    gap: 2,
    minHeight: 0,
    overflowY: "auto",
    overscrollBehaviorY: "contain",
  },
  searchFooter: {
    flexShrink: 0,
    paddingTop: 6,
    borderTopColor: t.border,
    borderTopStyle: "solid",
    borderTopWidth: 1,
  },
  expandResultsBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    minHeight: 36,
    paddingBlock: 6,
    paddingInline: 12,
    fontSize: 13,
    fontWeight: 500,
    color: t.text,
    cursor: "pointer",
    backgroundColor: {
      default: t.surface,
      ":hover": {
        default: t.surface,
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 10,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.98)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 150ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "background-color 120ms ease",
    },
  },
  statusText: {
    padding: 12,
    fontSize: 12.5,
    color: t.muted,
    textAlign: "center",
  },
  resultRow: {
    display: "flex",
    gap: 10,
    alignItems: "center",
    width: "100%",
    paddingBlock: 7,
    paddingInline: 10,
    fontFamily: "inherit",
    color: t.text,
    textAlign: "left",
    cursor: "pointer",
    backgroundColor: {
      default: "transparent",
      ":active": t.highlight,
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 9,
  },
  resultRowActive: {
    backgroundColor: t.highlight,
  },
  resultRowDisabled: {
    cursor: "not-allowed",
    opacity: 0.45,
  },
  resultArt: {
    flexShrink: 0,
    width: 32,
    height: 32,
    objectFit: "cover",
    borderRadius: 6,
  },
  artFallback: {
    display: "grid",
    placeItems: "center",
    color: t.muted,
    backgroundColor: t.hover,
  },
  metaCol: {
    display: "flex",
    flex: "1",
    flexDirection: "column",
    gap: 1,
    minWidth: 0,
  },
  metaTitle: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    fontSize: 13,
    fontWeight: 600,
    lineHeight: 1.3,
    whiteSpace: "nowrap",
  },
  metaArtist: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    fontSize: 11.5,
    lineHeight: 1.25,
    color: t.muted,
    whiteSpace: "nowrap",
  },
  addBadge: {
    display: "inline-flex",
    flexShrink: 0,
    gap: 4,
    alignItems: "center",
    fontSize: 12,
    fontWeight: 500,
    color: t.muted,
  },
});
