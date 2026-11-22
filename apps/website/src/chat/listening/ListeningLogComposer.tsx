import { create, props } from "@stylexjs/stylex";
import { Button } from "@ui-sdk/react/button";
import {
  Popover,
  PopoverClose,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@ui-sdk/react/popover";
import { useMemo, useState } from "react";
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
import {
  buildMonthGrid,
  formatJapaneseDate,
  formatMonthHeading,
  parseDateKey,
  shiftDateKey,
  WEEKDAY_LABELS,
} from "./calendar-utils";
import { useListeningLog } from "./ListeningLogContext";

function TargetDatePickerPill() {
  const { state, actions } = useListeningLog();
  const { selectedDate, todayKey } = state;
  const compact = useMediaQuery("(max-width: 540px)");
  const [open, setOpen] = useState(false);
  const initial = parseDateKey(selectedDate);
  const [pickerYear, setPickerYear] = useState(initial.year);
  const [pickerMonth, setPickerMonth] = useState(initial.monthIndex);

  const yesterdayKey = shiftDateKey(todayKey, -1);
  const twoDaysAgoKey = shiftDateKey(todayKey, -2);
  const cells = useMemo(
    () => buildMonthGrid(pickerYear, pickerMonth, todayKey),
    [pickerYear, pickerMonth, todayKey],
  );

  const pickDate = (dateKey: string) => {
    actions.selectDate(dateKey);
    setOpen(false);
  };

  const shiftPickerMonth = (delta: number) => {
    const next = new Date(pickerYear, pickerMonth + delta, 1);
    setPickerYear(next.getFullYear());
    setPickerMonth(next.getMonth());
  };

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen) {
      const parts = parseDateKey(selectedDate);
      setPickerYear(parts.year);
      setPickerMonth(parts.monthIndex);
    }
  };

  const quickDayItems = [
    { label: "今日", key: todayKey },
    { label: "昨日", key: yesterdayKey },
    { label: "一昨日", key: twoDaysAgoKey },
  ];

  const triggerLabel = `登録先の日付を変更（現在: ${formatJapaneseDate(selectedDate)}）`;

  const miniMonthBody = (
    <>
      <div {...props(styles.miniMonthHead)}>
        <span {...props(styles.miniMonthTitle)}>{formatMonthHeading(pickerYear, pickerMonth)}</span>
        <div {...props(styles.miniMonthNav)}>
          <Button
            type="button"
            aria-label="前の月"
            onClick={() => shiftPickerMonth(-1)}
            {...props(styles.miniNavBtn)}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            type="button"
            aria-label="次の月"
            onClick={() => shiftPickerMonth(1)}
            {...props(styles.miniNavBtn)}
          >
            <ChevronRightIcon />
          </Button>
        </div>
      </div>

      <div {...props(styles.miniGrid)}>
        {WEEKDAY_LABELS.map((w) => (
          <span key={w} {...props(styles.miniWeekday)}>
            {w}
          </span>
        ))}
        {cells.map((cell) => {
          const isSelected = cell.dateKey === selectedDate;
          return (
            <button
              key={cell.dateKey}
              type="button"
              aria-pressed={isSelected}
              aria-label={formatJapaneseDate(cell.dateKey)}
              onClick={() => pickDate(cell.dateKey)}
              {...props(
                styles.miniDayBtn,
                !cell.inCurrentMonth && styles.miniDayMuted,
                cell.isToday && styles.miniDayToday,
                isSelected && styles.miniDaySelected,
              )}
            >
              {cell.day}
            </button>
          );
        })}
      </div>
    </>
  );

  if (compact) {
    return (
      <BottomSheetDrawer.Root open={open} onOpenChange={handleOpenChange}>
        <BottomSheetDrawer.Trigger aria-label={triggerLabel} {...props(styles.datePillTrigger)}>
          <span>登録先: {formatJapaneseDate(selectedDate)}</span>
          <ChevronDownIcon />
        </BottomSheetDrawer.Trigger>
        <BottomSheetDrawer.Content>
          <div {...props(styles.datePopoverHead)}>
            <BottomSheetDrawer.Title {...props(styles.datePopoverTitle)}>
              登録先の日付を選ぶ
            </BottomSheetDrawer.Title>
          </div>
          <div {...props(styles.quickDays)}>
            {quickDayItems.map((item) => {
              const active = selectedDate === item.key;
              return (
                <BottomSheetDrawer.Close
                  key={item.key}
                  onClick={() => pickDate(item.key)}
                  {...props(styles.quickDayBtn, active && styles.quickDayBtnActive)}
                >
                  {item.label}
                </BottomSheetDrawer.Close>
              );
            })}
          </div>
          {miniMonthBody}
        </BottomSheetDrawer.Content>
      </BottomSheetDrawer.Root>
    );
  }

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger aria-label={triggerLabel} {...props(styles.datePillTrigger)}>
        <span>登録先: {formatJapaneseDate(selectedDate)}</span>
        <ChevronDownIcon />
      </PopoverTrigger>

      <PopoverContent side="top" align="end" sideOffset={8} {...props(styles.datePopover)}>
        <div {...props(styles.datePopoverHead)}>
          <PopoverTitle {...props(styles.datePopoverTitle)}>登録先の日付を選ぶ</PopoverTitle>
        </div>

        <div {...props(styles.quickDays)}>
          {quickDayItems.map((item) => {
            const active = selectedDate === item.key;
            return (
              <PopoverClose
                key={item.key}
                onClick={() => pickDate(item.key)}
                {...props(styles.quickDayBtn, active && styles.quickDayBtnActive)}
              >
                {item.label}
              </PopoverClose>
            );
          })}
        </div>

        {miniMonthBody}
      </PopoverContent>
    </Popover>
  );
}

function SearchPopover() {
  const { state, actions, meta } = useListeningLog();
  const chat = useChat();
  const { searchQuery, searchResults, searchLoading, highlightedIndex } = state;

  const showResults = searchQuery.trim().length > 0;
  const maxHeight = useComposerPopoverHeight({
    open: showResults,
    keyboardOpen: chat.state.keyboardOpen,
    composerEl: chat.meta.composerEl,
    composerRef: chat.meta.composerRef,
  });

  if (!showResults) {
    return null;
  }

  return (
    <div
      {...props(styles.searchPopover, styles.searchPopoverMaxHeight(maxHeight))}
      role="listbox"
      aria-label="聴いた曲の検索候補"
    >
      <div {...props(styles.searchList)}>
        {searchLoading ? (
          <div {...props(styles.statusText)}>曲を検索中…</div>
        ) : searchResults.length === 0 ? (
          <div {...props(styles.statusText)}>一致する曲が見つかりません</div>
        ) : (
          searchResults.slice(0, 8).map((song, idx) => {
            const { name, artistName, artwork, isrc } = song.attributes;
            const disabled = !isrc;
            return (
              <button
                key={song.id}
                type="button"
                role="option"
                disabled={disabled}
                aria-selected={idx === highlightedIndex}
                onMouseEnter={() => actions.setHighlightedIndex(idx)}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  actions.addSong(song);
                  if (window.matchMedia("(pointer: fine)").matches) {
                    meta.inputRef.current?.focus();
                  }
                }}
                {...props(
                  styles.resultRow,
                  idx === highlightedIndex && styles.resultRowActive,
                  disabled && styles.resultRowDisabled,
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
              </button>
            );
          })
        )}
      </div>

      {chat.state.keyboardOpen && searchResults.length > 0 ? (
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
    </div>
  );
}

function ModeSwitch() {
  const { state, actions, meta } = useListeningLog();
  const { composerMode } = state;

  return (
    <div {...props(styles.modeSwitch)} role="tablist" aria-label="入力モード">
      <button
        type="button"
        role="tab"
        aria-selected={composerMode === "review"}
        onClick={() => actions.setComposerMode("review")}
        {...props(styles.modeBtn, composerMode === "review" && styles.modeBtnActive)}
      >
        <MessageIcon />
        <span>感想を残す</span>
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={composerMode === "log"}
        onClick={() => {
          actions.setComposerMode("log");
          if (window.matchMedia("(pointer: fine)").matches) {
            requestAnimationFrame(() => meta.inputRef.current?.focus());
          }
        }}
        {...props(styles.modeBtn, composerMode === "log" && styles.modeBtnActive)}
      >
        <CalendarIcon />
        <span>聴いた曲を記録</span>
      </button>
    </div>
  );
}

function Input() {
  const { state, actions } = useListeningLog();
  const { searchQuery, searchResults, highlightedIndex, isSaving } = state;
  const { registerInput } = actions;
  const showResults = searchQuery.trim().length > 0;

  return (
    <div {...props(styles.logInputRow)}>
      <SearchIcon />
      <input
        ref={registerInput}
        type="text"
        enterKeyHint="search"
        aria-label="聴いた曲を検索して追加"
        placeholder={isSaving ? "記録を保存中…" : "曲名・アーティスト名で検索して追加（複数OK）…"}
        value={searchQuery}
        onChange={(event) => actions.setSearchQuery(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (!showResults || searchResults.length === 0) {
            return;
          }
          if (event.key === "Escape") {
            event.preventDefault();
            actions.setSearchQuery("");
          } else if (event.key === "ArrowDown") {
            event.preventDefault();
            actions.moveHighlight(1);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            actions.moveHighlight(-1);
          } else if (event.key === "Enter" && !event.nativeEvent.isComposing) {
            event.preventDefault();
            const target = searchResults[highlightedIndex];
            if (target?.attributes.isrc) {
              actions.addSong(target);
            }
          }
        }}
        {...props(styles.logInput)}
      />
    </div>
  );
}

export const ListeningComposer = {
  ModeSwitch,
  TargetDatePicker: TargetDatePickerPill,
  SearchPopover,
  Input,
};

const styles = create({
  modeSwitch: {
    display: "inline-flex",
    gap: 2,
    padding: 2,
    backgroundColor: t.bg,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9999,
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
  },
  modeBtnActive: {
    color: t.primaryFg,
    backgroundColor: t.primary,
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
  miniMonthHead: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 2,
  },
  miniMonthTitle: {
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
    display: "grid",
    gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
    gap: 2,
  },
  miniWeekday: {
    paddingBlock: 2,
    fontSize: 10.5,
    fontWeight: 600,
    color: t.muted,
    textAlign: "center",
  },
  miniDayBtn: {
    display: "grid",
    placeItems: "center",
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
    position: "absolute",
    right: 0,
    bottom: `calc(100% + ${chatPopover.sideOffset}px)`,
    left: 0,
    zIndex: 50,
    display: "flex",
    flexDirection: "column",
    gap: {
      default: 4,
      [chatMedia.compact]: 2,
    },
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
    transformOrigin: "center bottom",
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
