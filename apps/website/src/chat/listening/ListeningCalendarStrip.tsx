import { create, props } from "@stylexjs/stylex";
import { Button } from "@ui-sdk/react/button";
import { useMemo } from "react";
import type { ListeningLogView } from "#/lib/listening-logs";
import { artworkUrl } from "#/lib/musickit";
import { motion } from "#/motion.stylex";
import { CalendarIcon, ChevronLeftIcon, ChevronRightIcon, MusicNoteIcon } from "../icons";
import { chatMedia, chatTokens as t } from "../tokens.stylex";
import {
  buildMonthGrid,
  formatJapaneseDate,
  formatMonthHeading,
  getWeekContaining,
  WEEKDAY_LABELS,
} from "./calendar-utils";
import { useListeningLog } from "./ListeningLogContext";

function CoverTile({
  song,
  size,
  radius,
}: {
  song: ListeningLogView["song"];
  size: number;
  radius: number;
}) {
  if (song.artworkUrl) {
    return (
      <img
        src={artworkUrl(song.artworkUrl, size * 2)}
        alt=""
        width={size}
        height={size}
        loading="lazy"
        {...props(styles.coverImg)}
        style={{ width: size, height: size, borderRadius: radius }}
      />
    );
  }
  return (
    <span
      aria-hidden="true"
      {...props(styles.coverFallback)}
      style={{ width: size, height: size, borderRadius: radius }}
    >
      <MusicNoteIcon />
    </span>
  );
}

function CompactCovers({ entries }: { entries: ReadonlyArray<ListeningLogView> }) {
  if (entries.length === 0) {
    return null;
  }
  if (entries.length === 1) {
    return <CoverTile song={entries[0].song} size={16} radius={4} />;
  }
  if (entries.length === 2) {
    return (
      <>
        <CoverTile song={entries[0].song} size={15} radius={4} />
        <span {...props(styles.compactOverlap)}>
          <CoverTile song={entries[1].song} size={15} radius={4} />
        </span>
      </>
    );
  }
  return (
    <>
      <CoverTile song={entries[0].song} size={15} radius={4} />
      <span {...props(styles.extraCompact)}>+{entries.length - 1}</span>
    </>
  );
}

function CalendarTopBar({
  prevLabel,
  nextLabel,
  onPrev,
  onNext,
  toggleLabel,
  onToggleExpand,
}: {
  prevLabel: string;
  nextLabel: string;
  onPrev: () => void;
  onNext: () => void;
  toggleLabel: string;
  onToggleExpand: () => void;
}) {
  const { state, actions } = useListeningLog();
  const { todayKey, selectedDate, visibleYear, visibleMonthIndex } = state;

  return (
    <div {...props(styles.topRow)}>
      <div {...props(styles.headingGroup)}>
        <CalendarIcon />
        <span {...props(styles.monthText)}>
          {formatMonthHeading(visibleYear, visibleMonthIndex)}
        </span>
        <div {...props(styles.navGroup)}>
          <Button
            type="button"
            aria-label={prevLabel}
            onClick={onPrev}
            {...props(styles.navIconBtn)}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            type="button"
            aria-label={nextLabel}
            onClick={onNext}
            {...props(styles.navIconBtn)}
          >
            <ChevronRightIcon />
          </Button>
        </div>
      </div>

      <div {...props(styles.rightActions)}>
        {selectedDate !== todayKey ? (
          <Button
            type="button"
            onClick={() => actions.selectDate(todayKey)}
            {...props(styles.actionBtn)}
          >
            今日
          </Button>
        ) : null}
        <Button type="button" onClick={onToggleExpand} {...props(styles.actionBtn)}>
          {toggleLabel}
        </Button>
      </div>
    </div>
  );
}

function MonthCalendarView() {
  const { state, actions } = useListeningLog();
  const { todayKey, selectedDate, visibleYear, visibleMonthIndex, logsByDate } = state;

  const monthCells = useMemo(
    () => buildMonthGrid(visibleYear, visibleMonthIndex, todayKey),
    [visibleYear, visibleMonthIndex, todayKey],
  );

  return (
    <>
      <CalendarTopBar
        prevLabel="前の月"
        nextLabel="次の月"
        onPrev={() => actions.shiftMonth(-1)}
        onNext={() => actions.shiftMonth(1)}
        toggleLabel="週表示"
        onToggleExpand={() => actions.setCalendarExpanded(false)}
      />
      <div
        {...props(styles.monthMatrix)}
        role="grid"
        aria-label={formatMonthHeading(visibleYear, visibleMonthIndex)}
      >
        {WEEKDAY_LABELS.map((label) => (
          <span key={label} {...props(styles.matrixWeekday)}>
            {label}
          </span>
        ))}
        {monthCells.map((cell) => {
          const entries = logsByDate.get(cell.dateKey) ?? [];
          const isSelected = cell.dateKey === selectedDate;
          return (
            <button
              key={cell.dateKey}
              type="button"
              aria-pressed={isSelected}
              aria-label={`${formatJapaneseDate(cell.dateKey)}（${entries.length}曲）`}
              onClick={() => actions.selectDate(cell.dateKey)}
              {...props(
                styles.matrixCell,
                !cell.inCurrentMonth && styles.cellMuted,
                isSelected && styles.cellSelected,
              )}
            >
              <span {...props(styles.dayBadge, cell.isToday && styles.dayToday)}>{cell.day}</span>
              <div {...props(styles.matrixVisual)}>
                <span {...props(styles.visualDesktop)}>
                  {entries.slice(0, 2).map((item, idx) => (
                    <span key={item.id} {...props(idx > 0 && styles.miniOverlap)}>
                      <CoverTile song={item.song} size={18} radius={4} />
                    </span>
                  ))}
                  {entries.length > 2 ? (
                    <span {...props(styles.extraDesktop)}>+{entries.length - 2}</span>
                  ) : null}
                </span>
                <span {...props(styles.visualCompact)}>
                  <CompactCovers entries={entries} />
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </>
  );
}

function WeekCalendarView() {
  const { state, actions } = useListeningLog();
  const { todayKey, selectedDate, logsByDate } = state;

  const weekCells = useMemo(
    () => getWeekContaining(selectedDate, todayKey),
    [selectedDate, todayKey],
  );

  return (
    <>
      <CalendarTopBar
        prevLabel="前の週"
        nextLabel="次の週"
        onPrev={() => actions.shiftWeek(-1)}
        onNext={() => actions.shiftWeek(1)}
        toggleLabel="月全体"
        onToggleExpand={() => actions.setCalendarExpanded(true)}
      />
      <div {...props(styles.weekRow)}>
        {weekCells.map((cell) => {
          const entries = logsByDate.get(cell.dateKey) ?? [];
          const isSelected = cell.dateKey === selectedDate;
          return (
            <button
              key={cell.dateKey}
              type="button"
              aria-pressed={isSelected}
              aria-label={`${formatJapaneseDate(cell.dateKey)}（${entries.length}曲）`}
              onClick={() => actions.selectDate(cell.dateKey)}
              {...props(styles.weekCell, isSelected && styles.cellSelected)}
            >
              <div {...props(styles.weekCellTop)}>
                <span {...props(styles.weekDayLabel)}>{WEEKDAY_LABELS[cell.weekday]}</span>
                <span {...props(styles.dayBadge, cell.isToday && styles.dayToday)}>{cell.day}</span>
              </div>
              <div {...props(styles.weekVisual)}>
                <span {...props(styles.visualDesktop)}>
                  {entries.slice(0, 3).map((item, idx) => (
                    <span key={item.id} {...props(idx > 0 && styles.miniOverlap)}>
                      <CoverTile song={item.song} size={20} radius={5} />
                    </span>
                  ))}
                  {entries.length > 3 ? (
                    <span {...props(styles.extraDesktop)}>+{entries.length - 3}</span>
                  ) : null}
                </span>
                <span {...props(styles.visualCompact)}>
                  <CompactCovers entries={entries} />
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </>
  );
}

export function ListeningCalendarStrip() {
  const { state } = useListeningLog();

  return (
    <section {...props(styles.strip)} aria-label="聴いた曲カレンダー">
      <div {...props(styles.inner)}>
        {state.calendarExpanded ? <MonthCalendarView /> : <WeekCalendarView />}
      </div>
    </section>
  );
}

const styles = create({
  strip: {
    flexShrink: 0,
    backgroundColor: t.surface,
    borderBottomColor: t.border,
    borderBottomStyle: "solid",
    borderBottomWidth: 1,
  },
  inner: {
    display: "flex",
    flexDirection: "column",
    gap: {
      default: 8,
      [chatMedia.compact]: 6,
    },
    width: "min(100%, 760px)",
    paddingBlock: {
      default: 10,
      [chatMedia.compact]: 8,
    },
    paddingRight: {
      default: "max(16px, env(safe-area-inset-right, 0px))",
      [chatMedia.compact]: "max(10px, env(safe-area-inset-right, 0px))",
    },
    paddingLeft: {
      default: "max(16px, env(safe-area-inset-left, 0px))",
      [chatMedia.compact]: "max(10px, env(safe-area-inset-left, 0px))",
    },
    marginBlock: 0,
    marginInline: "auto",
  },
  topRow: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    justifyContent: "space-between",
  },
  headingGroup: {
    display: "flex",
    gap: 7,
    alignItems: "center",
    minWidth: 0,
  },
  monthText: {
    fontSize: 13.5,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
  },
  navGroup: {
    display: "inline-flex",
    gap: 2,
    alignItems: "center",
  },
  navIconBtn: {
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
  rightActions: {
    display: "inline-flex",
    gap: 6,
    alignItems: "center",
  },
  actionBtn: {
    height: 26,
    paddingBlock: 0,
    paddingInline: 9,
    fontSize: 12,
    fontWeight: 500,
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
  weekRow: {
    display: "grid",
    gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
    gap: {
      default: 6,
      [chatMedia.compact]: 4,
    },
  },
  weekCell: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    minWidth: 0,
    minHeight: {
      default: 62,
      [chatMedia.compact]: 54,
    },
    padding: {
      default: 7,
      [chatMedia.compact]: 5,
    },
    overflow: "hidden",
    fontFamily: "inherit",
    color: t.text,
    textAlign: "left",
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
    borderRadius: {
      default: 10,
      [chatMedia.compact]: 9,
    },
    transition: `background-color 150ms ${motion.easeOut}, border-color 150ms ${motion.easeOut}`,
  },
  weekCellTop: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  weekDayLabel: {
    fontSize: 11,
    color: t.muted,
  },
  weekVisual: {
    display: "flex",
    alignItems: "center",
    minWidth: 0,
    minHeight: 20,
    marginTop: 4,
    overflow: "hidden",
  },
  monthMatrix: {
    display: "grid",
    gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
    gap: {
      default: 5,
      [chatMedia.compact]: 4,
    },
  },
  matrixWeekday: {
    paddingInline: 4,
    fontSize: 10.5,
    fontWeight: 600,
    color: t.muted,
    textAlign: {
      default: "left",
      [chatMedia.compact]: "center",
    },
  },
  matrixCell: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    alignItems: {
      default: "flex-start",
      [chatMedia.compact]: "center",
    },
    minWidth: 0,
    minHeight: {
      default: 52,
      [chatMedia.compact]: 48,
    },
    padding: {
      default: 6,
      [chatMedia.compact]: 4,
    },
    overflow: "hidden",
    fontFamily: "inherit",
    color: t.text,
    textAlign: "left",
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
    borderRadius: 9,
    transition: `background-color 150ms ${motion.easeOut}, border-color 150ms ${motion.easeOut}`,
  },
  cellMuted: {
    opacity: 0.4,
  },
  cellSelected: {
    borderColor: t.primary,
    boxShadow: `0 0 0 1px ${t.primary}`,
  },
  dayBadge: {
    display: "inline-grid",
    placeItems: "center",
    minWidth: 20,
    height: 20,
    paddingInline: 3,
    fontSize: 11.5,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    borderRadius: 9999,
  },
  dayToday: {
    color: t.primaryFg,
    backgroundColor: t.primary,
  },
  matrixVisual: {
    display: "flex",
    alignItems: "center",
    justifyContent: {
      default: "flex-start",
      [chatMedia.compact]: "center",
    },
    maxWidth: "100%",
    minHeight: 18,
    overflow: "hidden",
  },
  visualDesktop: {
    display: {
      default: "inline-flex",
      [chatMedia.compact]: "none",
    },
    alignItems: "center",
  },
  visualCompact: {
    display: {
      default: "none",
      [chatMedia.compact]: "inline-flex",
    },
    alignItems: "center",
    minWidth: 0,
  },
  miniOverlap: {
    marginLeft: -6,
  },
  compactOverlap: {
    marginLeft: -5,
  },
  extraDesktop: {
    marginLeft: 4,
    fontSize: 10,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    color: t.muted,
  },
  extraCompact: {
    marginLeft: 3,
    fontSize: 9.5,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    color: t.muted,
    whiteSpace: "nowrap",
  },
  coverImg: {
    display: "block",
    flexShrink: 0,
    objectFit: "cover",
  },
  coverFallback: {
    display: "grid",
    flexShrink: 0,
    placeItems: "center",
    color: t.muted,
    backgroundColor: t.hover,
  },
});
