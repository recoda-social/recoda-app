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
import { useMemo } from "react";
import type { ListeningLogView } from "#/lib/listening-logs";
import { artworkUrl } from "#/lib/musickit";
import { motion } from "#/motion.stylex";
import { CalendarIcon, ChevronLeftIcon, ChevronRightIcon, MusicNoteIcon } from "../icons";
import { chatMedia, chatTokens as t } from "../tokens.stylex";
import {
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
  const visibleMonthIso = `${visibleYear}-${String(visibleMonthIndex + 1).padStart(2, "0")}`;

  return (
    <Calendar
      mode="single"
      locale="ja-JP"
      weekStartsOn={0}
      fixedWeeks
      showOutsideDays
      today={todayKey}
      value={selectedDate}
      onValueChange={(next) => {
        if (typeof next === "string") {
          actions.selectDate(next);
        }
      }}
      month={visibleMonthIso}
      onMonthChange={(next) => {
        const [yearStr, monthStr] = next.split("-");
        const nextYear = Number(yearStr);
        const nextMonthIndex = Number(monthStr) - 1;
        if (Number.isFinite(nextYear) && Number.isFinite(nextMonthIndex)) {
          actions.shiftMonth((nextYear - visibleYear) * 12 + (nextMonthIndex - visibleMonthIndex));
        }
      }}
      {...props(styles.monthCalendar)}
    >
      <div {...props(styles.topRow)}>
        <div {...props(styles.headingGroup)}>
          <CalendarIcon />
          <CalendarHeading {...props(styles.monthText)} />
          <div {...props(styles.navGroup)}>
            <CalendarPrevious aria-label="前の月" {...props(styles.navIconBtn)}>
              <ChevronLeftIcon />
            </CalendarPrevious>
            <CalendarNext aria-label="次の月" {...props(styles.navIconBtn)}>
              <ChevronRightIcon />
            </CalendarNext>
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
          <Button
            type="button"
            onClick={() => actions.setCalendarExpanded(false)}
            {...props(styles.actionBtn)}
          >
            週表示
          </Button>
        </div>
      </div>

      <CalendarGrid {...props(styles.monthMatrix)}>
        <CalendarWeekdays>
          {(index) => <CalendarWeekday index={index} {...props(styles.matrixWeekday)} />}
        </CalendarWeekdays>
        <CalendarWeeks>
          {(cell) => {
            const entries = logsByDate.get(cell.date) ?? [];
            const dayNum = Number(cell.date.slice(8));
            return (
              <CalendarDay
                value={cell.date}
                aria-label={`${formatJapaneseDate(cell.date)}（${entries.length}曲）`}
                {...props(styles.matrixDayTd)}
              >
                {(dayState) => (
                  <div
                    {...props(
                      styles.matrixCell,
                      dayState.outside && styles.cellMuted,
                      dayState.selected && styles.cellSelected,
                    )}
                  >
                    <span {...props(styles.dayBadge, dayState.today && styles.dayToday)}>
                      {dayNum}
                    </span>
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
                  </div>
                )}
              </CalendarDay>
            );
          }}
        </CalendarWeeks>
      </CalendarGrid>
    </Calendar>
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
            <Button
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
            </Button>
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
  monthCalendar: {
    display: "flex",
    flexDirection: "column",
    gap: {
      default: 8,
      [chatMedia.compact]: 6,
    },
    width: "100%",
    padding: 0,
    backgroundColor: "transparent",
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 0,
  },
  monthText: {
    margin: 0,
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
    gap: 0,
    alignItems: "stretch",
    justifyContent: "space-between",
    minWidth: 0,
    height: "auto",
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
    transition: "background-color 150ms ease, border-color 150ms ease",
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
    justifyContent: {
      default: "flex-start",
      [chatMedia.compact]: "center",
    },
    minWidth: 0,
    minHeight: 20,
    marginTop: 4,
    overflow: "hidden",
  },
  monthMatrix: {
    width: "100%",
    tableLayout: "fixed",
    borderSpacing: {
      default: 5,
      [chatMedia.compact]: 4,
    },
    borderCollapse: "separate",
  },
  matrixWeekday: {
    width: "auto",
    height: "auto",
    paddingBlock: 2,
    paddingInline: 4,
    fontSize: 10.5,
    fontWeight: 600,
    color: t.muted,
    textAlign: {
      default: "left",
      [chatMedia.compact]: "center",
    },
  },
  matrixDayTd: {
    width: "auto",
    height: "auto",
    padding: 0,
    verticalAlign: "top",
    backgroundColor: "transparent",
    borderRadius: 0,
  },
  matrixCell: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    alignItems: {
      default: "flex-start",
      [chatMedia.compact]: "center",
    },
    width: "100%",
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
    transition: "background-color 150ms ease, border-color 150ms ease",
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
