export type CalendarCell = {
  dateKey: string;
  day: number;
  inCurrentMonth: boolean;
  isToday: boolean;
  weekday: number;
};

export const WEEKDAY_LABELS = ["日", "月", "火", "水", "木", "金", "土"] as const;

function formatDateKey(year: number, monthIndex: number, day: number): string {
  const m = String(monthIndex + 1).padStart(2, "0");
  const d = String(day).padStart(2, "0");
  return `${year}-${m}-${d}`;
}

export function getTodayDateKey(now = new Date()): string {
  return formatDateKey(now.getFullYear(), now.getMonth(), now.getDate());
}

export function parseDateKey(dateKey: string) {
  const [y, m, d] = dateKey.split("-").map(Number);
  return {
    year: y || new Date().getFullYear(),
    monthIndex: (m || 1) - 1,
    day: d || 1,
  };
}

export function shiftDateKey(dateKey: string, deltaDays: number): string {
  const { year, monthIndex, day } = parseDateKey(dateKey);
  const next = new Date(year, monthIndex, day + deltaDays);
  return formatDateKey(next.getFullYear(), next.getMonth(), next.getDate());
}

export function formatJapaneseDate(dateKey: string): string {
  const { year, monthIndex, day } = parseDateKey(dateKey);
  const date = new Date(year, monthIndex, day);
  const weekday = WEEKDAY_LABELS[date.getDay()] ?? "";
  return `${monthIndex + 1}月${day}日（${weekday}）`;
}

export function formatMonthHeading(year: number, monthIndex: number): string {
  return `${year}年${monthIndex + 1}月`;
}

export function buildMonthGrid(
  year: number,
  monthIndex: number,
  todayKey: string,
): Array<CalendarCell> {
  const firstDay = new Date(year, monthIndex, 1);
  const startWeekday = firstDay.getDay();
  const daysInMonth = new Date(year, monthIndex + 1, 0).getDate();
  const prevMonthDays = new Date(year, monthIndex, 0).getDate();
  const cells: Array<CalendarCell> = [];

  for (let i = startWeekday - 1; i >= 0; i -= 1) {
    const day = prevMonthDays - i;
    const prevDate = new Date(year, monthIndex - 1, day);
    const dateKey = formatDateKey(prevDate.getFullYear(), prevDate.getMonth(), day);
    cells.push({
      dateKey,
      day,
      inCurrentMonth: false,
      isToday: dateKey === todayKey,
      weekday: prevDate.getDay(),
    });
  }

  for (let day = 1; day <= daysInMonth; day += 1) {
    const date = new Date(year, monthIndex, day);
    const dateKey = formatDateKey(year, monthIndex, day);
    cells.push({
      dateKey,
      day,
      inCurrentMonth: true,
      isToday: dateKey === todayKey,
      weekday: date.getDay(),
    });
  }

  const remaining = (7 - (cells.length % 7)) % 7;
  for (let day = 1; day <= remaining; day += 1) {
    const nextDate = new Date(year, monthIndex + 1, day);
    const dateKey = formatDateKey(nextDate.getFullYear(), nextDate.getMonth(), day);
    cells.push({
      dateKey,
      day,
      inCurrentMonth: false,
      isToday: dateKey === todayKey,
      weekday: nextDate.getDay(),
    });
  }

  return cells;
}

export function getWeekContaining(dateKey: string, todayKey: string): Array<CalendarCell> {
  const { year, monthIndex, day } = parseDateKey(dateKey);
  const anchor = new Date(year, monthIndex, day);
  const start = new Date(anchor);
  start.setDate(anchor.getDate() - anchor.getDay());

  const cells: Array<CalendarCell> = [];
  for (let i = 0; i < 7; i += 1) {
    const cur = new Date(start);
    cur.setDate(start.getDate() + i);
    const key = formatDateKey(cur.getFullYear(), cur.getMonth(), cur.getDate());
    cells.push({
      dateKey: key,
      day: cur.getDate(),
      inCurrentMonth: cur.getMonth() === anchor.getMonth(),
      isToday: key === todayKey,
      weekday: cur.getDay(),
    });
  }
  return cells;
}

export function getQueryRangeForMonth(year: number, monthIndex: number) {
  const cells = buildMonthGrid(year, monthIndex, "");
  return {
    from: cells[0]?.dateKey ?? formatDateKey(year, monthIndex, 1),
    to: cells.at(-1)?.dateKey ?? formatDateKey(year, monthIndex, 28),
  };
}
