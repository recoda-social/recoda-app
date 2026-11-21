import { describe, expect, it } from "vite-plus/test";
import {
  buildMonthGrid,
  formatJapaneseDate,
  getQueryRangeForMonth,
  getWeekContaining,
  shiftDateKey,
} from "./calendar-utils";

describe("calendar-utils", () => {
  it("builds a complete Sunday-to-Saturday month grid", () => {
    const cells = buildMonthGrid(2026, 10, "2026-11-22");
    expect(cells.length % 7).toBe(0);
    expect(cells[0]?.dateKey).toBe("2026-11-01");
    expect(cells.find((cell) => cell.isToday)?.dateKey).toBe("2026-11-22");
  });

  it("shifts dates across month boundaries", () => {
    expect(shiftDateKey("2026-11-01", -1)).toBe("2026-10-31");
    expect(shiftDateKey("2026-11-30", 1)).toBe("2026-12-01");
  });

  it("returns the 7-day week containing the anchor date", () => {
    const week = getWeekContaining("2026-11-22", "2026-11-22");
    expect(week).toHaveLength(7);
    expect(week[0]?.dateKey).toBe("2026-11-22");
    expect(week[6]?.dateKey).toBe("2026-11-28");
  });

  it("formats Japanese date labels and month query ranges", () => {
    expect(formatJapaneseDate("2026-11-22")).toBe("11月22日（日）");
    expect(getQueryRangeForMonth(2026, 10)).toEqual({
      from: "2026-11-01",
      to: "2026-12-05",
    });
  });
});
