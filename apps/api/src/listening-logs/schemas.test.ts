import { describe, expect, it } from "vite-plus/test";
import { createListeningLogInput, listListeningLogsQuery } from "./schemas.ts";

describe("listening-logs schemas", () => {
  it("accepts valid listening log input and uppercases ISRC", () => {
    const parsed = createListeningLogInput.safeParse({
      songId: "123456",
      isrc: "jpxx02000001",
      listenedOn: "2026-11-22",
    });
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.isrc).toBe("JPXX02000001");
    }
  });

  it("rejects impossible calendar dates", () => {
    const parsed = createListeningLogInput.safeParse({
      songId: "123456",
      isrc: "JPXX02000001",
      listenedOn: "2026-02-30",
    });
    expect(parsed.success).toBe(false);
  });

  it("validates query range bounds", () => {
    expect(listListeningLogsQuery.safeParse({ from: "2026-11-01", to: "2026-12-05" }).success).toBe(
      true,
    );
    expect(listListeningLogsQuery.safeParse({ from: "2026-12-05", to: "2026-11-01" }).success).toBe(
      false,
    );
    expect(listListeningLogsQuery.safeParse({ from: "2026-01-01", to: "2026-12-31" }).success).toBe(
      false,
    );
  });
});
