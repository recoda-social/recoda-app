// @vitest-environment jsdom
import "#/test/setup";
import { describe, expect, it } from "vite-plus/test";
import { initAgentBridge } from "./agent";

type MockTool = {
  name: string;
  execute: (input: Record<string, never>) => Promise<ReadonlyArray<Record<string, string>>>;
};

describe("agent bridge WebMCP registration", () => {
  it("registers snapshot, act, and query tools on document.modelContext", async () => {
    const tools = new Map<string, MockTool>();
    Object.assign(document, {
      modelContext: {
        registerTool: async (tool: MockTool) => {
          tools.set(tool.name, tool);
        },
      },
    });

    const unregister = initAgentBridge();
    try {
      const snapshotTool = tools.get("snapshot");
      expect(snapshotTool).toBeDefined();
      expect(await snapshotTool?.execute({})).toBeInstanceOf(Array);
    } finally {
      unregister();
    }
  });
});
