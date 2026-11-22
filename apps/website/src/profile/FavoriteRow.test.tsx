// @vitest-environment jsdom
import "#/test/setup";
import { render } from "@testing-library/react";
import { createAgent } from "@ui-sdk/react/agent";
import { describe, expect, it, vi } from "vite-plus/test";
import { FavoriteRow } from "./FavoriteRow";

let approveConfirm = false;
const agent = createAgent({
  allow: (step) => !step.confirm || approveConfirm,
});

describe("FavoriteRow agent readiness", () => {
  it("exposes named controls and gates removal behind data-agent-confirm", async () => {
    const onMove = vi.fn();
    const onRemove = vi.fn(async () => {});

    render(
      <ul>
        <FavoriteRow.Root
          color={null}
          index={1}
          total={3}
          saving={false}
          label="夜に駆ける"
          noun="好きな曲"
          onMove={onMove}
          onRemove={onRemove}
        >
          <FavoriteRow.Main>
            <FavoriteRow.Rank />
            <FavoriteRow.Meta>
              <span>夜に駆ける</span>
            </FavoriteRow.Meta>
            <FavoriteRow.Actions>
              <FavoriteRow.MoveButtons />
              <FavoriteRow.RemoveDialog />
            </FavoriteRow.Actions>
          </FavoriteRow.Main>
        </FavoriteRow.Root>
      </ul>,
    );

    const components = agent
      .snapshot()
      .filter((entry) => !entry.offPage && entry.component !== "more");
    for (const entry of components) {
      expect(entry.label, `${entry.component} ${entry.ref} has no name`).toBeTruthy();
    }

    const labels = components.map((entry) => entry.label);
    expect(labels).toContain("「夜に駆ける」を上に移動");
    expect(labels).toContain("「夜に駆ける」を下に移動");
    expect(labels).toContain("「夜に駆ける」を好きな曲から削除");

    const moveResult = await agent.act({
      steps: [{ ref: "「夜に駆ける」を上に移動", action: "press" }],
    });
    expect(moveResult.ok, moveResult.error?.message).toBe(true);
    expect(onMove).toHaveBeenCalledWith(-1);

    const removeRequest = {
      steps: [
        {
          ref: "「夜に駆ける」を好きな曲から削除",
          action: "press",
          args: ["リストから外す"],
        },
      ],
    };

    approveConfirm = false;
    const refused = await agent.act(removeRequest);
    expect(refused.ok).toBe(false);
    expect(onRemove).not.toHaveBeenCalled();

    approveConfirm = true;
    const approved = await agent.act(removeRequest);
    expect(approved.ok, approved.error?.message).toBe(true);
    expect(onRemove).toHaveBeenCalledTimes(1);
  });
});
