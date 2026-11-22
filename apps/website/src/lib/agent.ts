import { createAgent, registerTools } from "@ui-sdk/react/agent";

const agent = createAgent({
  allow: (step) => (step.confirm ? window.confirm(step.confirm) : true),
});

export function initAgentBridge() {
  // allow の確認は createAgent 側で 1 回だけ行うため、registerTools 側は素通しにする。
  return registerTools(agent, { allow: () => true });
}
