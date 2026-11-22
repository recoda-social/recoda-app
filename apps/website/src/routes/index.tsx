import { createFileRoute } from "@tanstack/react-router";
import { useAuth } from "#/auth/AuthContext";
import { ChatWorkspace } from "#/chat/ChatWorkspace";
import { Landing } from "#/landing/Landing";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  const { state } = useAuth();

  if (state.isPending) {
    return (
      <main
        style={{
          minHeight: "100dvh",
          display: "grid",
          placeItems: "center",
          fontSize: 14,
          opacity: 0.7,
        }}
      >
        読み込み中…
      </main>
    );
  }

  if (!state.user) {
    return <Landing />;
  }

  return <ChatWorkspace user={state.user} />;
}
