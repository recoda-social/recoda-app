import { createFileRoute, Link } from "@tanstack/react-router";
import { useAuth } from "#/auth/AuthContext";
import { ProfileScreen } from "#/profile/ProfileScreen";

export const Route = createFileRoute("/profile")({ component: ProfileRoute });

function ProfileRoute() {
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
    return (
      <main
        style={{
          minHeight: "100dvh",
          display: "grid",
          placeItems: "center",
          padding: 24,
          textAlign: "center",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
          <p style={{ margin: 0, fontSize: 15 }}>プロフィールを表示するにはログインが必要です。</p>
          <Link to="/" style={{ fontSize: 14 }}>
            トップページへ戻る
          </Link>
        </div>
      </main>
    );
  }

  return <ProfileScreen user={state.user} />;
}
