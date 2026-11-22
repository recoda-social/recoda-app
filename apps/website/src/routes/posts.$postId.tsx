import { createFileRoute, Link } from "@tanstack/react-router";
import { useAuth } from "#/auth/AuthContext";
import { PostDetailScreen } from "#/posts/PostDetailScreen";

export const Route = createFileRoute("/posts/$postId")({ component: PostDetailRoute });

function PostDetailRoute() {
  const { postId } = Route.useParams();
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
          <p style={{ margin: 0, fontSize: 15 }}>投稿を表示するにはログインが必要です。</p>
          <Link to="/" style={{ fontSize: 14 }}>
            トップページへ戻る
          </Link>
        </div>
      </main>
    );
  }

  return <PostDetailScreen postId={postId} />;
}
