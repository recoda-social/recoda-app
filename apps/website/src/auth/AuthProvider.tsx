import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useMemo } from "react";
import type { ReactNode } from "react";
import { meQuery } from "#/lib/auth";
import { authClient } from "#/lib/auth-client";
import { usePreviewPlayer } from "#/songs/preview-player";
import { AuthContext } from "./AuthContext";

const EMPTY_META: Record<string, never> = {};

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const player = usePreviewPlayer();
  const me = useQuery(meQuery);

  const { close: closePlayer } = player.actions;

  const signInWithGoogle = useCallback(() => {
    void authClient.signIn.social({ provider: "google", callbackURL: "/" });
  }, []);

  const signOut = useCallback(async () => {
    closePlayer();
    await authClient.signOut();
    await queryClient.invalidateQueries({ queryKey: meQuery.queryKey });
    await navigate({ to: "/" });
  }, [closePlayer, navigate, queryClient]);

  const actions = useMemo(
    () => ({
      signInWithGoogle,
      signOut,
    }),
    [signInWithGoogle, signOut],
  );

  return (
    <AuthContext
      value={{
        state: {
          user: me.data ?? null,
          isPending: me.isPending,
        },
        actions,
        meta: EMPTY_META,
      }}
    >
      {children}
    </AuthContext>
  );
}
