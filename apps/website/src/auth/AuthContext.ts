import { createContext, use } from "react";
import type { Me } from "#/lib/auth";

type AuthState = {
  user: Me | null;
  isPending: boolean;
};

type AuthActions = {
  signInWithGoogle: () => void;
  signOut: () => Promise<void>;
};

type AuthMeta = Record<string, never>;

export type AuthContextValue = {
  state: AuthState;
  actions: AuthActions;
  meta: AuthMeta;
};

export const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth() {
  const value = use(AuthContext);
  if (!value) {
    throw new Error("useAuth は AuthProvider の中で使う");
  }
  return value;
}
