import { props } from "@stylexjs/stylex";
import {
  AlertDialog,
  AlertDialogBackdrop,
  AlertDialogClose,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@ui-sdk/react/alert-dialog";
import { Button } from "@ui-sdk/react/button";
import { createContext, use } from "react";
import type { ReactNode } from "react";
import { artworkUrl } from "#/lib/musickit";
import type { FavoriteDecoration } from "#/lib/profile";
import { favoriteAccentStyles } from "./favorite-colors";
import { favoriteStyles as styles } from "./favorite-styles";
import { ChevronDownIcon, ChevronUpIcon, TrashIcon } from "./icons";

type FavoriteRowContextValue = {
  state: {
    index: number;
    total: number;
    saving: boolean;
  };
  actions: {
    move: (direction: -1 | 1) => void;
    remove: () => Promise<void>;
  };
  meta: {
    label: string;
    noun: string;
  };
};

const FavoriteRowContext = createContext<FavoriteRowContextValue | null>(null);

function useFavoriteRow() {
  const value = use(FavoriteRowContext);
  if (!value) {
    throw new Error("FavoriteRow の部品は FavoriteRow.Root の中で使う");
  }
  return value;
}

function Root({
  color,
  index,
  total,
  saving,
  label,
  noun,
  onMove,
  onRemove,
  children,
}: {
  color: FavoriteDecoration["color"];
  index: number;
  total: number;
  saving: boolean;
  label: string;
  noun: string;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => Promise<void>;
  children: ReactNode;
}) {
  const accentStyle = color ? favoriteAccentStyles[color] : null;
  return (
    <FavoriteRowContext
      value={{
        state: { index, total, saving },
        actions: { move: onMove, remove: onRemove },
        meta: { label, noun },
      }}
    >
      <li {...props(styles.card, accentStyle)}>{children}</li>
    </FavoriteRowContext>
  );
}

function Main({ children }: { children: ReactNode }) {
  return <div {...props(styles.cardMain)}>{children}</div>;
}

function Rank() {
  const { state } = useFavoriteRow();
  return (
    <span {...props(styles.rank)} aria-label={`${state.index + 1} 番目`}>
      {String(state.index + 1).padStart(2, "0")}
    </span>
  );
}

type ArtProps = {
  src: string | null | undefined;
  fallback: ReactNode;
  emoji: string | null;
};

function SquareArt({ src, fallback, emoji }: ArtProps) {
  return (
    <div {...props(styles.artWrap)}>
      {src ? (
        <img src={artworkUrl(src, 112)} alt="" width={52} height={52} {...props(styles.cardArt)} />
      ) : (
        <div {...props(styles.cardArt, styles.artFallback)} aria-hidden="true">
          {fallback}
        </div>
      )}
      {emoji ? (
        <span {...props(styles.emojiBadge)} aria-hidden="true">
          {emoji}
        </span>
      ) : null}
    </div>
  );
}

function RoundArt({ src, fallback, emoji }: ArtProps) {
  return (
    <div {...props(styles.artWrap)}>
      {src ? (
        <img
          src={artworkUrl(src, 112)}
          alt=""
          width={52}
          height={52}
          {...props(styles.cardArt, styles.roundCardArt)}
        />
      ) : (
        <div {...props(styles.cardArt, styles.artFallback, styles.roundCardArt)} aria-hidden="true">
          {fallback}
        </div>
      )}
      {emoji ? (
        <span {...props(styles.emojiBadge, styles.roundEmojiBadge)} aria-hidden="true">
          {emoji}
        </span>
      ) : null}
    </div>
  );
}

function Meta({ children }: { children: ReactNode }) {
  return <div {...props(styles.meta)}>{children}</div>;
}

function Actions({ children }: { children: ReactNode }) {
  return <div {...props(styles.rowActions)}>{children}</div>;
}

function MoveButtons() {
  const { state, actions, meta } = useFavoriteRow();
  return (
    <>
      <Button
        type="button"
        disabled={state.index === 0 || state.saving}
        aria-label={`「${meta.label}」を上に移動`}
        onClick={() => actions.move(-1)}
        {...props(styles.iconBtn)}
      >
        <ChevronUpIcon />
      </Button>
      <Button
        type="button"
        disabled={state.index === state.total - 1 || state.saving}
        aria-label={`「${meta.label}」を下に移動`}
        onClick={() => actions.move(1)}
        {...props(styles.iconBtn)}
      >
        <ChevronDownIcon />
      </Button>
    </>
  );
}

function RemoveDialog() {
  const { state, actions, meta } = useFavoriteRow();
  return (
    <AlertDialog pendingAnnouncement={`${meta.noun}から削除しています`}>
      <AlertDialogTrigger
        disabled={state.saving}
        aria-label={`「${meta.label}」を${meta.noun}から削除`}
        {...props(styles.deleteBtn)}
      >
        <TrashIcon />
      </AlertDialogTrigger>
      <AlertDialogBackdrop {...props(styles.backdrop)} />
      <AlertDialogContent {...props(styles.alertDialog)}>
        <AlertDialogTitle {...props(styles.dialogTitle)}>
          {meta.noun}から外しますか？
        </AlertDialogTitle>
        <AlertDialogDescription {...props(styles.dialogDesc)}>
          「{meta.label}」を{meta.noun}リストから削除します。
        </AlertDialogDescription>
        <div {...props(styles.dialogFooter)}>
          <AlertDialogClose {...props(styles.secondaryBtn)}>キャンセル</AlertDialogClose>
          <AlertDialogClose
            value="confirm"
            data-agent-confirm={`「${meta.label}」を${meta.noun}から削除します`}
            action={actions.remove}
            {...props(styles.dangerBtn)}
          >
            リストから外す
          </AlertDialogClose>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function Comment({ children }: { children: string | null }) {
  return children ? <p {...props(styles.comment)}>{children}</p> : null;
}

/** お気に入り 1 件分の行。Root が共有コンテキストを提供し、内部の部品を組み合わせる。 */
export const FavoriteRow = {
  Root,
  Main,
  Rank,
  SquareArt,
  RoundArt,
  Meta,
  Actions,
  MoveButtons,
  RemoveDialog,
  Comment,
};
