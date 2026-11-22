import { props } from "@stylexjs/stylex";
import { Button } from "@ui-sdk/react/button";
import {
  Dialog,
  DialogBackdrop,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@ui-sdk/react/dialog";
import { Field, FieldLabel } from "@ui-sdk/react/field";
import { Input } from "@ui-sdk/react/input";
import { createContext, use, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { CloseIcon } from "#/chat/icons";
import type { FavoriteDecoration } from "#/lib/profile";
import { FavoriteDecorationFields, validateDecoration } from "./FavoriteDecorationEditor";
import { favoriteStyles as styles } from "./favorite-styles";
import { EditIcon, SearchIcon } from "./icons";

const EMPTY_DECORATION: FavoriteDecoration = { comment: null, emoji: null, color: null };
const EMPTY_META: Record<string, never> = {};

const normalizeDecoration = (decoration: FavoriteDecoration): FavoriteDecoration => ({
  comment: decoration.comment?.trim() || null,
  emoji: decoration.emoji?.trim() || null,
  color: decoration.color,
});

type FavoriteDialogContextValue = {
  state: {
    open: boolean;
    decoration: FavoriteDecoration;
    canSave: boolean;
    saving: boolean;
  };
  actions: {
    onOpenChange: (next: boolean) => void;
    setDecoration: (next: FavoriteDecoration) => void;
    submit: (event: { preventDefault: () => void }) => void;
  };
  meta: {
    fieldPrefix: string;
  };
};

const FavoriteDialogContext = createContext<FavoriteDialogContextValue | null>(null);

function useFavoriteDialog() {
  const value = use(FavoriteDialogContext);
  if (!value) {
    throw new Error("FavoriteDialog の部品は Provider の中で使う");
  }
  return value;
}

type FavoriteSearchUiContextValue = {
  state: {
    query: string;
    resultCount: number;
    searching: boolean;
    error: string | null;
  };
  actions: {
    setQuery: (value: string) => void;
    clearSelected: () => void;
  };
  meta: Record<string, never>;
};

const FavoriteSearchUiContext = createContext<FavoriteSearchUiContextValue | null>(null);

function useFavoriteSearchUi() {
  const value = use(FavoriteSearchUiContext);
  if (!value) {
    throw new Error("FavoriteDialog の検索部品は AddProvider の中で使う");
  }
  return value;
}

type TypedFavoriteSearchContextValue<T> = {
  state: {
    results: Array<T>;
    selected: T | null;
  };
  actions: {
    select: (item: T) => void;
    clearSelected: () => void;
  };
  meta: Record<string, never>;
};

/** 検索対象の型 T ごとに型安全な AddProvider と Hook を生成する。 */
export function createFavoriteAddContext<T>() {
  const ItemContext = createContext<TypedFavoriteSearchContextValue<T> | null>(null);

  function useFavoriteAdd() {
    const value = use(ItemContext);
    if (!value) {
      throw new Error("useFavoriteAdd は AddProvider の中で使う");
    }
    return value;
  }

  function AddProvider({
    searchFn,
    isSelectable = () => true,
    saving,
    fieldPrefix,
    onAdd,
    children,
  }: {
    searchFn: (query: string) => Promise<Array<T>>;
    isSelectable?: (item: T) => boolean;
    saving: boolean;
    fieldPrefix: string;
    onAdd: (item: T, decoration: FavoriteDecoration) => Promise<void>;
    children: ReactNode;
  }) {
    const [open, setOpen] = useState(false);
    const [selected, setSelected] = useState<T | null>(null);
    const [decoration, setDecoration] = useState<FavoriteDecoration>(EMPTY_DECORATION);
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<Array<T>>([]);
    const [searching, setSearching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const savedScrollYRef = useRef<number | null>(null);

    const trimmedQuery = open ? query.trim() : "";

    useEffect(() => {
      if (!trimmedQuery) {
        return;
      }

      let cancelled = false;

      const timer = window.setTimeout(() => {
        setSearching(true);
        setError(null);
        searchFn(trimmedQuery)
          .then((found) => {
            if (cancelled) {
              return;
            }
            setResults(found);
            setSearching(false);
          })
          .catch((error_: unknown) => {
            if (cancelled) {
              return;
            }
            setResults([]);
            setSearching(false);
            setError(error_ instanceof Error ? error_.message : "検索に失敗しました");
          });
      }, 220);

      return () => {
        cancelled = true;
        window.clearTimeout(timer);
      };
    }, [trimmedQuery, searchFn]);

    const onOpenChange = (next: boolean) => {
      if (next) {
        savedScrollYRef.current = window.scrollY;
      } else {
        setQuery("");
        setResults([]);
        setSearching(false);
        setError(null);
        setSelected(null);
        setDecoration(EMPTY_DECORATION);
        if (savedScrollYRef.current !== null) {
          const targetY = savedScrollYRef.current;
          requestAnimationFrame(() => window.scrollTo(0, targetY));
        }
      }
      setOpen(next);
    };

    const onOpenChangeComplete = (isOpen: boolean) => {
      if (!isOpen && savedScrollYRef.current !== null) {
        const targetY = savedScrollYRef.current;
        savedScrollYRef.current = null;
        window.scrollTo(0, targetY);
      }
    };

    const { emojiError, commentError } = validateDecoration(decoration);
    const canSave =
      Boolean(selected && isSelectable(selected)) && !emojiError && !commentError && !saving;

    const submit = (event: { preventDefault: () => void }) => {
      event.preventDefault();
      if (!canSave || !selected) {
        return;
      }
      void onAdd(selected, normalizeDecoration(decoration))
        .then(() => onOpenChange(false))
        .catch(() => {
          // エラー通知は Provider 側の toast で行う。
        });
    };

    const visibleResults = trimmedQuery ? results : [];
    const clearSelected = () => setSelected(null);

    return (
      <FavoriteDialogContext
        value={{
          state: { open, decoration, canSave, saving },
          actions: { onOpenChange, setDecoration, submit },
          meta: { fieldPrefix },
        }}
      >
        <FavoriteSearchUiContext
          value={{
            state: {
              query,
              resultCount: visibleResults.length,
              searching: trimmedQuery ? searching : false,
              error: trimmedQuery ? error : null,
            },
            actions: { setQuery, clearSelected },
            meta: EMPTY_META,
          }}
        >
          <ItemContext
            value={{
              state: { results: visibleResults, selected },
              actions: { select: setSelected, clearSelected },
              meta: EMPTY_META,
            }}
          >
            <Dialog
              open={open}
              onOpenChange={onOpenChange}
              onOpenChangeComplete={onOpenChangeComplete}
              modal="trap-focus"
            >
              {children}
            </Dialog>
          </ItemContext>
        </FavoriteSearchUiContext>
      </FavoriteDialogContext>
    );
  }

  return { AddProvider, useFavoriteAdd };
}

/** 既存のお気に入り編集ダイアログ用の Provider。共通の FavoriteDialogContext を実装する。 */
function EditProvider({
  initial,
  saving,
  fieldPrefix,
  onSave,
  children,
}: {
  initial: FavoriteDecoration;
  saving: boolean;
  fieldPrefix: string;
  onSave: (decoration: FavoriteDecoration) => Promise<void>;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [decoration, setDecoration] = useState<FavoriteDecoration>(initial);
  const savedScrollYRef = useRef<number | null>(null);

  const onOpenChange = (next: boolean) => {
    if (next) {
      savedScrollYRef.current = window.scrollY;
      setDecoration(initial);
    } else if (savedScrollYRef.current !== null) {
      const targetY = savedScrollYRef.current;
      requestAnimationFrame(() => window.scrollTo(0, targetY));
    }
    setOpen(next);
  };

  const onOpenChangeComplete = (isOpen: boolean) => {
    if (!isOpen && savedScrollYRef.current !== null) {
      const targetY = savedScrollYRef.current;
      savedScrollYRef.current = null;
      window.scrollTo(0, targetY);
    }
  };

  const { emojiError, commentError } = validateDecoration(decoration);
  const canSave = !emojiError && !commentError && !saving;

  const submit = (event: { preventDefault: () => void }) => {
    event.preventDefault();
    if (!canSave) {
      return;
    }
    void onSave(normalizeDecoration(decoration))
      .then(() => setOpen(false))
      .catch(() => {
        // エラー通知は Provider 側の toast で行う。
      });
  };

  return (
    <FavoriteDialogContext
      value={{
        state: { open, decoration, canSave, saving },
        actions: { onOpenChange, setDecoration, submit },
        meta: { fieldPrefix },
      }}
    >
      <Dialog
        open={open}
        onOpenChange={onOpenChange}
        onOpenChangeComplete={onOpenChangeComplete}
        modal="trap-focus"
      >
        {children}
      </Dialog>
    </FavoriteDialogContext>
  );
}

function Content({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <>
      <DialogBackdrop {...props(styles.backdrop)} />
      <DialogContent {...props(styles.dialog)}>
        <div {...props(styles.dialogHeader)}>
          <div {...props(styles.dialogHeading)}>
            <DialogTitle {...props(styles.dialogTitle)}>{title}</DialogTitle>
            {description ? (
              <DialogDescription {...props(styles.dialogDesc)}>{description}</DialogDescription>
            ) : null}
          </div>
          <DialogClose aria-label="閉じる" {...props(styles.closeBtn)}>
            <CloseIcon />
          </DialogClose>
        </div>
        {children}
      </DialogContent>
    </>
  );
}

function Form({ children }: { children: ReactNode }) {
  const { actions } = useFavoriteDialog();
  return (
    <form {...props(styles.form)} onSubmit={actions.submit}>
      {children}
    </form>
  );
}

function DecorationFields() {
  const { state, actions, meta } = useFavoriteDialog();
  return (
    <FavoriteDecorationFields
      value={state.decoration}
      onChange={actions.setDecoration}
      fieldPrefix={meta.fieldPrefix}
    />
  );
}

function Footer({ idleLabel, savingLabel }: { idleLabel: string; savingLabel: string }) {
  const { state } = useFavoriteDialog();
  return (
    <div {...props(styles.dialogFooter)}>
      <DialogClose {...props(styles.secondaryBtn)}>キャンセル</DialogClose>
      <Button type="submit" disabled={!state.canSave} {...props(styles.primaryBtn)}>
        {state.saving ? savingLabel : idleLabel}
      </Button>
    </div>
  );
}

function SelectedBox({ resetLabel, children }: { resetLabel: string; children: ReactNode }) {
  const { actions } = useFavoriteSearchUi();
  return (
    <div {...props(styles.selectedBox)}>
      {children}
      <Button type="button" {...props(styles.secondaryBtnSm)} onClick={actions.clearSelected}>
        {resetLabel}
      </Button>
    </div>
  );
}

function SearchSection({
  name,
  label,
  placeholder,
  idleMessage,
  notFoundMessage,
  children,
}: {
  name: string;
  label: string;
  placeholder: string;
  idleMessage: string;
  notFoundMessage: string;
  children: ReactNode;
}) {
  const { state, actions } = useFavoriteSearchUi();
  const { query, resultCount, searching, error } = state;
  const status = searching
    ? "検索中…"
    : (error ??
      (query.trim() && resultCount === 0
        ? notFoundMessage
        : resultCount > 0
          ? `${resultCount} 件の候補`
          : idleMessage));

  return (
    <div {...props(styles.searchSection)}>
      <Field name={name}>
        <FieldLabel {...props(styles.srOnly)}>{label}</FieldLabel>
        <div {...props(styles.searchInputWrap)}>
          <span {...props(styles.searchIcon)} aria-hidden="true">
            <SearchIcon />
          </span>
          <Input
            value={query}
            onValueChange={actions.setQuery}
            enterKeyHint="search"
            placeholder={placeholder}
            {...props(styles.searchInput)}
          />
        </div>
      </Field>

      <div role="status" aria-live="polite" {...props(styles.searchStatus)}>
        {status}
      </div>

      {resultCount > 0 ? children : null}
    </div>
  );
}

export const FavoriteDialog = {
  EditProvider,
  Trigger: DialogTrigger,
  Content,
  Form,
  DecorationFields,
  Footer,
  SelectedBox,
  SearchSection,
};

/** 既存のお気に入りのメモ・絵文字・色を編集する明示的バリアント。 */
export function EditFavoriteDialog({
  title,
  initial,
  fieldPrefix,
  saving,
  onSave,
}: {
  title: string;
  initial: FavoriteDecoration;
  fieldPrefix: string;
  saving: boolean;
  onSave: (decoration: FavoriteDecoration) => Promise<void>;
}) {
  return (
    <FavoriteDialog.EditProvider
      initial={initial}
      saving={saving}
      fieldPrefix={fieldPrefix}
      onSave={onSave}
    >
      <FavoriteDialog.Trigger
        aria-label={`「${title}」のメモと飾りを編集`}
        {...props(styles.iconBtn)}
      >
        <EditIcon />
      </FavoriteDialog.Trigger>
      <FavoriteDialog.Content
        title={`「${title}」を編集`}
        description="ひとことメモ、絵文字バッジ、アクセントカラーを設定できます。"
      >
        <FavoriteDialog.Form>
          <FavoriteDialog.DecorationFields />
          <FavoriteDialog.Footer idleLabel="変更を保存" savingLabel="保存中…" />
        </FavoriteDialog.Form>
      </FavoriteDialog.Content>
    </FavoriteDialog.EditProvider>
  );
}
