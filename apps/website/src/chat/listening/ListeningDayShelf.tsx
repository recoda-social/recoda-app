import { create, props } from "@stylexjs/stylex";
import {
  Popover,
  PopoverClose,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@ui-sdk/react/popover";
import { useState } from "react";
import type { ListeningLogView } from "#/lib/listening-logs";
import { artworkUrl } from "#/lib/musickit";
import { useMediaQuery } from "#/lib/use-media-query";
import { motion } from "#/motion.stylex";
import { MusicNoteIcon } from "../icons";
import { chatMedia, chatTokens as t } from "../tokens.stylex";
import { BottomSheetDrawer } from "./BottomSheetDrawer";
import { formatJapaneseDate } from "./calendar-utils";
import { useListeningLog } from "./ListeningLogContext";

function SongArt({
  song,
  size,
  radius,
}: {
  song: ListeningLogView["song"];
  size: number;
  radius: number;
}) {
  if (song.artworkUrl) {
    return (
      <img
        src={artworkUrl(song.artworkUrl, size * 2)}
        alt=""
        width={size}
        height={size}
        loading="lazy"
        {...props(styles.artImg)}
        style={{ width: size, height: size, borderRadius: radius }}
      />
    );
  }
  return (
    <span
      aria-hidden="true"
      {...props(styles.artFallback)}
      style={{ width: size, height: size, borderRadius: radius }}
    >
      <MusicNoteIcon />
    </span>
  );
}

type LoggedSongPillProps = {
  log: ListeningLogView;
};

function PillContent({ song }: { song: ListeningLogView["song"] }) {
  const title = song.title ?? "（曲名なし）";
  const artist = song.artistName ?? "";
  return (
    <>
      <SongArt song={song} size={32} radius={7} />
      <span {...props(styles.metaCol)}>
        <span {...props(styles.metaTitle)}>{title}</span>
        {artist ? <span {...props(styles.metaArtist)}>{artist}</span> : null}
      </span>
    </>
  );
}

function MobileLoggedSongPill({ log }: LoggedSongPillProps) {
  const { actions } = useListeningLog();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const title = log.song.title ?? "（曲名なし）";
  const artist = log.song.artistName ?? "";
  const triggerLabel = `${title}${artist ? `（${artist}）` : ""}のメニューを開く`;

  return (
    <BottomSheetDrawer.Root open={drawerOpen} onOpenChange={setDrawerOpen}>
      <BottomSheetDrawer.Trigger aria-label={triggerLabel} {...props(styles.pill)}>
        <PillContent song={log.song} />
      </BottomSheetDrawer.Trigger>
      <BottomSheetDrawer.Content>
        <div {...props(styles.actionHead)}>
          <SongArt song={log.song} size={36} radius={7} />
          <div {...props(styles.metaCol)}>
            <BottomSheetDrawer.Title {...props(styles.metaTitle)}>{title}</BottomSheetDrawer.Title>
            {artist ? <span {...props(styles.metaArtist)}>{artist}</span> : null}
          </div>
        </div>
        <div {...props(styles.actionButtons)}>
          <BottomSheetDrawer.Close
            onClick={() => actions.writeReviewForLog(log)}
            {...props(styles.actionBtn)}
          >
            この曲の感想を書く
          </BottomSheetDrawer.Close>
          <BottomSheetDrawer.Close
            onClick={() => actions.removeLog(log.id)}
            {...props(styles.actionBtn, styles.actionBtnDanger)}
          >
            この日の記録から外す
          </BottomSheetDrawer.Close>
        </div>
      </BottomSheetDrawer.Content>
    </BottomSheetDrawer.Root>
  );
}

function DesktopLoggedSongPill({ log }: LoggedSongPillProps) {
  const { actions } = useListeningLog();
  const title = log.song.title ?? "（曲名なし）";
  const artist = log.song.artistName ?? "";
  const triggerLabel = `${title}${artist ? `（${artist}）` : ""}のメニューを開く`;

  return (
    <Popover>
      <PopoverTrigger aria-label={triggerLabel} {...props(styles.pill)}>
        <PillContent song={log.song} />
      </PopoverTrigger>

      <PopoverContent side="bottom" align="start" sideOffset={6} {...props(styles.actionPopover)}>
        <div {...props(styles.actionHead)}>
          <SongArt song={log.song} size={36} radius={7} />
          <div {...props(styles.metaCol)}>
            <PopoverTitle {...props(styles.metaTitle)}>{title}</PopoverTitle>
            {artist ? <span {...props(styles.metaArtist)}>{artist}</span> : null}
          </div>
        </div>
        <div {...props(styles.actionButtons)}>
          <PopoverClose onClick={() => actions.writeReviewForLog(log)} {...props(styles.actionBtn)}>
            この曲の感想を書く
          </PopoverClose>
          <PopoverClose
            onClick={() => actions.removeLog(log.id)}
            {...props(styles.actionBtn, styles.actionBtnDanger)}
          >
            この日の記録から外す
          </PopoverClose>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function ListeningDayShelf() {
  const { state, actions } = useListeningLog();
  const { selectedDate, selectedEntries, errorMessage, isLoading, shelfExpanded } = state;
  const compact = useMediaQuery("(max-width: 540px)");

  const visible = shelfExpanded ? selectedEntries : selectedEntries.slice(0, 2);
  const hidden = shelfExpanded ? [] : selectedEntries.slice(2);

  return (
    <section {...props(styles.shelf)} aria-label={`${formatJapaneseDate(selectedDate)} に聴いた曲`}>
      <div {...props(styles.header)}>
        <h2 {...props(styles.title)}>{formatJapaneseDate(selectedDate)} に聴いた曲</h2>
        <div {...props(styles.headerRight)}>
          <span {...props(styles.count)}>{selectedEntries.length} 曲</span>
          {selectedEntries.length > 2 ? (
            <button
              type="button"
              aria-expanded={shelfExpanded}
              onClick={actions.toggleShelfExpanded}
              {...props(styles.toggleLink)}
            >
              {shelfExpanded ? "折りたたむ" : "すべて見る"}
            </button>
          ) : null}
        </div>
      </div>

      {errorMessage ? (
        <p role="alert" {...props(styles.errorText)}>
          {errorMessage}
        </p>
      ) : null}

      {isLoading && selectedEntries.length === 0 ? (
        <p {...props(styles.emptyText)}>読み込み中…</p>
      ) : selectedEntries.length === 0 ? (
        <p {...props(styles.emptyText)}>
          まだ記録がありません。下のバーを「聴いた曲を記録」にして追加できます。
        </p>
      ) : (
        <div {...props(styles.stackWrap)}>
          {visible.map((log) =>
            compact ? (
              <MobileLoggedSongPill key={log.id} log={log} />
            ) : (
              <DesktopLoggedSongPill key={log.id} log={log} />
            ),
          )}

          {hidden.length > 0 ? (
            <button
              type="button"
              onClick={() => actions.setShelfExpanded(true)}
              aria-label={`ほか ${hidden.length} 曲を表示`}
              {...props(styles.clusterBtn)}
            >
              <span {...props(styles.clusterCovers)} aria-hidden="true">
                {hidden.slice(0, 3).map((item, idx) => (
                  <span key={item.id} {...props(idx > 0 && styles.clusterOverlap)}>
                    <SongArt song={item.song} size={22} radius={5} />
                  </span>
                ))}
              </span>
              <span {...props(styles.clusterLabel)}>+{hidden.length} 曲</span>
            </button>
          ) : null}
        </div>
      )}
    </section>
  );
}

const styles = create({
  shelf: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    padding: {
      default: 14,
      [chatMedia.compact]: 12,
    },
    marginBottom: {
      default: 18,
      [chatMedia.compact]: 12,
    },
    backgroundColor: t.surface,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 16,
  },
  header: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerRight: {
    display: "inline-flex",
    gap: 10,
    alignItems: "center",
  },
  title: {
    margin: 0,
    fontSize: 13.5,
    fontWeight: 600,
  },
  count: {
    fontSize: 12,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    color: t.muted,
  },
  toggleLink: {
    paddingBlock: 2,
    paddingInline: 6,
    fontFamily: "inherit",
    fontSize: 12,
    fontWeight: 500,
    color: t.text,
    cursor: "pointer",
    backgroundColor: {
      default: "transparent",
      ":hover": {
        default: "transparent",
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 6,
  },
  errorText: {
    margin: 0,
    fontSize: 12.5,
    color: t.danger,
  },
  emptyText: {
    margin: 0,
    fontSize: 12.5,
    color: t.muted,
  },
  stackWrap: {
    display: "flex",
    flexWrap: "wrap",
    gap: 7,
    alignItems: "center",
  },
  pill: {
    display: "inline-flex",
    gap: 8,
    alignItems: "center",
    maxWidth: "min(100%, 230px)",
    height: "auto",
    paddingBlock: 7,
    paddingRight: 12,
    paddingLeft: 7,
    fontFamily: "inherit",
    color: t.text,
    textAlign: "left",
    cursor: "pointer",
    backgroundColor: {
      default: t.surfaceElevated,
      ":hover": {
        default: t.surfaceElevated,
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 12,
    opacity: {
      "@starting-style": 0,
      default: 1,
    },
    transform: {
      "@starting-style": {
        default: "scale(0.96) translateY(4px)",
        [chatMedia.reducedMotion]: "none",
      },
      default: "scale(1) translateY(0)",
      ":active": {
        default: "scale(0.98)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `opacity 150ms ${motion.easeOut}, transform 160ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "opacity 120ms ease, background-color 120ms ease",
    },
  },
  clusterBtn: {
    display: "inline-flex",
    gap: 7,
    alignItems: "center",
    height: 46,
    paddingBlock: 0,
    paddingRight: 12,
    paddingLeft: 9,
    fontFamily: "inherit",
    color: t.text,
    cursor: "pointer",
    backgroundColor: {
      default: t.bg,
      ":hover": {
        default: t.bg,
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 12,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.97)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 150ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "background-color 120ms ease",
    },
  },
  clusterCovers: {
    display: "inline-flex",
    alignItems: "center",
  },
  clusterOverlap: {
    marginLeft: -6,
  },
  clusterLabel: {
    fontSize: 12,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
  },
  artImg: {
    display: "block",
    flexShrink: 0,
    objectFit: "cover",
  },
  artFallback: {
    display: "grid",
    flexShrink: 0,
    placeItems: "center",
    color: t.muted,
    backgroundColor: t.hover,
  },
  metaCol: {
    display: "flex",
    flex: "1",
    flexDirection: "column",
    gap: 2,
    minWidth: 0,
  },
  metaTitle: {
    margin: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    fontSize: 12.5,
    fontWeight: 600,
    lineHeight: 1.3,
    whiteSpace: "nowrap",
  },
  metaArtist: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    fontSize: 11,
    lineHeight: 1.25,
    color: t.muted,
    whiteSpace: "nowrap",
  },
  actionPopover: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    width: "min(260px, calc(100vw - 24px))",
    padding: 12,
    color: t.text,
    backgroundColor: t.surfaceElevated,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 14,
    boxShadow: t.shadow,
    opacity: {
      "@starting-style": 0,
      default: 1,
    },
    transform: {
      "@starting-style": {
        default: "scale(0.96)",
        [chatMedia.reducedMotion]: "none",
      },
      default: "scale(1)",
    },
    transformOrigin: "var(--transform-origin)",
    transition: {
      default: `opacity 150ms ${motion.easeOut}, transform 160ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "opacity 120ms ease",
    },
  },
  actionHead: {
    display: "flex",
    gap: 9,
    alignItems: "center",
  },
  actionButtons: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  actionBtn: {
    width: "100%",
    paddingBlock: 8,
    paddingInline: 10,
    fontFamily: "inherit",
    fontSize: 13,
    fontWeight: 500,
    color: t.text,
    textAlign: "left",
    cursor: "pointer",
    backgroundColor: {
      default: t.surface,
      ":hover": {
        default: t.surface,
        [chatMedia.hover]: t.hover,
      },
      ":active": t.hover,
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 9,
  },
  actionBtnDanger: {
    color: t.danger,
  },
});
