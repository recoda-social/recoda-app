import { create, props } from "@stylexjs/stylex";
import { Link } from "@tanstack/react-router";
import { Avatar, AvatarFallback, AvatarImage } from "@ui-sdk/react/avatar";
import { Button } from "@ui-sdk/react/button";
import { Toggle } from "@ui-sdk/react/toggle";
import { MusicNoteIcon } from "#/chat/icons";
import { chatMedia, chatTokens as t } from "#/chat/tokens.stylex";
import { artworkUrl } from "#/lib/musickit";
import type { AppleSong } from "#/lib/musickit";
import type { PostView } from "#/lib/posts";
import { motion } from "#/motion.stylex";
import { ArrowLeftIcon } from "#/profile/icons";
import { AppShell } from "#/shell/AppShell";
import { MiniPlayerBar } from "#/songs/MiniPlayerBar";
import { PreviewToggle } from "#/songs/PreviewToggle";
import { usePostDetail } from "./PostDetailContext";
import { PostDetailProvider } from "./PostDetailProvider";

const dateTimeFormatter = new Intl.DateTimeFormat("ja-JP", {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

const expiresFormatter = new Intl.DateTimeFormat("ja-JP", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatDateTime(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return dateTimeFormatter.format(date);
}

function formatExpiresAt(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return expiresFormatter.format(date);
}

function initials(name: string) {
  return Array.from(name.trim()).slice(0, 2).join("") || "U";
}

function SongRow({ post, catalogSong }: { post: PostView; catalogSong?: AppleSong }) {
  if (!post.song) {
    return null;
  }
  const title = post.song.title ?? "（曲名なし）";

  return (
    <div {...props(styles.songRow)}>
      {post.song.artworkUrl ? (
        <img
          src={artworkUrl(post.song.artworkUrl, 96)}
          alt=""
          width={44}
          height={44}
          {...props(styles.art)}
        />
      ) : (
        <div {...props(styles.art, styles.artFallback)} aria-hidden="true">
          <MusicNoteIcon />
        </div>
      )}
      <div {...props(styles.songMeta)}>
        {post.song.url ? (
          <a
            href={post.song.url}
            target="_blank"
            rel="noreferrer"
            {...props(styles.songTitle, styles.songTitleLink)}
          >
            {title}
          </a>
        ) : (
          <span {...props(styles.songTitle)}>{title}</span>
        )}
        {post.song.artistName ? (
          <span {...props(styles.songArtist)}>{post.song.artistName}</span>
        ) : null}
      </div>
      {catalogSong ? (
        <PreviewToggle song={catalogSong} title={title} style={styles.iconBtn} />
      ) : null}
    </div>
  );
}

function SimilarPostCard({ post, catalogSong }: { post: PostView; catalogSong?: AppleSong }) {
  return (
    <article {...props(styles.similarCard)}>
      <SongRow post={post} catalogSong={catalogSong} />
      <p {...props(styles.similarQuote)}>{post.content}</p>
      <div {...props(styles.similarAuthor)}>
        <Avatar {...props(styles.avatarSm)}>
          {post.author.image ? (
            <AvatarImage
              src={post.author.image}
              alt={post.author.name}
              referrerPolicy="no-referrer"
              {...props(styles.avatarImg)}
            />
          ) : null}
          <AvatarFallback>
            <span aria-hidden="true">{initials(post.author.name)}</span>
            <span data-slot="visually-hidden">{post.author.name}</span>
          </AvatarFallback>
        </Avatar>
        <span>{post.author.name}</span>
      </div>
    </article>
  );
}

function PostDetailHeader() {
  return (
    <div {...props(styles.stickyHeader)}>
      <AppShell.Topbar>
        <div {...props(styles.leftNav)}>
          <Button render={<Link to="/" />} {...props(styles.backBtn)}>
            <ArrowLeftIcon />
            <span>ホーム</span>
          </Button>
          <Link to="/" {...props(styles.brand)}>
            <AppShell.BrandIcon />
            <AppShell.BrandName style={styles.brandName} />
          </Link>
        </div>

        <AppShell.SignOutButton />
      </AppShell.Topbar>

      <MiniPlayerBar />
    </div>
  );
}

function PostArticleCard() {
  const { state, actions } = usePostDetail();
  if (!state.post) {
    return null;
  }

  return (
    <section aria-label="あなたの投稿" {...props(styles.postCard)}>
      <SongRow post={state.post} catalogSong={actions.lookupCatalogSong(state.post)} />
      <p {...props(styles.postContent)}>{state.post.content}</p>
      <footer {...props(styles.postFooter)}>
        <time dateTime={state.post.createdAt} {...props(styles.timestamp)}>
          {formatDateTime(state.post.createdAt)}
        </time>
      </footer>
    </section>
  );
}

function SimilarPostsSection() {
  const { state, actions } = usePostDetail();
  const {
    similarPosts,
    isSimilarPending,
    similarError,
    similarExpired,
    similarExpiresAt,
    includeSameSong,
  } = state;

  return (
    <section aria-labelledby="similar-section-heading" {...props(styles.section)}>
      <div {...props(styles.sectionHeader)}>
        <div {...props(styles.sectionTitleGroup)}>
          <h2 id="similar-section-heading" {...props(styles.sectionTitle)}>
            近い感覚の曲
          </h2>
          {similarExpiresAt ? (
            <span {...props(styles.expiryHint)}>
              {formatExpiresAt(similarExpiresAt)} まで閲覧可能
            </span>
          ) : null}
        </div>
        {similarExpiresAt ? (
          <Toggle
            {...props(styles.filterToggle, includeSameSong && styles.filterTogglePressed)}
            pressed={includeSameSong}
            onPressedChange={actions.setIncludeSameSong}
          >
            同じ曲も含める
          </Toggle>
        ) : null}
      </div>

      {isSimilarPending ? (
        <div {...props(styles.stateBox)} aria-busy="true">
          検索結果を読み込んでいます…
        </div>
      ) : similarError ? (
        <p role="alert" {...props(styles.inlineError)}>
          {similarError}
        </p>
      ) : similarExpired ? (
        <div {...props(styles.expiredCard)}>
          <p {...props(styles.expiredTitle)}>検索結果の閲覧期限（24時間）を過ぎました</p>
          <p {...props(styles.expiredDesc)}>
            投稿した感想は引き続きいつでもこのページから見返せます。
          </p>
        </div>
      ) : similarPosts.length > 0 ? (
        <div {...props(styles.similarList)}>
          {similarPosts.map((post) => (
            <SimilarPostCard
              key={post.id}
              post={post}
              catalogSong={actions.lookupCatalogSong(post)}
            />
          ))}
        </div>
      ) : (
        <div {...props(styles.expiredCard)}>
          <p {...props(styles.expiredDesc)}>まだ近い感覚の感想は見つかりませんでした。</p>
        </div>
      )}
    </section>
  );
}

function PostDetailBody() {
  const { state } = usePostDetail();

  return (
    <main {...props(styles.main)}>
      <div {...props(styles.inner)}>
        {state.isPostPending ? (
          <div {...props(styles.stateBox)} aria-busy="true">
            投稿を読み込んでいます…
          </div>
        ) : state.postError ? (
          <div role="alert" {...props(styles.errorBox)}>
            <p {...props(styles.errorMessage)}>{state.postError}</p>
            <Button render={<Link to="/" />} {...props(styles.errorAction)}>
              ホームに戻る
            </Button>
          </div>
        ) : (
          <>
            <PostArticleCard />
            <SimilarPostsSection />
          </>
        )}
      </div>
    </main>
  );
}

export function PostDetailScreen({ postId }: { postId: string }) {
  return (
    <PostDetailProvider postId={postId}>
      <div {...props(styles.shell)}>
        <PostDetailHeader />
        <PostDetailBody />
      </div>
    </PostDetailProvider>
  );
}

const styles = create({
  shell: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100dvh",
    color: t.text,
    colorScheme: "light dark",
    backgroundColor: t.bg,
  },
  stickyHeader: {
    position: "sticky",
    top: 0,
    zIndex: 10,
  },
  leftNav: {
    display: "flex",
    gap: 12,
    alignItems: "center",
    minWidth: 0,
  },
  backBtn: {
    display: "inline-flex",
    gap: 6,
    alignItems: "center",
    height: 30,
    paddingBlock: 0,
    paddingInline: 10,
    fontSize: 13,
    fontWeight: 500,
    color: t.text,
    textDecoration: "none",
    cursor: "pointer",
    backgroundColor: {
      default: "transparent",
      ":hover": {
        default: "transparent",
        [chatMedia.hover]: t.hover,
      },
    },
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9999,
    transition: "background-color 150ms ease",
  },
  brand: {
    display: "inline-flex",
    gap: 8,
    alignItems: "center",
    color: "inherit",
    textDecoration: "none",
  },
  brandName: {
    display: {
      default: "inline",
      [chatMedia.compact]: "none",
    },
  },
  main: {
    flexGrow: 1,
    width: "100%",
  },
  inner: {
    display: "flex",
    flexDirection: "column",
    gap: 24,
    width: "min(100%, 720px)",
    paddingTop: {
      default: 28,
      [chatMedia.compact]: 16,
    },
    paddingRight: {
      default: 20,
      [chatMedia.compact]: 14,
    },
    paddingBottom: "max(48px, env(safe-area-inset-bottom, 0px))",
    paddingLeft: {
      default: 20,
      [chatMedia.compact]: 14,
    },
    marginBlock: 0,
    marginInline: "auto",
  },
  postCard: {
    display: "flex",
    flexDirection: "column",
    gap: 14,
    padding: 18,
    backgroundColor: t.surface,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 18,
  },
  postContent: {
    margin: 0,
    fontSize: 16,
    lineHeight: 1.7,
    color: t.text,
    wordBreak: "break-word",
    whiteSpace: "pre-wrap",
  },
  postFooter: {
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-end",
    paddingTop: 10,
    borderTopColor: t.border,
    borderTopStyle: "solid",
    borderTopWidth: 1,
  },
  timestamp: {
    fontSize: 12,
    fontVariantNumeric: "tabular-nums",
    color: t.muted,
  },
  songRow: {
    display: "flex",
    gap: 10,
    alignItems: "center",
    paddingBlock: 8,
    paddingInline: 10,
    backgroundColor: t.surfaceElevated,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 12,
  },
  art: {
    flexShrink: 0,
    width: 44,
    height: 44,
    objectFit: "cover",
    borderRadius: 8,
  },
  artFallback: {
    display: "grid",
    placeItems: "center",
    color: t.muted,
    backgroundColor: t.hover,
  },
  songMeta: {
    display: "flex",
    flex: "1",
    flexDirection: "column",
    gap: 2,
    minWidth: 0,
  },
  songTitle: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    fontSize: 14,
    fontWeight: 600,
    color: t.text,
    whiteSpace: "nowrap",
  },
  songTitleLink: {
    textDecoration: {
      default: "none",
      ":hover": {
        default: "none",
        [chatMedia.hover]: "underline",
      },
    },
  },
  songArtist: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    fontSize: 12,
    color: t.muted,
    whiteSpace: "nowrap",
  },
  iconBtn: {
    display: "grid",
    flexShrink: 0,
    placeItems: "center",
    width: 30,
    height: 30,
    padding: 0,
    color: {
      default: t.muted,
      ":hover": {
        default: t.muted,
        [chatMedia.hover]: t.text,
      },
    },
    cursor: "pointer",
    backgroundColor: {
      default: "transparent",
      ":hover": {
        default: "transparent",
        [chatMedia.hover]: t.hover,
      },
    },
    borderStyle: "none",
    borderWidth: 0,
    borderRadius: 9999,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.97)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 160ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}, color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]: "background-color 120ms ease, color 120ms ease",
    },
  },
  section: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  sectionHeader: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    alignItems: "center",
    justifyContent: "space-between",
  },
  sectionTitleGroup: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    alignItems: "baseline",
  },
  sectionTitle: {
    margin: 0,
    fontSize: 15,
    fontWeight: 600,
    letterSpacing: "-0.01em",
  },
  expiryHint: {
    fontSize: 12,
    fontVariantNumeric: "tabular-nums",
    color: t.muted,
  },
  filterToggle: {
    height: 28,
    paddingBlock: 0,
    paddingInline: 10,
    fontSize: 12,
    fontWeight: 500,
    color: t.muted,
    cursor: "pointer",
    backgroundColor: "transparent",
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9999,
    transform: {
      default: null,
      ":active": {
        default: "scale(0.97)",
        [chatMedia.reducedMotion]: "none",
      },
    },
    transition: {
      default: `transform 160ms ${motion.easeOut}, background-color 150ms ${motion.easeOut}, color 150ms ${motion.easeOut}, border-color 150ms ${motion.easeOut}`,
      [chatMedia.reducedMotion]:
        "background-color 120ms ease, color 120ms ease, border-color 120ms ease",
    },
  },
  filterTogglePressed: {
    color: t.primaryFg,
    backgroundColor: t.primary,
    borderColor: t.primary,
  },
  similarList: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  similarCard: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    padding: 14,
    backgroundColor: t.surface,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 14,
  },
  similarQuote: {
    margin: 0,
    fontSize: 14,
    lineHeight: 1.65,
    wordBreak: "break-word",
    whiteSpace: "pre-wrap",
  },
  similarAuthor: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    fontSize: 12,
    color: t.muted,
  },
  avatarSm: {
    display: "inline-flex",
    flexShrink: 0,
    alignItems: "center",
    justifyContent: "center",
    width: 24,
    height: 24,
    overflow: "hidden",
    fontSize: 11,
    borderRadius: 9999,
  },
  avatarImg: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  expiredCard: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    paddingBlock: 24,
    paddingInline: 18,
    textAlign: "center",
    backgroundColor: t.surface,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 14,
  },
  expiredTitle: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
    color: t.text,
  },
  expiredDesc: {
    margin: 0,
    fontSize: 13,
    lineHeight: 1.6,
    color: t.muted,
  },
  stateBox: {
    paddingBlock: 36,
    paddingInline: 16,
    fontSize: 14,
    color: t.muted,
    textAlign: "center",
  },
  errorBox: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
    alignItems: "center",
    padding: 24,
    textAlign: "center",
    backgroundColor: t.surface,
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 16,
  },
  errorMessage: {
    margin: 0,
    fontSize: 14,
    color: t.danger,
  },
  errorAction: {
    display: "inline-flex",
    alignItems: "center",
    height: 34,
    paddingBlock: 0,
    paddingInline: 14,
    fontSize: 13,
    fontWeight: 500,
    color: t.text,
    textDecoration: "none",
    borderColor: t.border,
    borderStyle: "solid",
    borderWidth: 1,
    borderRadius: 9999,
  },
  inlineError: {
    margin: 0,
    fontSize: 14,
    color: t.danger,
  },
});
