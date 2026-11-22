import { props } from "@stylexjs/stylex";
import type { StyleXStyles } from "@stylexjs/stylex";
import { MusicNoteIcon, PlusIcon } from "#/chat/icons";
import { artworkUrl, searchSongs } from "#/lib/musickit";
import type { AppleSong } from "#/lib/musickit";
import type { FavoriteSongItem } from "#/lib/profile";
import { PreviewToggle } from "#/songs/PreviewToggle";
import { createFavoriteAddContext, EditFavoriteDialog, FavoriteDialog } from "./FavoriteDialog";
import { FavoritePanel } from "./FavoritePanel";
import { FavoriteRow } from "./FavoriteRow";
import { favoriteStyles as styles } from "./favorite-styles";
import { useProfile } from "./ProfileContext";

const SongAddDialog = createFavoriteAddContext<AppleSong>();

function SongArt({ song, size, style }: { song: AppleSong; size: number; style: StyleXStyles }) {
  const { artwork } = song.attributes;
  return artwork ? (
    <img
      src={artworkUrl(artwork.url, size * 2)}
      alt=""
      width={size}
      height={size}
      {...props(style)}
    />
  ) : (
    <div {...props(style, styles.artFallback)} aria-hidden="true">
      <MusicNoteIcon />
    </div>
  );
}

function AddFavoriteSongBody() {
  const { state } = useProfile();
  const { state: searchState, actions: searchActions } = SongAddDialog.useFavoriteAdd();
  const { selected } = searchState;
  const existingIsrcs = new Set(state.favoriteSongs.map((item) => item.song.isrc.toUpperCase()));

  return (
    <FavoriteDialog.Form>
      {selected ? (
        <FavoriteDialog.SelectedBox resetLabel="別の曲を選ぶ">
          <SongArt song={selected} size={44} style={styles.art} />
          <div {...props(styles.meta)}>
            <span {...props(styles.title)}>{selected.attributes.name}</span>
            <span {...props(styles.sub)}>{selected.attributes.artistName}</span>
          </div>
        </FavoriteDialog.SelectedBox>
      ) : (
        <FavoriteDialog.SearchSection
          name="song-search"
          label="曲を検索"
          placeholder="曲名やアーティスト名を入力…"
          idleMessage="キーワードを入力して曲を検索してください"
          notFoundMessage="一致する曲が見つかりませんでした"
        >
          <ul {...props(styles.resultList)} aria-label="曲の検索結果">
            {searchState.results.map((song) => {
              const isrc = song.attributes.isrc;
              const alreadyAdded = Boolean(isrc && existingIsrcs.has(isrc.toUpperCase()));
              const disabled = !isrc || alreadyAdded;
              return (
                <li key={song.id} {...props(styles.resultRow)}>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => searchActions.select(song)}
                    {...props(styles.resultSelectBtn)}
                  >
                    <SongArt song={song} size={38} style={styles.resultArt} />
                    <span {...props(styles.meta)}>
                      <span {...props(styles.title)}>{song.attributes.name}</span>
                      <span {...props(styles.sub)}>{song.attributes.artistName}</span>
                    </span>
                    {alreadyAdded ? (
                      <span {...props(styles.badge)}>登録済み</span>
                    ) : !isrc ? (
                      <span {...props(styles.badge)}>選択不可</span>
                    ) : null}
                  </button>
                  <PreviewToggle song={song} title={song.attributes.name} style={styles.iconBtn} />
                </li>
              );
            })}
          </ul>
        </FavoriteDialog.SearchSection>
      )}

      {selected ? <FavoriteDialog.DecorationFields /> : null}

      <FavoriteDialog.Footer idleLabel="リストに追加" savingLabel="追加中…" />
    </FavoriteDialog.Form>
  );
}

function AddFavoriteSongDialog() {
  const { state, actions, meta } = useProfile();
  const atLimit = state.favoriteSongs.length >= meta.favoriteLimit;

  return (
    <SongAddDialog.AddProvider
      searchFn={searchSongs}
      isSelectable={(song) => Boolean(song.attributes.isrc)}
      saving={state.isSavingSongs}
      fieldPrefix="add-song"
      onAdd={actions.addFavoriteSong}
    >
      <FavoriteDialog.Trigger disabled={atLimit} {...props(styles.addBtn)}>
        <PlusIcon />
        <span>好きな曲を追加</span>
      </FavoriteDialog.Trigger>
      <FavoriteDialog.Content
        title="好きな曲を追加"
        description={`Apple Music から曲を検索して、最大 ${meta.favoriteLimit} 曲まで登録できます。`}
      >
        <AddFavoriteSongBody />
      </FavoriteDialog.Content>
    </SongAddDialog.AddProvider>
  );
}

function FavoriteSongRow({
  item,
  index,
  total,
}: {
  item: FavoriteSongItem;
  index: number;
  total: number;
}) {
  const { state, actions } = useProfile();
  const title = item.song.title ?? "（曲名なし）";
  const catalogSong =
    state.catalogSongs.byId.get(item.song.appleSongId) ??
    state.catalogSongs.byIsrc.get(item.song.isrc.toUpperCase());

  return (
    <FavoriteRow.Root
      color={item.color}
      index={index}
      total={total}
      saving={state.isSavingSongs}
      label={title}
      noun="好きな曲"
      onMove={(direction) => void actions.moveFavoriteSong(index, direction)}
      onRemove={() => actions.removeFavoriteSong(item.song.isrc)}
    >
      <FavoriteRow.Main>
        <FavoriteRow.Rank />
        <FavoriteRow.SquareArt
          src={item.song.artworkUrl}
          emoji={item.emoji}
          fallback={<MusicNoteIcon />}
        />

        <FavoriteRow.Meta>
          {item.song.url ? (
            <a
              href={item.song.url}
              target="_blank"
              rel="noreferrer"
              {...props(styles.title, styles.titleLink)}
            >
              {title}
            </a>
          ) : (
            <span {...props(styles.title)}>{title}</span>
          )}
          {item.song.artistName ? <span {...props(styles.sub)}>{item.song.artistName}</span> : null}
        </FavoriteRow.Meta>

        <FavoriteRow.Actions>
          {catalogSong ? (
            <PreviewToggle song={catalogSong} title={title} style={styles.iconBtn} />
          ) : null}
          <FavoriteRow.MoveButtons />
          <EditFavoriteDialog
            title={title}
            initial={{ comment: item.comment, emoji: item.emoji, color: item.color }}
            fieldPrefix={`edit-song-${item.song.isrc}`}
            saving={state.isSavingSongs}
            onSave={(decoration) => actions.updateFavoriteSong(item.song.isrc, decoration)}
          />
          <FavoriteRow.RemoveDialog />
        </FavoriteRow.Actions>
      </FavoriteRow.Main>

      <FavoriteRow.Comment>{item.comment}</FavoriteRow.Comment>
    </FavoriteRow.Root>
  );
}

export function FavoriteSongsPanel() {
  const { state, meta } = useProfile();

  return (
    <FavoritePanel.Root>
      <FavoritePanel.Bar
        title="好きな曲"
        count={state.favoriteSongs.length}
        limit={meta.favoriteLimit}
      >
        <AddFavoriteSongDialog />
      </FavoritePanel.Bar>

      {state.isProfilePending ? (
        <FavoritePanel.Loading>好きな曲を読み込んでいます…</FavoritePanel.Loading>
      ) : state.favoriteSongs.length === 0 ? (
        <FavoritePanel.Empty icon={<MusicNoteIcon />} title="まだ好きな曲が登録されていません">
          自分を象徴する曲や、いま聴いている曲を最大 {meta.favoriteLimit} 曲まで並べられます。
        </FavoritePanel.Empty>
      ) : (
        <FavoritePanel.List label="好きな曲一覧">
          {state.favoriteSongs.map((item, index) => (
            <FavoriteSongRow
              key={item.song.isrc}
              item={item}
              index={index}
              total={state.favoriteSongs.length}
            />
          ))}
        </FavoritePanel.List>
      )}
    </FavoritePanel.Root>
  );
}
