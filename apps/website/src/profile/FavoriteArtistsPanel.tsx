import { props } from "@stylexjs/stylex";
import type { StyleXStyles } from "@stylexjs/stylex";
import { Button } from "@ui-sdk/react/button";
import { PlusIcon } from "#/chat/icons";
import { artworkUrl, searchArtists } from "#/lib/musickit";
import type { AppleArtist } from "#/lib/musickit";
import type { FavoriteArtistItem } from "#/lib/profile";
import { createFavoriteAddContext, EditFavoriteDialog, FavoriteDialog } from "./FavoriteDialog";
import { FavoritePanel } from "./FavoritePanel";
import { FavoriteRow } from "./FavoriteRow";
import { favoriteStyles as styles } from "./favorite-styles";
import { UserIcon } from "./icons";
import { useProfile } from "./ProfileContext";

const ArtistAddDialog = createFavoriteAddContext<AppleArtist>();

function ArtistArt({
  artist,
  size,
  style,
  roundStyle,
}: {
  artist: AppleArtist;
  size: number;
  style: StyleXStyles;
  roundStyle: StyleXStyles;
}) {
  const { artwork } = artist.attributes;
  return artwork ? (
    <img
      src={artworkUrl(artwork.url, size * 2)}
      alt=""
      width={size}
      height={size}
      {...props(style, roundStyle)}
    />
  ) : (
    <div {...props(style, styles.artFallback, roundStyle)} aria-hidden="true">
      <UserIcon />
    </div>
  );
}

function ArtistGenres({ artist }: { artist: AppleArtist }) {
  const genres = artist.attributes.genreNames;
  return genres?.length ? (
    <span {...props(styles.sub)}>{genres.slice(0, 2).join(" / ")}</span>
  ) : null;
}

function AddFavoriteArtistBody() {
  const { state } = useProfile();
  const { state: searchState, actions: searchActions } = ArtistAddDialog.useFavoriteAdd();
  const { selected } = searchState;
  const existingIds = new Set(state.favoriteArtists.map((item) => item.artist.id));

  return (
    <FavoriteDialog.Form>
      {selected ? (
        <FavoriteDialog.SelectedBox resetLabel="別のアーティストを選ぶ">
          <ArtistArt artist={selected} size={44} style={styles.art} roundStyle={styles.roundArt} />
          <div {...props(styles.meta)}>
            <span {...props(styles.title)}>{selected.attributes.name}</span>
            <ArtistGenres artist={selected} />
          </div>
        </FavoriteDialog.SelectedBox>
      ) : (
        <FavoriteDialog.SearchSection
          name="artist-search"
          label="アーティストを検索"
          placeholder="アーティスト名を入力…"
          idleMessage="キーワードを入力してアーティストを検索してください"
          notFoundMessage="一致するアーティストが見つかりませんでした"
        >
          <ul {...props(styles.resultList)} aria-label="アーティストの検索結果">
            {searchState.results.map((artist) => {
              const alreadyAdded = existingIds.has(artist.id);
              return (
                <li key={artist.id} {...props(styles.resultRow)}>
                  <Button
                    type="button"
                    disabled={alreadyAdded}
                    onClick={() => searchActions.select(artist)}
                    {...props(styles.resultSelectBtn)}
                  >
                    <ArtistArt
                      artist={artist}
                      size={38}
                      style={styles.resultArt}
                      roundStyle={styles.roundResultArt}
                    />
                    <span {...props(styles.meta)}>
                      <span {...props(styles.title)}>{artist.attributes.name}</span>
                      <ArtistGenres artist={artist} />
                    </span>
                    {alreadyAdded ? <span {...props(styles.badge)}>登録済み</span> : null}
                  </Button>
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

function AddFavoriteArtistDialog() {
  const { state, actions, meta } = useProfile();
  const atLimit = state.favoriteArtists.length >= meta.favoriteLimit;

  return (
    <ArtistAddDialog.AddProvider
      searchFn={searchArtists}
      saving={state.isSavingArtists}
      fieldPrefix="add-artist"
      onAdd={actions.addFavoriteArtist}
    >
      <FavoriteDialog.Trigger disabled={atLimit} {...props(styles.addBtn)}>
        <PlusIcon />
        <span>好きなアーティストを追加</span>
      </FavoriteDialog.Trigger>
      <FavoriteDialog.Content title="好きなアーティストを追加">
        <AddFavoriteArtistBody />
      </FavoriteDialog.Content>
    </ArtistAddDialog.AddProvider>
  );
}

function FavoriteArtistRow({
  item,
  index,
  total,
}: {
  item: FavoriteArtistItem;
  index: number;
  total: number;
}) {
  const { state, actions } = useProfile();
  const name = item.artist.name ?? "（アーティスト名なし）";

  return (
    <FavoriteRow.Root
      color={item.color}
      index={index}
      total={total}
      saving={state.isSavingArtists}
      label={name}
      noun="好きなアーティスト"
      onMove={(direction) => void actions.moveFavoriteArtist(index, direction)}
      onRemove={() => actions.removeFavoriteArtist(item.artist.id)}
    >
      <FavoriteRow.Main>
        <FavoriteRow.Rank />
        <FavoriteRow.RoundArt
          src={item.artist.artworkUrl}
          emoji={item.emoji}
          fallback={<UserIcon />}
        />

        <FavoriteRow.Meta>
          {item.artist.url ? (
            <a
              href={item.artist.url}
              target="_blank"
              rel="noreferrer"
              {...props(styles.title, styles.titleLink)}
            >
              {name}
            </a>
          ) : (
            <span {...props(styles.title)}>{name}</span>
          )}
        </FavoriteRow.Meta>

        <FavoriteRow.Actions>
          <FavoriteRow.MoveButtons />
          <EditFavoriteDialog
            title={name}
            initial={{ comment: item.comment, emoji: item.emoji, color: item.color }}
            fieldPrefix={`edit-artist-${item.artist.id}`}
            saving={state.isSavingArtists}
            onSave={(decoration) => actions.updateFavoriteArtist(item.artist.id, decoration)}
          />
          <FavoriteRow.RemoveDialog />
        </FavoriteRow.Actions>
      </FavoriteRow.Main>

      <FavoriteRow.Comment>{item.comment}</FavoriteRow.Comment>
    </FavoriteRow.Root>
  );
}

export function FavoriteArtistsPanel() {
  const { state, meta } = useProfile();

  return (
    <FavoritePanel.Root>
      <FavoritePanel.Bar
        title="好きなアーティスト"
        count={state.favoriteArtists.length}
        limit={meta.favoriteLimit}
      >
        <AddFavoriteArtistDialog />
      </FavoritePanel.Bar>

      {state.isProfilePending ? (
        <FavoritePanel.Loading>好きなアーティストを読み込んでいます…</FavoritePanel.Loading>
      ) : state.favoriteArtists.length === 0 ? (
        <FavoritePanel.Empty icon={<UserIcon />} title="まだ好きなアーティストが登録されていません">
          よく聴くアーティストや推しのバンドを最大 {meta.favoriteLimit} 組まで並べられます。
        </FavoritePanel.Empty>
      ) : (
        <FavoritePanel.List label="好きなアーティスト一覧">
          {state.favoriteArtists.map((item, index) => (
            <FavoriteArtistRow
              key={item.artist.id}
              item={item}
              index={index}
              total={state.favoriteArtists.length}
            />
          ))}
        </FavoritePanel.List>
      )}
    </FavoritePanel.Root>
  );
}
