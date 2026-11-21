/** お気に入りの飾り（コメント、絵文字、色）。送られなかった項目は NULL として保存する。 */
export const decorationOf = <Color extends string>(item: {
  comment?: string | null;
  emoji?: string | null;
  color?: Color | null;
}) => ({ comment: item.comment ?? null, emoji: item.emoji ?? null, color: item.color ?? null });

/** 配列内で同じキーが重複している最初の要素を返す。重複がなければ undefined。 */
export function findDuplicateByKey<T>(items: ReadonlyArray<T>, keyOf: (item: T) => string) {
  const seen = new Set<string>();
  return items.find((item) => {
    const key = keyOf(item);
    return seen.size === seen.add(key).size;
  });
}
