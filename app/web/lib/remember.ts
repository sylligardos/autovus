// Page settings that survive moving between pages: a useState whose value
// is kept in memory for the browser tab. A value from the URL wins over the
// remembered one, which wins over the default. A reload starts fresh.

import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

const store = new Map<string, unknown>();

export function useRemembered<T>(key: string, initial: T, fromUrl?: T | null): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => (fromUrl !== undefined && fromUrl !== null ? fromUrl : store.has(key) ? (store.get(key) as T) : initial));
  useEffect(() => { store.set(key, value); }, [key, value]);
  return [value, setValue];
}
