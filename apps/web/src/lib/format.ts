export function tokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}K`;
  return String(n);
}

export function usd(n: number): string {
  return `$${n.toFixed(2)}`;
}

export function duration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}

export function percent(part: number, whole: number): string {
  return whole ? `${Math.round((part / whole) * 100)}%` : "0%";
}

export function splitPath(path: string): { dir: string; name: string } {
  const i = path.lastIndexOf("/");
  return i < 0 ? { dir: "", name: path } : { dir: path.slice(0, i + 1), name: path.slice(i + 1) };
}

export function fileKind(path: string): "test" | "docs" | "source" {
  if (/(^|\/)tests?\//.test(path) || /(^|\/)test_[^/]*\.py$/.test(path)) return "test";
  if (path.startsWith("docs/") || /\.(txt|rst|md)$/.test(path)) return "docs";
  return "source";
}
