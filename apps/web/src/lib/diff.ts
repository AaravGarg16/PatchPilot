// A small unified-diff parser: enough to render a patch with line numbers.

export type DiffLine = { kind: "add" | "del" | "ctx"; text: string; oldNo: number | null; newNo: number | null };
export type DiffHunk = { header: string; context: string; lines: DiffLine[] };
export type DiffFile = { path: string; hunks: DiffHunk[]; additions: number; deletions: number };

export function parseDiff(patch: string): DiffFile[] {
  const files: DiffFile[] = [];
  let file: DiffFile | null = null;
  let hunk: DiffHunk | null = null;
  let oldNo = 0;
  let newNo = 0;

  for (const raw of patch.split("\n")) {
    if (raw.startsWith("diff --git ")) {
      const path = raw.split(" b/").pop() ?? raw;
      file = { path, hunks: [], additions: 0, deletions: 0 };
      files.push(file);
      hunk = null;
      continue;
    }
    if (!file) continue;
    const header = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$/.exec(raw);
    if (header) {
      oldNo = Number(header[1]);
      newNo = Number(header[2]);
      hunk = { header: raw, context: header[3], lines: [] };
      file.hunks.push(hunk);
      continue;
    }
    if (!hunk) continue; // index / --- / +++ lines before the first hunk
    if (raw.startsWith("+")) {
      hunk.lines.push({ kind: "add", text: raw.slice(1), oldNo: null, newNo: newNo++ });
      file.additions++;
    } else if (raw.startsWith("-")) {
      hunk.lines.push({ kind: "del", text: raw.slice(1), oldNo: oldNo++, newNo: null });
      file.deletions++;
    } else if (raw.startsWith(" ") || raw === "") {
      if (raw === "" && hunk.lines.length === 0) continue;
      hunk.lines.push({ kind: "ctx", text: raw.slice(1), oldNo: oldNo++, newNo: newNo++ });
    }
  }
  // A trailing newline in the patch leaves one phantom context line per file; drop it.
  for (const f of files) {
    const last = f.hunks.at(-1);
    if (last && last.lines.at(-1)?.kind === "ctx" && last.lines.at(-1)?.text === "") last.lines.pop();
  }
  return files;
}
