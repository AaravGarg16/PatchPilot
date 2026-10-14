import { parseDiff } from "@/lib/diff";
import { splitPath } from "@/lib/format";

export function DiffView({ patch }: { patch: string }) {
  const files = parseDiff(patch);
  let row = 0;
  return (
    <div className="diff">
      {files.map((file) => {
        const { dir, name } = splitPath(file.path);
        return (
          <section className="diff-file" key={file.path}>
            <header className="diff-head">
              <span className="path">
                <span className="path-dir">{dir}</span>
                {name}
              </span>
              <span className="diff-stat">
                <span className="add">+{file.additions}</span>
                <span className="del">−{file.deletions}</span>
              </span>
            </header>
            <div className="diff-body" role="table" aria-label={`Changes to ${file.path}`}>
              {file.hunks.map((hunk) => (
                <div key={hunk.header} role="rowgroup">
                  <div className="diff-hunk" role="row">
                    <span role="cell">{hunk.context || "…"}</span>
                  </div>
                  {hunk.lines.map((line, i) => (
                    <div
                      className={`diff-line ${line.kind}`}
                      role="row"
                      key={`${hunk.header}-${i}`}
                      style={{ "--i": Math.min(row++, 60) } as React.CSSProperties}
                    >
                      <span className="ln" role="cell">{line.oldNo ?? ""}</span>
                      <span className="ln" role="cell">{line.newNo ?? ""}</span>
                      <span className="sign" role="cell">{line.kind === "add" ? "+" : line.kind === "del" ? "−" : ""}</span>
                      <code role="cell">{line.text || " "}</code>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
