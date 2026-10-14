import { Github } from "@/components/Icons";
import { Logo } from "@/components/Logo";
import { ReplayPlayer } from "@/components/ReplayPlayer";
import data from "@/data/replays.json";
import type { ReplayData } from "@/lib/types";

const REPO_URL = "https://github.com/AaravGarg16/PatchPilot";

const HOW = [
  ["Retrieve", "Code-aware search over the repo: semantic, keyword and symbol matches, fused."],
  ["Plan", "A planner turns the issue and retrieved code into the files to change."],
  ["Execute", "An executor edits files through tools it discovers over MCP."],
  ["Test", "Tests run in a disposable Docker sandbox with no network."],
  ["Evaluate", "An evaluator reviews the diff and results, and can send it back."],
  ["Retry", "Failures are fed into the next attempt, up to a fixed limit."],
];

const STACK = ["LangGraph", "MCP", "PostgreSQL + pgvector", "Docker", "Langfuse", "SWE-bench Lite", "Claude Sonnet 5"];

export default function Home() {
  const { replays } = data as ReplayData;
  return (
    <main>
      <header className="top">
        <a className="brand" href={REPO_URL}>
          <Logo />
          PatchPilot
        </a>
        <span className="top-right">
          <span className="live-note">Recorded runs · nothing here is live</span>
          <a className="ghost-link" href={REPO_URL}><Github size={15} />Source</a>
        </span>
      </header>

      <section className="hero">
        <p className="kicker">Autonomous software engineering agent</p>
        <h1>Watch an AI agent fix real GitHub issues, step by step.</h1>
        <p className="sub">
          PatchPilot reads an issue, finds the relevant code, plans a fix, edits files, runs the tests in a sandbox and judges
          its own patch. Below are replays of real runs on SWE-bench Lite, graded by the benchmark's official harness.
        </p>
      </section>

      <ReplayPlayer replays={replays} />

      <section className="how" aria-labelledby="how-title">
        <h2 id="how-title">How a run works</h2>
        <ol className="how-grid">
          {HOW.map(([title, text], i) => (
            <li key={title}>
              <span className="how-number mono">{String(i + 1).padStart(2, "0")}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </li>
          ))}
        </ol>
        <ul className="stack" aria-label="Built with">
          {STACK.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <footer className="foot">
        <p>
          These three runs were chosen because the official grader marked them resolved. They are examples, not the
          agent's average result; the full evaluation is in the repository.
        </p>
        <a className="ghost-link" href={REPO_URL}><Github size={15} />AaravGarg16/PatchPilot</a>
      </footer>
    </main>
  );
}
