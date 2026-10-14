import { DiffView } from "@/components/DiffView";
import { Check, Cross, Retry } from "@/components/Icons";
import { duration, fileKind, percent, splitPath, tokens, usd } from "@/lib/format";
import type { Step } from "@/lib/steps";
import type { Attempt, FileRef, Replay } from "@/lib/types";

const stagger = (i: number) => ({ "--i": i }) as React.CSSProperties;

function Path({ path }: { path: string }) {
  const { dir, name } = splitPath(path);
  return (
    <span className="path">
      <span className="path-dir">{dir}</span>
      {name}
    </span>
  );
}

function FileList({ files, ranked }: { files: FileRef[]; ranked?: boolean }) {
  return (
    <ol className="files">
      {files.map((file, i) => (
        <li className={`file reveal ${file.inPatch ? "hit" : ""}`} key={file.path} style={stagger(i)}>
          {ranked && <span className="rank">{i + 1}</span>}
          <Path path={file.path} />
          <span className={`kind ${fileKind(file.path)}`}>{fileKind(file.path)}</span>
          {file.inPatch && <span className="badge good">Changed in the final patch</span>}
        </li>
      ))}
    </ol>
  );
}

// Issue text is plain text with occasional ``` fences; keep it as written and only set code apart.
function IssueBody({ body }: { body: string }) {
  const parts = body.split(/```(?:\w+)?\n?/);
  return (
    <div className="issue-body">
      {parts.map((part, i) =>
        i % 2 === 1 ? <pre key={i}><code>{part.trimEnd()}</code></pre> : part.trim() && <p key={i}>{part.trim()}</p>,
      )}
    </div>
  );
}

function IssuePanel({ replay }: { replay: Replay }) {
  return (
    <>
      <p className="eyebrow">
        {replay.repo}
        {replay.version ? ` · v${replay.version}` : ""} · <span className="mono">{replay.instanceId}</span>
      </p>
      <h2>{replay.issue.title}</h2>
      <p className="lede">The agent starts with only this: the issue as it was filed, and the repository at the commit before the fix.</p>
      <IssueBody body={replay.issue.body} />
    </>
  );
}

function RetrievalPanel({ replay }: { replay: Replay }) {
  const hit = replay.retrieval.findIndex((f) => f.inPatch);
  return (
    <>
      <p className="eyebrow">Step 2 · Retrieval</p>
      <h2>Find the code the issue is about</h2>
      <p className="lede">
        The repository is split into functions and classes, embedded, and searched three ways at once: by meaning, by keyword
        and by symbol name. These are the top files.
      </p>
      <FileList files={replay.retrieval} ranked />
      <p className="note">
        {hit >= 0
          ? `The file that ends up being fixed is ranked ${hit + 1} of ${replay.retrieval.length}.`
          : "The file that ends up being fixed is not in this list; the agent found it through its tools."}
      </p>
    </>
  );
}

function PlanPanel({ replay }: { replay: Replay }) {
  return (
    <>
      <p className="eyebrow">Step 3 · Plan</p>
      <h2>Decide what to change</h2>
      <p className="lede">
        The planner reads the issue and the retrieved code, then narrows the work to the files the executor should focus on.
      </p>
      <FileList files={replay.plan.files} />
      <p className="note">Only the planner's file selection was stored for this run, so its written steps are not shown.</p>
    </>
  );
}

function TestsCard({ attempt }: { attempt: Attempt }) {
  const { tests } = attempt;
  return (
    <div className="card reveal" style={stagger(1)}>
      <div className="card-head">
        <h3>Tests in the sandbox</h3>
        <span className={`pill ${tests.passed ? "good" : "bad"}`}>
          {tests.passed ? <Check size={13} /> : <Cross size={13} />}
          {tests.passed ? "Passed" : "Failed"}
        </span>
      </div>
      <p className="card-text">Related tests run inside a locked-down Docker container: no network, read-only code, capped CPU and memory.</p>
      {tests.selected ? (
        <ul className="chips">
          {tests.selected.map((path) => (
            <li className="chip mono" key={path}>{path}</li>
          ))}
        </ul>
      ) : (
        <p className="note">The list of test files was only stored for attempts that were sent back.</p>
      )}
    </div>
  );
}

function VerdictCard({ attempt, replay }: { attempt: Attempt; replay: Replay }) {
  const byEvaluator = attempt.judge === "evaluator";
  const last = attempt.number === replay.maxAttempts;
  return (
    <div className="card reveal" style={stagger(2)}>
      <div className="card-head">
        <h3>{byEvaluator ? "Evaluator verdict" : "Verdict"}</h3>
        <span className={`pill ${attempt.accepted ? "good" : "warn"}`}>
          {attempt.accepted ? <Check size={13} /> : <Retry size={13} />}
          {attempt.accepted ? "Accepted" : last ? "Not accepted" : "Sent back"}
        </span>
      </div>
      {attempt.accepted ? (
        <p className="card-text">
          {byEvaluator
            ? "The evaluator reviewed the diff against the issue and found nothing to send back."
            : "This configuration accepts a patch when its related tests pass."}
        </p>
      ) : (
        <>
          <p className="card-text">
            {byEvaluator
              ? "A second model reviews the diff and the test results, and writes down what still needs to change:"
              : "The related tests failed, so the test output goes back to the executor."}
          </p>
          <ul className="problems">
            {attempt.problems.map((problem, i) => (
              <li className="reveal" key={i} style={stagger(3 + i)}>{problem}</li>
            ))}
          </ul>
          <p className="note">
            {last
              ? `That was attempt ${attempt.number} of ${replay.maxAttempts}, the limit. The patch goes to the official grader as it stands.`
              : "This feedback and the failing test output are fed into the next attempt."}
          </p>
        </>
      )}
    </div>
  );
}

function AttemptPanel({ replay, attempt }: { replay: Replay; attempt: Attempt }) {
  return (
    <>
      <p className="eyebrow">
        Attempt {attempt.number} of {replay.maxAttempts}
      </p>
      <h2>{attempt.number === 1 ? "Edit the code, then check the work" : "Try again with the feedback"}</h2>
      <div className="card reveal" style={stagger(0)}>
        <div className="card-head">
          <h3>Executor</h3>
        </div>
        <p className="card-text">
          Reads, searches and edits files through tools it discovers over MCP. It cannot touch existing tests, so it has to fix
          the code they test.
        </p>
      </div>
      <TestsCard attempt={attempt} />
      <VerdictCard attempt={attempt} replay={replay} />
    </>
  );
}

function PatchPanel({ replay }: { replay: Replay }) {
  return (
    <>
      <p className="eyebrow">Final patch</p>
      <h2>The diff the agent produced</h2>
      <p className="lede">
        Exactly as submitted to the grader{replay.attempts.length > 1 ? `, after ${replay.attempts.length} attempts` : ""}.
      </p>
      <DiffView patch={replay.patch} />
    </>
  );
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>{value}</dd>
      {detail && <span className="stat-detail">{detail}</span>}
    </div>
  );
}

function ResultPanel({ replay }: { replay: Replay }) {
  const { grade, metrics: m } = replay;
  return (
    <>
      <p className="eyebrow">Result</p>
      <h2>Graded by the official SWE-bench harness</h2>
      <div className="verdicts">
        <div className="verdict good reveal" style={stagger(0)}>
          <span className="verdict-icon"><Check size={20} /></span>
          <div>
            <p className="verdict-title">Resolved</p>
            <p className="verdict-text">
              The benchmark's hidden test{grade.failToPass[1] === 1 ? "" : "s"} for this issue: {grade.failToPass[0]} of{" "}
              {grade.failToPass[1]} now pass. Existing tests: {grade.passToPass[0]} of {grade.passToPass[1]} still pass.
            </p>
          </div>
        </div>
        <div className={`verdict reveal ${replay.selfAccepted ? "good" : "warn"}`} style={stagger(1)}>
          <span className="verdict-icon">{replay.selfAccepted ? <Check size={20} /> : <Retry size={20} />}</span>
          <div>
            <p className="verdict-title">{replay.selfAccepted ? "Agent accepted its own patch" : "Agent did not accept its own patch"}</p>
            <p className="verdict-text">
              {replay.selfAccepted
                ? "Its own check agreed with the grader, which it never sees."
                : "Its evaluator was stricter than the grader here. Both outcomes are shown as recorded."}
            </p>
          </div>
        </div>
      </div>
      <p className="takeaway reveal" style={stagger(2)}>{replay.takeaway}</p>
      <dl className="stats reveal" style={stagger(3)}>
        <Stat label="Attempts" value={String(m.attempts)} detail={`limit ${replay.maxAttempts}`} />
        <Stat label="Tool calls" value={String(m.toolCalls)} detail={m.toolErrors ? `${m.toolErrors} returned an error` : "none failed"} />
        <Stat label="Model calls" value={String(m.llmCalls)} />
        <Stat label="Input tokens" value={tokens(m.inputTokens)} detail={`${percent(m.cacheReadTokens, m.inputTokens)} served from cache`} />
        <Stat label="Output tokens" value={tokens(m.outputTokens)} />
        <Stat label="Cost" value={usd(m.costUsd)} />
        <Stat label="Wall time" value={duration(m.latencySeconds)} />
      </dl>
      <p className="note mono">{replay.source}</p>
    </>
  );
}

export function StepPanel({ replay, step }: { replay: Replay; step: Step }) {
  switch (step.kind) {
    case "issue":
      return <IssuePanel replay={replay} />;
    case "retrieval":
      return <RetrievalPanel replay={replay} />;
    case "plan":
      return <PlanPanel replay={replay} />;
    case "attempt":
      return <AttemptPanel replay={replay} attempt={step.attempt} />;
    case "patch":
      return <PatchPanel replay={replay} />;
    case "result":
      return <ResultPanel replay={replay} />;
  }
}
