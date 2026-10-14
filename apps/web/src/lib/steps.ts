import type { Attempt, Replay } from "./types";

export type Step =
  | { id: string; kind: "issue" | "retrieval" | "plan" | "patch" | "result"; label: string; hint: string; ms: number }
  | { id: string; kind: "attempt"; label: string; hint: string; ms: number; attempt: Attempt };

// How long autoplay rests on each step.
const DWELL = { issue: 6500, retrieval: 5500, plan: 5000, attempt: 8000, patch: 9000, result: 12000 };

export function buildSteps(replay: Replay): Step[] {
  const attempts: Step[] = replay.attempts.map((attempt) => ({
    id: `attempt-${attempt.number}`,
    kind: "attempt",
    label: `Attempt ${attempt.number}`,
    hint: attempt.accepted
      ? "Tests pass, accepted"
      : `${attempt.tests.passed ? "Not accepted" : "Tests fail"}, ${attempt.number === replay.maxAttempts ? "limit reached" : "sent back"}`,
    ms: DWELL.attempt + Math.min(attempt.problems.join("").length * 12, 6000),
    attempt,
  }));
  return [
    { id: "issue", kind: "issue", label: "Issue", hint: "What needs fixing", ms: DWELL.issue },
    { id: "retrieval", kind: "retrieval", label: "Retrieve", hint: `${replay.retrieval.length} files found`, ms: DWELL.retrieval },
    { id: "plan", kind: "plan", label: "Plan", hint: `${replay.plan.files.length} files chosen`, ms: DWELL.plan },
    ...attempts,
    { id: "patch", kind: "patch", label: "Final patch", hint: `${replay.patchFiles.length} file${replay.patchFiles.length === 1 ? "" : "s"} changed`, ms: DWELL.patch },
    { id: "result", kind: "result", label: "Result", hint: "Official grade and cost", ms: DWELL.result },
  ];
}
