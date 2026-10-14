// The shape of src/data/replays.json, written by `python -m evals.export_replays`.

export type FileRef = { path: string; inPatch: boolean };

export type Attempt = {
  number: number;
  tests: { ran: boolean; passed: boolean; selected: string[] | null };
  accepted: boolean;
  judge: "tests" | "evaluator";
  problems: string[];
};

export type Replay = {
  id: string;
  instanceId: string;
  repo: string;
  version: string | null;
  tag: string;
  takeaway: string;
  config: { key: string; label: string; components: string[]; judge: "tests" | "evaluator"; note: string };
  issue: { title: string; body: string };
  retrieval: FileRef[];
  plan: { files: FileRef[] };
  attempts: Attempt[];
  maxAttempts: number;
  selfAccepted: boolean;
  patch: string;
  patchFiles: string[];
  grade: { resolved: boolean; failToPass: [number, number]; passToPass: [number, number] };
  metrics: {
    attempts: number;
    llmCalls: number;
    toolCalls: number;
    toolErrors: number;
    inputTokens: number;
    outputTokens: number;
    cacheReadTokens: number;
    costUsd: number;
    latencySeconds: number;
  };
  source: string;
};

export type ReplayData = { model: string; benchmark: string; replays: Replay[] };
