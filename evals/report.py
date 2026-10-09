"""Turn graded results into the ablation table, the context-engineering comparison and the full metric list.

    python -m evals.report evals/results/smoke     # prints, and writes report.md in that folder
"""
import argparse
from pathlib import Path

from evals import results as store
from evals.configs import EXPERIMENTS, LABELS
from evals.grade import config_dirs
from evals.metrics import Summary, summarize
from evals.tasks import suite_instance_ids


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _tokens(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:,.1f}K"


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.3f}"


def ablation_table(summaries: dict[str, Summary]) -> str | None:
    rows = [summaries[c] for c in EXPERIMENTS["ablation"] if c in summaries]
    if len(rows) < 2:
        return None
    lines = ["| System | Success | Tokens / task | Avg retries | Cost / task | Graded |", "|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| {LABELS[s.config]} | {_pct(s.success_rate)} | {_tokens(s.tokens_per_task)} | "
                     f"{_num(s.avg_retries)} | {_money(s.cost_per_task)} | {s.graded}/{s.tasks} |")
    return "\n".join(lines)


def context_table(summaries: dict[str, Summary]) -> str | None:
    dynamic, history = summaries.get("full"), summaries.get("full-history")
    if not (dynamic and history):
        return None
    lines = [
        "| Context | Success | Input tokens / task | Tokens / successful task | Mean prompt | Max prompt |",
        "|---|---|---|---|---|---|",
    ]
    for s in (history, dynamic):
        lines.append(f"| {LABELS[s.config]} | {_pct(s.success_rate)} | {_tokens(s.input_tokens_per_task)} | "
                     f"{_tokens(s.tokens_per_successful_task)} | {_tokens(s.mean_prompt_tokens)} | "
                     f"{_tokens(s.max_prompt_tokens)} |")
    table = "\n".join(lines)
    if history.input_tokens_per_task:
        reduction = 1 - dynamic.input_tokens_per_task / history.input_tokens_per_task
        same = dynamic.success_rate == history.success_rate
        verdict = "at equal task success" if same else (
            f"but success differs ({_pct(dynamic.success_rate)} vs {_pct(history.success_rate)}), "
            "so this is not yet an equal-success comparison"
        )
        table += f"\n\nDynamic context uses {reduction:.0%} fewer input tokens per task than full history, {verdict}."
    return table


METRICS = [
    ("Tasks (graded / attempted)", lambda s: f"{s.graded}/{s.tasks}"),
    ("Task success rate", lambda s: _pct(s.success_rate)),
    ("Tests passed (target tests)", lambda s: _pct(s.tests_passed)),
    ("Patch acceptance (applies cleanly)", lambda s: _pct(s.patch_acceptance_rate)),
    ("Self-accepted by the agent", lambda s: _pct(s.self_acceptance_rate)),
    ("Self-acceptance precision", lambda s: _pct(s.self_acceptance_precision)),
    ("Invalid tool calls / task", lambda s: _num(s.invalid_tool_calls_per_task, 2)),
    ("Average retries", lambda s: _num(s.avg_retries, 2)),
    ("Input tokens / task", lambda s: _tokens(s.input_tokens_per_task)),
    ("Output tokens / task", lambda s: _tokens(s.output_tokens_per_task)),
    ("Tokens / task", lambda s: _tokens(s.tokens_per_task)),
    ("Tokens / successful task", lambda s: _tokens(s.tokens_per_successful_task)),
    ("Cost / task", lambda s: _money(s.cost_per_task)),
    ("Input served from prompt cache", lambda s: _pct(s.cached_input_share)),
    ("P50 latency", lambda s: f"{s.p50_latency:.0f}s"),
    ("P95 latency", lambda s: f"{s.p95_latency:.0f}s"),
    ("Retrieval precision (files)", lambda s: _pct(s.retrieval_precision)),
    ("Retrieval recall (files)", lambda s: _pct(s.retrieval_recall)),
    ("Mean prompt (context utilization)", lambda s: _tokens(s.mean_prompt_tokens)),
    ("Max prompt", lambda s: _tokens(s.max_prompt_tokens)),
    ("Agent errors / environment errors", lambda s: f"{s.agent_errors} / {s.env_errors}"),
]


def metrics_table(summaries: dict[str, Summary]) -> str:
    configs = list(summaries)
    lines = ["| Metric | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for name, fmt in METRICS:
        lines.append(f"| {name} | " + " | ".join(fmt(summaries[c]) for c in configs) + " |")
    return "\n".join(lines)


def build_report(run_dir: Path, suite: str | None = None) -> str:
    """Report on a results folder, optionally limited to one suite's tasks (runs for several suites share a folder)."""
    ids = set(suite_instance_ids(suite)) if suite else None
    summaries = {}
    for d in config_dirs(run_dir):
        results = [r for r in store.load_all(d) if ids is None or r.instance_id in ids]
        if results:
            summaries[d.name] = summarize(d.name, results)
    title = f"{Path(run_dir).name}" + (f", {suite} tasks" if suite else "")
    sections = [f"# PatchPilot eval: {title}"]
    ungraded = {c: s.tasks - s.graded for c, s in summaries.items() if s.tasks - s.graded}
    if ungraded:
        sections.append("Not graded yet (run `python -m evals.grade`): "
                        + ", ".join(f"{c}: {n}" for c, n in ungraded.items()))
    if table := ablation_table(summaries):
        sections.append("## Component ablation\n\n" + table)
    if table := context_table(summaries):
        sections.append("## Context engineering\n\n" + table)
    sections.append("## All metrics\n\n" + metrics_table(summaries))
    sections.append(
        "Success and tests passed come from the official SWE-bench harness (custom tasks: their own test suite). "
        "Tokens per successful task is all tokens spent divided by tasks resolved. Retrieval precision compares the "
        "files retrieved for the prompt with the files the reference patch changes. Latency percentiles are "
        "nearest-rank, so they are observed values."
    )
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--suite", help="only this suite's tasks, e.g. ablation8 for the ablation, lite20 for the headline")
    args = parser.parse_args()
    report = build_report(args.run_dir, args.suite)
    (args.run_dir / (f"report-{args.suite}.md" if args.suite else "report.md")).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
