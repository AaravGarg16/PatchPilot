"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, Next, Pause, Play, Prev, Restart, Retry } from "@/components/Icons";
import { StepPanel } from "@/components/StepPanels";
import { duration, usd } from "@/lib/format";
import { buildSteps, type Step } from "@/lib/steps";
import type { Replay } from "@/lib/types";

function RunCard({ replay, active, onSelect }: { replay: Replay; active: boolean; onSelect: () => void }) {
  const m = replay.metrics;
  return (
    <button className={`run ${active ? "active" : ""}`} onClick={onSelect} aria-pressed={active}>
      <span className="run-top">
        <span className="run-repo mono">{replay.repo}</span>
        <span className="pill good"><Check size={12} />Resolved</span>
      </span>
      <span className="run-title">{replay.issue.title}</span>
      <span className="run-tag">{replay.tag} · {replay.config.label}</span>
      <span className="run-stats mono">
        <span>{m.attempts} attempt{m.attempts === 1 ? "" : "s"}</span>
        <span>{m.toolCalls} tool calls</span>
        <span>{usd(m.costUsd)}</span>
        <span>{duration(m.latencySeconds)}</span>
      </span>
    </button>
  );
}

function StepDot({ step, state }: { step: Step; state: "done" | "active" | "todo" }) {
  if (state === "done") {
    const sentBack = step.kind === "attempt" && !step.attempt.accepted;
    return <span className={`dot done ${sentBack ? "warn" : ""}`}>{sentBack ? <Retry size={11} /> : <Check size={11} />}</span>;
  }
  return <span className={`dot ${state}`} />;
}

export function ReplayPlayer({ replays }: { replays: Replay[] }) {
  const [runIndex, setRunIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  // Bumped whenever a step's clock restarts, so the progress bar's animation restarts with it.
  const [clock, setClock] = useState(0);
  const timelineRef = useRef<HTMLOListElement>(null);

  const replay = replays[runIndex];
  const steps = useMemo(() => buildSteps(replay), [replay]);
  const step = steps[stepIndex];
  const atEnd = stepIndex === steps.length - 1;

  const goTo = useCallback(
    (index: number) => {
      setStepIndex(Math.max(0, Math.min(index, steps.length - 1)));
      setClock((c) => c + 1);
    },
    [steps.length],
  );

  // A link can open a specific run and step (?run=<id>&step=<n>), paused there. Otherwise autoplay from the start,
  // unless the visitor asked their system for less motion.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const linkedRun = replays.findIndex((r) => r.id === params.get("run"));
    const linkedStep = Number(params.get("step"));
    if (linkedRun >= 0) setRunIndex(linkedRun);
    if (params.has("step") && Number.isInteger(linkedStep) && linkedStep >= 1) {
      const count = buildSteps(replays[Math.max(linkedRun, 0)]).length;
      setStepIndex(Math.min(linkedStep, count) - 1);
      return;
    }
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(true);
  }, [replays]);

  useEffect(() => {
    if (!playing) return;
    if (atEnd) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => goTo(stepIndex + 1), step.ms);
    return () => window.clearTimeout(timer);
  }, [playing, atEnd, stepIndex, step.ms, clock, goTo]);

  // On narrow screens the timeline is a horizontal strip; keep the active step in view without moving the page.
  useEffect(() => {
    const strip = timelineRef.current;
    const active = strip?.querySelector<HTMLElement>("li.active");
    if (strip && active && strip.scrollWidth > strip.clientWidth) {
      strip.scrollTo({ left: active.offsetLeft - (strip.clientWidth - active.offsetWidth) / 2, behavior: "smooth" });
    }
  }, [runIndex, stepIndex]);

  const selectRun = (index: number) => {
    setRunIndex(index);
    setStepIndex(0);
    setClock((c) => c + 1);
    setPlaying(true);
  };

  const togglePlay = useCallback(() => {
    if (atEnd) goTo(0);
    else setClock((c) => c + 1);
    setPlaying((p) => (atEnd ? true : !p));
  }, [atEnd, goTo]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, textarea, select")) return;
      if (event.key === "ArrowRight") { setPlaying(false); goTo(stepIndex + 1); }
      else if (event.key === "ArrowLeft") { setPlaying(false); goTo(stepIndex - 1); }
      else if (event.key === " " && !target.closest("button, a")) { event.preventDefault(); togglePlay(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goTo, stepIndex, togglePlay]);

  return (
    <section className="player" aria-label="Run replays">
      <div className="runs" role="group" aria-label="Choose a run">
        {replays.map((r, i) => (
          <RunCard key={r.id} replay={r} active={i === runIndex} onSelect={() => selectRun(i)} />
        ))}
      </div>

      <div className="stage">
        <div className="stage-bar">
          <div className="pipeline" aria-label="Pipeline for this run">
            {replay.config.components.map((component) => (
              <span className="component" key={component}>{component}</span>
            ))}
          </div>
          <div className="controls">
            <button className="icon-button" onClick={() => { setPlaying(false); goTo(stepIndex - 1); }} disabled={stepIndex === 0} aria-label="Previous step">
              <Prev />
            </button>
            <button className="play-button" onClick={togglePlay}>
              {atEnd ? <Restart size={15} /> : playing ? <Pause size={15} /> : <Play size={15} />}
              {atEnd ? "Replay" : playing ? "Pause" : "Play"}
            </button>
            <button className="icon-button" onClick={() => { setPlaying(false); goTo(stepIndex + 1); }} disabled={atEnd} aria-label="Next step">
              <Next />
            </button>
            <span className="counter mono">{stepIndex + 1} / {steps.length}</span>
          </div>
        </div>

        <div className="progress" aria-hidden>
          {steps.map((s, i) => (
            <span className="segment" key={s.id}>
              <span
                key={i === stepIndex ? clock : "idle"}
                className={`fill ${i < stepIndex ? "full" : i === stepIndex ? "running" : ""}`}
                style={i === stepIndex ? { animationDuration: `${s.ms}ms`, animationPlayState: playing ? "running" : "paused" } : undefined}
              />
            </span>
          ))}
        </div>

        <div className="stage-body">
          <ol className="timeline" ref={timelineRef}>
            {steps.map((s, i) => {
              const state = i < stepIndex ? "done" : i === stepIndex ? "active" : "todo";
              return (
                <li key={s.id} className={state}>
                  <button onClick={() => { setPlaying(false); goTo(i); }} aria-current={state === "active" ? "step" : undefined}>
                    <StepDot step={s} state={state} />
                    <span className="step-text">
                      <span className="step-label">{s.label}</span>
                      <span className="step-hint">{s.hint}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>

          <div className="panel" key={`${replay.id}-${step.id}`} aria-live="polite">
            <StepPanel replay={replay} step={step} />
          </div>
        </div>
      </div>
      <p className="hint-keys">Use <kbd>←</kbd> <kbd>→</kbd> to step and <kbd>space</kbd> to play or pause.</p>
    </section>
  );
}
