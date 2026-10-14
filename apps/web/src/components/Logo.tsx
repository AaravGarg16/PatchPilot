// The PatchPilot mark: a P with a plus, the sign a diff uses for an added line. Colours come from the theme.
// The favicon in src/app/icon.svg is the same drawing with fixed colours; keep the two in step.
export function Logo({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className="logo">
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <g fill="none" stroke="var(--accent-ink)" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 24.5V8.5h6.25a4.75 4.75 0 0 1 0 9.5H11" strokeWidth="3.4" />
        <path d="M22.5 20.5v6M19.5 23.5h6" strokeWidth="2.4" />
      </g>
    </svg>
  );
}
