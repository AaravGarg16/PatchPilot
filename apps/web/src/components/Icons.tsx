type Props = { size?: number };

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

export const Check = ({ size = 16 }: Props) => (
  <svg {...base(size)}><path d="M20 6 9 17l-5-5" /></svg>
);
export const Cross = ({ size = 16 }: Props) => (
  <svg {...base(size)}><path d="M18 6 6 18M6 6l12 12" /></svg>
);
export const Play = ({ size = 16 }: Props) => (
  <svg {...base(size)} fill="currentColor" stroke="none"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5Z" /></svg>
);
export const Pause = ({ size = 16 }: Props) => (
  <svg {...base(size)} fill="currentColor" stroke="none"><rect x="5" y="4" width="5" height="16" rx="1.2" /><rect x="14" y="4" width="5" height="16" rx="1.2" /></svg>
);
export const Prev = ({ size = 16 }: Props) => (
  <svg {...base(size)}><path d="m15 18-6-6 6-6" /></svg>
);
export const Next = ({ size = 16 }: Props) => (
  <svg {...base(size)}><path d="m9 18 6-6-6-6" /></svg>
);
export const Restart = ({ size = 16 }: Props) => (
  <svg {...base(size)}><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /></svg>
);
export const Retry = ({ size = 16 }: Props) => (
  <svg {...base(size)}><path d="M21 12a9 9 0 1 1-3-6.7" /><path d="M21 4v5h-5" /></svg>
);
export const Github = ({ size = 16 }: Props) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.43-2.7 5.4-5.27 5.69.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
  </svg>
);
