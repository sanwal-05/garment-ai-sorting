import type { Thresholds } from "../types";

interface Props {
  label: string;
  value: number;
  thresholds?: Thresholds;
}

export default function ScoreBar({ label, value, thresholds }: Props) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const lo = (thresholds?.lo ?? 0.25) * 100;
  const hi = (thresholds?.hi ?? 0.7) * 100;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="font-medium">{label.replaceAll("_", " ")}</span>
        <span className="tabular-nums text-stone-600">{pct.toFixed(0)}%</span>
      </div>
      <div
        className="relative h-3 overflow-hidden rounded-full bg-sand"
        role="img"
        aria-label={`${label} ${pct.toFixed(0)} percent, low threshold ${lo.toFixed(0)}, high threshold ${hi.toFixed(0)}`}
      >
        <div className="absolute inset-y-0 left-0 rounded-full bg-clay" style={{ width: `${pct}%` }} />
        <span
          className="absolute top-0 h-full w-0.5 bg-ink/70"
          style={{ left: `${lo}%` }}
          title={`Low threshold ${lo.toFixed(0)}%`}
        />
        <span
          className="absolute top-0 h-full w-0.5 bg-ink"
          style={{ left: `${hi}%` }}
          title={`High threshold ${hi.toFixed(0)}%`}
        />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-stone-500">
        <span>lo {lo.toFixed(0)}%</span>
        <span>hi {hi.toFixed(0)}%</span>
      </div>
    </div>
  );
}
