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

  // Determine accent level based on thresholds
  const isHigh = pct >= hi;
  const isMid = pct >= lo && !isHigh;
  const barColor = isHigh ? "bg-critical" : isMid ? "bg-warning" : "bg-sage";
  const statusLabel = isHigh ? "High" : isMid ? "Moderate" : "Low";
  const badgeColor = isHigh
    ? "bg-critical/15 text-[#73291D]"
    : isMid
      ? "bg-warning/20 text-[#6D5015]"
      : "bg-sage/15 text-[#25442A]";

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between text-sm">
        <div className="flex items-center gap-2">
          <span className="font-medium capitalize text-charcoal">{label.replaceAll("_", " ")}</span>
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${badgeColor}`}>
            {statusLabel}
          </span>
        </div>
        <span className="tabular-nums font-semibold text-charcoal">{pct.toFixed(0)}%</span>
      </div>
      <div
        className="relative h-3 overflow-hidden rounded-full bg-sand/70 ring-1 ring-sand"
        role="img"
        aria-label={`${label} ${pct.toFixed(0)} percent, low threshold ${lo.toFixed(0)}, high threshold ${hi.toFixed(0)}`}
      >
        <div
          className={`absolute inset-y-0 left-0 rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
        <span
          className="absolute top-0 h-full w-0.5 bg-charcoal/50"
          style={{ left: `${lo}%` }}
          title={`Low threshold ${lo.toFixed(0)}%`}
        />
        <span
          className="absolute top-0 h-full w-0.5 bg-charcoal/80"
          style={{ left: `${hi}%` }}
          title={`High threshold ${hi.toFixed(0)}%`}
        />
      </div>
      <div className="flex justify-between text-[11px] text-stone-500">
        <span>lo {lo.toFixed(0)}%</span>
        <span className="text-stone-400">threshold zones</span>
        <span>hi {hi.toFixed(0)}%</span>
      </div>
    </div>
  );
}
