import type { Decision, Disposition } from "../types";

export function DecisionBadge({ decision }: { decision: Decision }) {
  const styles: Record<Decision, string> = {
    ACCEPT: "bg-sage/20 text-[#25442A] ring-sage/50",
    REVIEW: "bg-warning/20 text-[#6D5015] ring-warning/50",
    REJECT: "bg-critical/20 text-[#73291D] ring-critical/50",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${styles[decision]}`}>
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          decision === "ACCEPT" ? "bg-sage" : decision === "REVIEW" ? "bg-warning" : "bg-critical"
        }`}
      />
      {decision}
    </span>
  );
}

export function DispositionBadge({ disposition }: { disposition: Disposition }) {
  const styleMap: Record<Disposition, string> = {
    RESELL_AS_NEW: "bg-sage/15 text-[#25442A] ring-sage/35",
    RESELL_AS_USED: "bg-secondary/20 text-[#274454] ring-secondary/40",
    REFURBISH: "bg-warning/15 text-[#6D5015] ring-warning/35",
    RECYCLE: "bg-critical/15 text-[#73291D] ring-critical/35",
    INSPECT_MANUALLY: "bg-sand text-charcoal ring-sand/80",
  };

  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold tracking-wide ring-1 ${
        styleMap[disposition] ?? "bg-sand text-charcoal ring-stone-300"
      }`}
    >
      {disposition.replaceAll("_", " ")}
    </span>
  );
}
