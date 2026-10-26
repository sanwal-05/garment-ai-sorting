import type { Decision, Disposition } from "../types";

export function DecisionBadge({ decision }: { decision: Decision }) {
  const styles: Record<Decision, string> = {
    ACCEPT: "bg-sage/15 text-sage ring-sage/30",
    REVIEW: "bg-amber-100 text-amber-800 ring-amber-300",
    REJECT: "bg-red-100 text-red-800 ring-red-300",
  };
  return (
    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${styles[decision]}`}>
      {decision}
    </span>
  );
}

export function DispositionBadge({ disposition }: { disposition: Disposition }) {
  return (
    <span className="inline-flex rounded-full bg-sand px-2.5 py-1 text-[11px] font-semibold tracking-wide text-ink ring-1 ring-stone-300">
      {disposition.replaceAll("_", " ")}
    </span>
  );
}
