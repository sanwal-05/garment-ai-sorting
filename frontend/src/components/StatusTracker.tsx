import type { CustomerStatus } from "../types";

const STEPS: { key: CustomerStatus; label: string }[] = [
  { key: "received", label: "Received" },
  { key: "under_review", label: "Under review" },
  { key: "needs_more_info", label: "Needs more info" },
  { key: "approved", label: "Approved" },
  { key: "declined", label: "Declined" },
];

const STATUS_CONFIG: Record<
  CustomerStatus,
  {
    title: string;
    description: string;
    badgeBg: string;
    textColor: string;
    borderColor: string;
    accentDot: string;
    icon: string;
  }
> = {
  received: {
    title: "Received & Queued",
    description: "We have your photos and garment claim. Automated vetting is in progress.",
    badgeBg: "bg-secondary/15",
    textColor: "text-[#274454]",
    borderColor: "border-secondary/40",
    accentDot: "bg-secondary",
    icon: "📦",
  },
  under_review: {
    title: "Under Specialist Review",
    description: "A garment specialist is performing a detailed visual inspection of your photos.",
    badgeBg: "bg-warning/15",
    textColor: "text-[#6D5015]",
    borderColor: "border-warning/40",
    accentDot: "bg-warning",
    icon: "🔍",
  },
  needs_more_info: {
    title: "More Information Needed",
    description: "We may need a clearer photo or a brief follow-up before finishing your return.",
    badgeBg: "bg-warning/20",
    textColor: "text-[#6D5015]",
    borderColor: "border-warning/50",
    accentDot: "bg-warning",
    icon: "💬",
  },
  approved: {
    title: "Return Approved",
    description: "Your return has been verified and approved. Check your email for return shipping labels.",
    badgeBg: "bg-sage/20",
    textColor: "text-[#25442A]",
    borderColor: "border-sage/50",
    accentDot: "bg-sage",
    icon: "✓",
  },
  declined: {
    title: "Return Not Approved",
    description: "Based on policy and the provided photos, this return does not qualify for processing.",
    badgeBg: "bg-critical/15",
    textColor: "text-[#73291D]",
    borderColor: "border-critical/40",
    accentDot: "bg-critical",
    icon: "✕",
  },
};

export default function StatusTracker({ status }: { status: CustomerStatus }) {
  const current = STATUS_CONFIG[status] ?? STATUS_CONFIG.received;

  return (
    <div className={`rounded-3xl border ${current.borderColor} ${current.badgeBg} p-6 transition-all`} aria-live="polite">
      <div className="flex items-start gap-4">
        <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-white shadow-sm text-xl font-bold">
          {current.icon}
        </span>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${current.accentDot} animate-pulse`} />
            <span className="text-xs font-semibold uppercase tracking-wider text-stone-500">Live Status</span>
          </div>
          <h3 className={`mt-1 font-display text-2xl font-semibold ${current.textColor}`}>{current.title}</h3>
          <p className="mt-1 text-sm text-charcoal/80 leading-relaxed">{current.description}</p>
        </div>
      </div>

      <div className="mt-6 border-t border-black/5 pt-4">
        <p className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-2">Stage progression</p>
        <div className="flex flex-wrap gap-2">
          {STEPS.map((step) => {
            const active = step.key === status;
            return (
              <span
                key={step.key}
                className={`rounded-full px-3 py-1 text-xs font-medium transition-all ${
                  active
                    ? "bg-charcoal text-mist shadow-sm font-semibold"
                    : "bg-white/60 text-stone-600 border border-sand/50"
                }`}
              >
                {step.label}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
