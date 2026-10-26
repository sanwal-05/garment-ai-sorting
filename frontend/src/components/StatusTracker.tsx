import type { CustomerStatus } from "../types";

const STEPS: { key: CustomerStatus; label: string }[] = [
  { key: "received", label: "Received" },
  { key: "under_review", label: "Under review" },
  { key: "needs_more_info", label: "Needs more info" },
  { key: "approved", label: "Approved" },
  { key: "declined", label: "Declined" },
];

const COPY: Record<CustomerStatus, string> = {
  received: "We have your photos and are checking the item.",
  under_review: "A specialist is taking a closer look.",
  needs_more_info: "We may need a clearer photo or a little more detail.",
  approved: "Your return has been approved.",
  declined: "We could not approve this return from the photos provided.",
};

export default function StatusTracker({ status }: { status: CustomerStatus }) {
  const terminal = status === "approved" || status === "declined" || status === "needs_more_info";

  return (
    <div className="rounded-2xl border border-sand bg-mist p-5" aria-live="polite">
      <p className="font-display text-2xl">{status.replaceAll("_", " ")}</p>
      <p className="mt-1 text-sm text-stone-600">{COPY[status]}</p>
      <ol className="mt-5 grid gap-2 sm:grid-cols-5">
        {STEPS.map((step) => {
          const active = step.key === status;
          const muted =
            (terminal && !active) ||
            (status === "received" && step.key !== "received" && step.key !== "under_review");
          return (
            <li
              key={step.key}
              className={`rounded-xl px-3 py-2 text-center text-xs font-medium ${
                active ? "bg-ink text-mist" : muted ? "bg-sand/60 text-stone-500" : "bg-sand text-ink"
              }`}
            >
              {step.label}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
