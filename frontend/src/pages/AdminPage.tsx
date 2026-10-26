import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import {
  clearAdminToken,
  getAdminToken,
  getReturn,
  listReturns,
  overrideReturn,
  probeAdminToken,
  setAdminToken,
} from "../api";
import { DecisionBadge, DispositionBadge } from "../components/Badges";
import ScoreBar from "../components/ScoreBar";
import { CLAIM_OPTIONS, type AdminReturn, type Decision, type Disposition, type ScoreKey } from "../types";

const SCORE_KEYS: ScoreKey[] = ["stain_present", "hole_major", "heavily_worn"];
const DEV_TOKEN = "dev-admin-token";

const loginSchema = z.object({
  token: z.string().min(4, "Enter the admin token."),
});

const overrideSchema = z.object({
  decision: z.enum(["ACCEPT", "REVIEW", "REJECT"]),
  disposition: z.enum(["RESELL_AS_NEW", "RESELL_AS_USED", "REFURBISH", "RECYCLE", "INSPECT_MANUALLY"]),
  note: z.string().trim().min(3, "Add a short note for the audit trail.").max(1000),
});

export default function AdminPage() {
  const [authed, setAuthed] = useState(Boolean(getAdminToken()));
  const [statusFilter, setStatusFilter] = useState("");
  const [decisionFilter, setDecisionFilter] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loginError, setLoginError] = useState<string | null>(null);

  const loginForm = useForm<z.infer<typeof loginSchema>>({
    resolver: zodResolver(loginSchema),
    defaultValues: { token: "" },
  });

  if (!authed) {
    return (
      <div className="mx-auto max-w-md px-4 py-16">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-secondary" />
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-600">Reviewer desk</p>
        </div>
        <h1 className="mt-2 font-display text-4xl text-charcoal">Sign in to console</h1>
        <p className="mt-2 text-sm text-stone-600 leading-relaxed">
          Sign in with your configured admin bearer token. In local development, this is defined in your{" "}
          <code className="rounded bg-sand/60 px-1.5 py-0.5 font-mono text-xs text-charcoal">.env</code> file.
        </p>

        <form
          className="mt-8 space-y-4 rounded-3xl bg-white p-7 shadow-soft border border-sand/60"
          onSubmit={loginForm.handleSubmit(async ({ token }) => {
            setLoginError(null);
            try {
              await probeAdminToken(token);
              setAdminToken(token);
              setAuthed(true);
            } catch (error) {
              setLoginError((error as Error).message);
            }
          })}
        >
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-charcoal" htmlFor="token">
                Admin token
              </label>
              <button
                type="button"
                className="text-xs font-medium text-sage-dark hover:underline"
                onClick={() => loginForm.setValue("token", DEV_TOKEN)}
              >
                Use dev token ({DEV_TOKEN})
              </button>
            </div>
            <input
              id="token"
              type="password"
              placeholder="Enter token..."
              autoComplete="current-password"
              className="w-full rounded-xl border border-sand bg-mist px-3.5 py-3 text-sm focus:border-sage focus:ring-2 focus:ring-sage/40 transition-all font-mono"
              {...loginForm.register("token")}
            />
            {loginForm.formState.errors.token ? (
              <p className="mt-1.5 text-xs text-critical flex items-center gap-1 font-medium">
                <span>⚠</span> {loginForm.formState.errors.token.message}
              </p>
            ) : null}
            {loginError ? (
              <p className="mt-2 rounded-xl bg-critical/15 border border-critical/30 p-2.5 text-xs text-[#73291D] font-medium" role="alert">
                <span>⚠</span> {loginError}
              </p>
            ) : null}
          </div>

          <button
            type="submit"
            className="w-full rounded-full bg-charcoal py-3 text-sm font-semibold text-mist hover:bg-black transition-all shadow-sm"
          >
            Enter reviewer desk →
          </button>

          <div className="pt-2 text-center">
            <span className="text-[11px] text-stone-500">
              Default token: <code className="font-mono font-medium text-charcoal">{DEV_TOKEN}</code>
            </span>
          </div>
        </form>
      </div>
    );
  }

  return (
    <Desk
      statusFilter={statusFilter}
      decisionFilter={decisionFilter}
      selectedId={selectedId}
      onStatus={setStatusFilter}
      onDecision={setDecisionFilter}
      onSelect={setSelectedId}
      onSignOut={() => {
        clearAdminToken();
        setAuthed(false);
      }}
    />
  );
}

function Desk({
  statusFilter,
  decisionFilter,
  selectedId,
  onStatus,
  onDecision,
  onSelect,
  onSignOut,
}: {
  statusFilter: string;
  decisionFilter: string;
  selectedId: string | null;
  onStatus: (value: string) => void;
  onDecision: (value: string) => void;
  onSelect: (id: string | null) => void;
  onSignOut: () => void;
}) {
  const [search, setSearch] = useState("");
  const listQuery = useQuery({
    queryKey: ["admin-returns", statusFilter, decisionFilter],
    queryFn: () => listReturns({ status: statusFilter || undefined, decision: decisionFilter || undefined }),
  });

  const filteredData = useMemo(() => {
    if (!listQuery.data) return [];
    if (!search.trim()) return listQuery.data;
    const q = search.toLowerCase();
    return listQuery.data.filter(
      (item) =>
        item.id.toLowerCase().includes(q) ||
        item.claim_text.toLowerCase().includes(q) ||
        item.claim_category.toLowerCase().includes(q)
    );
  }, [listQuery.data, search]);

  const stats = useMemo(() => {
    const all = listQuery.data ?? [];
    return {
      total: all.length,
      underReview: all.filter((r) => r.decision === "REVIEW" || r.status === "under_review").length,
      accepted: all.filter((r) => r.decision === "ACCEPT").length,
      rejected: all.filter((r) => r.decision === "REJECT").length,
    };
  }, [listQuery.data]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-secondary" />
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-600">Internal triage</p>
          </div>
          <h1 className="mt-1 font-display text-4xl text-charcoal">Returns queue</h1>
        </div>
        <button
          type="button"
          className="rounded-full border border-sand bg-white px-4 py-1.5 text-xs font-medium text-stone-700 hover:bg-sand/30 hover:text-charcoal transition-all shadow-sm"
          onClick={onSignOut}
        >
          Sign out
        </button>
      </div>

      {/* Summary KPI Cards */}
      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-2xl border border-sand/70 bg-white p-4 shadow-card">
          <span className="text-xs font-medium text-stone-500 uppercase tracking-wider">Total Claims</span>
          <p className="mt-1 text-2xl font-semibold text-charcoal">{stats.total}</p>
        </div>
        <div className="rounded-2xl border border-warning/40 bg-warning/10 p-4 shadow-card">
          <span className="text-xs font-medium text-warning-dark uppercase tracking-wider">Needs Review</span>
          <p className="mt-1 text-2xl font-semibold text-warning-dark">{stats.underReview}</p>
        </div>
        <div className="rounded-2xl border border-sage/40 bg-sage/10 p-4 shadow-card">
          <span className="text-xs font-medium text-sage-dark uppercase tracking-wider">Accepted</span>
          <p className="mt-1 text-2xl font-semibold text-sage-dark">{stats.accepted}</p>
        </div>
        <div className="rounded-2xl border border-critical/40 bg-critical/10 p-4 shadow-card">
          <span className="text-xs font-medium text-critical-dark uppercase tracking-wider">Rejected</span>
          <p className="mt-1 text-2xl font-semibold text-critical-dark">{stats.rejected}</p>
        </div>
      </div>

      {/* Filters & Search */}
      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-sand/70 bg-white p-3.5 shadow-card">
        <div className="flex flex-1 min-w-[240px] items-center gap-2 rounded-xl border border-sand bg-mist px-3 py-2 text-sm">
          <span className="text-stone-400">🔍</span>
          <input
            type="text"
            placeholder="Search return ID, keyword or claim..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-transparent text-xs text-charcoal outline-none placeholder:text-stone-400"
          />
          {search ? (
            <button
              type="button"
              onClick={() => setSearch("")}
              className="text-xs text-stone-400 hover:text-charcoal"
            >
              ✕
            </button>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs font-medium text-stone-600 flex items-center gap-1.5">
            Status:
            <select
              className="rounded-lg border border-sand bg-mist px-2.5 py-1.5 text-xs text-charcoal focus:border-sage outline-none"
              value={statusFilter}
              onChange={(event) => onStatus(event.target.value)}
            >
              <option value="">All statuses</option>
              <option value="approved">Approved</option>
              <option value="under_review">Under review</option>
              <option value="needs_more_info">Needs more info</option>
              <option value="declined">Declined</option>
              <option value="received">Received</option>
            </select>
          </label>
          <label className="text-xs font-medium text-stone-600 flex items-center gap-1.5">
            Decision:
            <select
              className="rounded-lg border border-sand bg-mist px-2.5 py-1.5 text-xs text-charcoal focus:border-sage outline-none"
              value={decisionFilter}
              onChange={(event) => onDecision(event.target.value)}
            >
              <option value="">All decisions</option>
              <option value="ACCEPT">ACCEPT</option>
              <option value="REVIEW">REVIEW</option>
              <option value="REJECT">REJECT</option>
            </select>
          </label>
        </div>
      </div>

      {listQuery.isLoading ? (
        <div className="mt-8 text-center text-sm text-stone-600">
          <div className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-sage border-t-transparent" />
          <p className="mt-2">Loading returns queue…</p>
        </div>
      ) : null}

      {listQuery.isError ? (
        <div className="mt-8 rounded-2xl bg-critical/15 border border-critical/30 p-4 text-xs font-medium text-critical" role="alert">
          {(listQuery.error as Error).message}
        </div>
      ) : null}

      {listQuery.data ? (
        <div className="mt-6 overflow-x-auto rounded-3xl bg-white shadow-soft border border-sand/60">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-sand bg-mist/60 text-xs uppercase tracking-wide text-stone-500">
              <tr>
                <th className="px-5 py-3.5">Garment Photos</th>
                <th className="px-5 py-3.5">Claim Reason</th>
                <th className="px-5 py-3.5">AI Decision</th>
                <th className="px-5 py-3.5">Disposition</th>
                <th className="px-5 py-3.5">Current Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sand/50">
              {filteredData.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-5 py-12 text-center text-stone-500">
                    <p className="text-sm font-medium">No returns found</p>
                    <p className="mt-1 text-xs text-stone-400">
                      {search || statusFilter || decisionFilter
                        ? "Try clearing your filters or search term."
                        : "Submit a return from the customer page to see items populate here."}
                    </p>
                  </td>
                </tr>
              ) : (
                filteredData.map((row) => {
                  const claim = CLAIM_OPTIONS.find((item) => item.value === row.claim_category)?.label ?? row.claim_category;
                  const isSelected = row.id === selectedId;
                  return (
                    <tr
                      key={row.id}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? "bg-sage/10" : "hover:bg-mist"
                      }`}
                      onClick={() => onSelect(row.id)}
                    >
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-3">
                          <img src={row.front_url} alt="" className="h-12 w-12 rounded-xl object-cover border border-sand shadow-sm" />
                          <img src={row.back_url} alt="" className="h-12 w-12 rounded-xl object-cover border border-sand shadow-sm" />
                          <div>
                            <span className="font-mono text-xs font-medium text-charcoal">{row.id.slice(0, 8)}</span>
                            {row.reviewer_override ? (
                              <span className="block text-[10px] text-terracotta font-semibold">Overridden</span>
                            ) : null}
                          </div>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="font-medium text-charcoal">{claim}</span>
                        <p className="line-clamp-1 max-w-xs text-xs text-stone-500">{row.claim_text}</p>
                      </td>
                      <td className="px-5 py-3.5">
                        <DecisionBadge decision={row.model_decision ?? row.decision} />
                      </td>
                      <td className="px-5 py-3.5">
                        <DispositionBadge disposition={row.disposition} />
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="capitalize text-xs font-medium text-charcoal">
                          {row.status.replaceAll("_", " ")}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      ) : null}

      {selectedId ? <DetailDrawer id={selectedId} onClose={() => onSelect(null)} /> : null}
    </div>
  );
}

function DetailDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const detailQuery = useQuery({
    queryKey: ["admin-return", id],
    queryFn: () => getReturn(id),
  });
  const form = useForm<z.infer<typeof overrideSchema>>({
    resolver: zodResolver(overrideSchema),
    values: detailQuery.data
      ? {
          decision: detailQuery.data.decision,
          disposition: detailQuery.data.disposition,
          note: detailQuery.data.reviewer_note ?? "",
        }
      : undefined,
  });

  const mutation = useMutation({
    mutationFn: (values: z.infer<typeof overrideSchema>) => overrideReturn(id, values),
    onSuccess: async () => {
      setSuccessMsg("Decision override saved to audit trail.");
      await queryClient.invalidateQueries({ queryKey: ["admin-returns"] });
      await queryClient.invalidateQueries({ queryKey: ["admin-return", id] });
      setTimeout(() => setSuccessMsg(null), 4000);
    },
  });

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-charcoal/40 backdrop-blur-xs" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
      <button className="h-full flex-1 cursor-default" aria-label="Close drawer" onClick={onClose} />
      <aside className="h-full w-full max-w-xl overflow-y-auto bg-white p-6 shadow-soft border-l border-sand">
        <div className="flex items-start justify-between gap-4 border-b border-sand/40 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-secondary" />
              <span className="text-[11px] font-semibold uppercase tracking-wider text-stone-500">Inspection</span>
            </div>
            <h2 id="drawer-title" className="mt-1 font-display text-2xl text-charcoal">
              Return {id.slice(0, 8).toUpperCase()}
            </h2>
          </div>
          <button
            type="button"
            className="rounded-full border border-sand bg-mist px-3 py-1 text-xs font-medium text-stone-700 hover:bg-sand/30"
            onClick={onClose}
          >
            ✕ Close
          </button>
        </div>
        {detailQuery.isLoading ? <p className="mt-6 text-sm text-stone-500">Loading details…</p> : null}
        {detailQuery.isError ? (
          <p className="mt-6 text-xs text-critical font-medium" role="alert">
            {(detailQuery.error as Error).message}
          </p>
        ) : null}
        {detailQuery.data ? (
          <DetailBody
            row={detailQuery.data}
            form={form}
            saving={mutation.isPending}
            saveError={mutation.isError ? (mutation.error as Error).message : null}
            successMsg={successMsg}
            onSave={form.handleSubmit((values) => mutation.mutate(values))}
          />
        ) : null}
      </aside>
    </div>
  );
}

function DetailBody({
  row,
  form,
  saving,
  saveError,
  successMsg,
  onSave,
}: {
  row: AdminReturn;
  form: ReturnType<typeof useForm<z.infer<typeof overrideSchema>>>;
  saving: boolean;
  saveError: string | null;
  successMsg: string | null;
  onSave: () => void;
}) {
  const [zoomUrl, setZoomUrl] = useState<string | null>(null);
  const claim = CLAIM_OPTIONS.find((item) => item.value === row.claim_category)?.label ?? row.claim_category;
  const trail = useMemo(() => {
    const items = [
      `Submitted: ${row.created_at ?? "—"}`,
      `Model output: ${row.model_decision ?? row.decision} / ${row.model_disposition ?? row.disposition}`,
    ];
    if (row.reviewer_override) {
      items.push(`Reviewer overrode to ${row.decision} / ${row.disposition}${row.reviewer_note ? ` — "${row.reviewer_note}"` : ""}`);
    }
    return items;
  }, [row]);

  return (
    <div className="mt-6 space-y-6">
      {/* Lightbox Modal */}
      {zoomUrl ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setZoomUrl(null)}
        >
          <div className="relative max-h-[90vh] max-w-[90vw]" onClick={(e) => e.stopPropagation()}>
            <img src={zoomUrl} alt="Inspection view" className="max-h-[85vh] max-w-full rounded-2xl object-contain shadow-2xl" />
            <button
              type="button"
              className="absolute top-3 right-3 rounded-full bg-white/90 px-3 py-1 text-xs font-bold text-charcoal shadow hover:bg-white"
              onClick={() => setZoomUrl(null)}
            >
              ✕ Close Zoom
            </button>
          </div>
        </div>
      ) : null}

      {/* Garment Images with Zoom Trigger */}
      <div>
        <p className="mb-1 text-xs font-medium text-stone-500">Inspection photos (click to zoom)</p>
        <div className="grid grid-cols-2 gap-3">
          <div
            className="group relative cursor-zoom-in overflow-hidden rounded-2xl border border-sand bg-mist"
            onClick={() => setZoomUrl(row.front_url)}
          >
            <span className="absolute top-2 left-2 z-10 rounded-md bg-charcoal/80 px-2 py-0.5 text-[10px] font-medium text-white backdrop-blur">
              Front View
            </span>
            <img
              src={row.front_url}
              alt="Front of garment"
              className="h-44 w-full object-cover transition-transform group-hover:scale-105"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-charcoal/20 opacity-0 group-hover:opacity-100 transition-opacity">
              <span className="rounded-full bg-white/90 px-2.5 py-1 text-[11px] font-semibold text-charcoal shadow-sm">
                🔍 Zoom
              </span>
            </div>
          </div>
          <div
            className="group relative cursor-zoom-in overflow-hidden rounded-2xl border border-sand bg-mist"
            onClick={() => setZoomUrl(row.back_url)}
          >
            <span className="absolute top-2 left-2 z-10 rounded-md bg-charcoal/80 px-2 py-0.5 text-[10px] font-medium text-white backdrop-blur">
              Back View
            </span>
            <img
              src={row.back_url}
              alt="Back of garment"
              className="h-44 w-full object-cover transition-transform group-hover:scale-105"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-charcoal/20 opacity-0 group-hover:opacity-100 transition-opacity">
              <span className="rounded-full bg-white/90 px-2.5 py-1 text-[11px] font-semibold text-charcoal shadow-sm">
                🔍 Zoom
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Claim details */}
      <div className="rounded-2xl border border-sand/70 bg-mist p-4">
        <span className="text-[11px] uppercase tracking-wider font-semibold text-secondary-dark">{claim}</span>
        <p className="mt-1.5 whitespace-pre-wrap text-sm text-charcoal leading-relaxed">{row.claim_text}</p>
      </div>

      {/* Decision badges */}
      <div className="flex flex-wrap items-center gap-2">
        <DecisionBadge decision={row.decision} />
        <DispositionBadge disposition={row.disposition} />
        {row.reviewer_override ? (
          <span className="rounded-full bg-critical/15 px-2.5 py-1 text-xs font-semibold text-[#73291D]">
            Human override active
          </span>
        ) : null}
      </div>

      {/* Model Scores with Dual Thresholds */}
      <div className="rounded-2xl border border-sand/70 bg-white p-4 shadow-card">
        <h3 className="font-display text-lg text-charcoal mb-3">Model Defect Scores</h3>
        <div className="space-y-4">
          {SCORE_KEYS.map((key) => (
            <ScoreBar key={key} label={key} value={row.scores[key] ?? 0} thresholds={row.thresholds?.[key]} />
          ))}
        </div>
      </div>

      {/* Policy Reasons */}
      <div className="rounded-2xl border border-sand/70 bg-white p-4 shadow-card">
        <h3 className="font-display text-lg text-charcoal mb-2">Automated Policy Reasons</h3>
        <ul className="space-y-2 text-sm text-stone-700">
          {row.reasons.map((reason) => (
            <li key={reason} className="flex items-start gap-2">
              <span className="text-secondary font-bold">•</span>
              <span>{reason}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Override Form */}
      <form
        className="space-y-4 rounded-2xl border border-sand/80 bg-mist p-5 shadow-card"
        onSubmit={(event) => {
          event.preventDefault();
          onSave();
        }}
      >
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-warning" />
          <h3 className="font-display text-lg text-charcoal">Reviewer Override</h3>
        </div>

        {successMsg ? (
          <div className="rounded-xl bg-sage/20 border border-sage/40 p-3 text-xs font-semibold text-[#25442A] flex items-center gap-2">
            <span>✓</span> {successMsg}
          </div>
        ) : null}

        <div className="grid grid-cols-2 gap-3">
          <label className="block text-xs font-medium text-stone-600">
            Decision
            <select
              className="mt-1 w-full rounded-xl border border-sand bg-white px-3 py-2 text-xs font-medium text-charcoal focus:border-sage outline-none"
              {...form.register("decision")}
            >
              {(["ACCEPT", "REVIEW", "REJECT"] as Decision[]).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs font-medium text-stone-600">
            Disposition
            <select
              className="mt-1 w-full rounded-xl border border-sand bg-white px-3 py-2 text-xs font-medium text-charcoal focus:border-sage outline-none"
              {...form.register("disposition")}
            >
              {(["RESELL_AS_NEW", "RESELL_AS_USED", "REFURBISH", "RECYCLE", "INSPECT_MANUALLY"] as Disposition[]).map(
                (value) => (
                  <option key={value} value={value}>
                    {value.replaceAll("_", " ")}
                  </option>
                ),
              )}
            </select>
          </label>
        </div>

        <label className="block text-xs font-medium text-stone-600">
          Audit Note (required)
          <textarea
            className="mt-1 w-full rounded-xl border border-sand bg-white p-3 text-xs text-charcoal focus:border-sage outline-none leading-relaxed"
            rows={3}
            placeholder="Document why this decision or disposition is being overridden..."
            {...form.register("note")}
          />
        </label>
        {form.formState.errors.note ? (
          <p className="text-xs text-critical font-medium">⚠ {form.formState.errors.note.message}</p>
        ) : null}
        {saveError ? (
          <p className="text-xs text-critical font-medium" role="alert">
            ⚠ {saveError}
          </p>
        ) : null}
        <button
          type="submit"
          className="rounded-full bg-charcoal px-5 py-2.5 text-xs font-semibold text-mist hover:bg-black disabled:opacity-60 transition-all shadow-sm flex items-center gap-2"
          disabled={saving}
        >
          {saving ? (
            <>
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-mist border-t-transparent inline-block" />
              Saving override…
            </>
          ) : (
            "Save decision override"
          )}
        </button>
      </form>

      {/* Audit Trail */}
      <div className="rounded-2xl border border-sand/70 bg-white p-4 shadow-card">
        <h3 className="font-display text-lg text-charcoal mb-2">Audit Trail</h3>
        <ol className="space-y-2 text-xs text-stone-600">
          {trail.map((item, idx) => (
            <li key={item} className="flex items-start gap-2">
              <span className="font-mono text-stone-400 font-semibold">{idx + 1}.</span>
              <span className="leading-relaxed">{item}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
