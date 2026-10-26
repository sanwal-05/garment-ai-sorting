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

const SCORE_KEYS: ScoreKey[] = ["stain_present", "hole_present", "hole_major", "heavily_worn"];

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
        <h1 className="font-display text-4xl">Reviewer desk</h1>
        <p className="mt-2 text-sm text-stone-600">Sign in with the shared admin token. It stays in this tab only.</p>
        <form
          className="mt-8 space-y-4 rounded-3xl bg-white p-6 shadow-soft"
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
          <label className="block text-sm font-medium" htmlFor="token">
            Admin token
          </label>
          <input
            id="token"
            type="password"
            autoComplete="current-password"
            className="w-full rounded-xl border border-sand bg-mist px-3 py-3"
            {...loginForm.register("token")}
          />
          {loginForm.formState.errors.token ? (
            <p className="text-sm text-red-700">{loginForm.formState.errors.token.message}</p>
          ) : null}
          {loginError ? (
            <p className="text-sm text-red-700" role="alert">
              {loginError}
            </p>
          ) : null}
          <button type="submit" className="w-full rounded-full bg-ink py-3 text-mist">
            Enter desk
          </button>
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
  const listQuery = useQuery({
    queryKey: ["admin-returns", statusFilter, decisionFilter],
    queryFn: () => listReturns({ status: statusFilter || undefined, decision: decisionFilter || undefined }),
  });

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-clay">Internal</p>
          <h1 className="font-display text-4xl">Returns queue</h1>
        </div>
        <button type="button" className="text-sm text-stone-600 hover:text-ink" onClick={onSignOut}>
          Sign out
        </button>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <label className="text-sm">
          Status
          <select
            className="ml-2 rounded-lg border border-sand bg-white px-2 py-2"
            value={statusFilter}
            onChange={(event) => onStatus(event.target.value)}
          >
            <option value="">All</option>
            <option value="approved">approved</option>
            <option value="under_review">under_review</option>
            <option value="needs_more_info">needs_more_info</option>
            <option value="declined">declined</option>
            <option value="received">received</option>
          </select>
        </label>
        <label className="text-sm">
          Decision
          <select
            className="ml-2 rounded-lg border border-sand bg-white px-2 py-2"
            value={decisionFilter}
            onChange={(event) => onDecision(event.target.value)}
          >
            <option value="">All</option>
            <option value="ACCEPT">ACCEPT</option>
            <option value="REVIEW">REVIEW</option>
            <option value="REJECT">REJECT</option>
          </select>
        </label>
      </div>

      {listQuery.isLoading ? <p className="mt-8 text-stone-600">Loading returns…</p> : null}
      {listQuery.isError ? (
        <p className="mt-8 text-red-700" role="alert">
          {(listQuery.error as Error).message}
        </p>
      ) : null}

      {listQuery.data ? (
        <div className="mt-6 overflow-x-auto rounded-3xl bg-white shadow-soft">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-sand text-xs uppercase tracking-wide text-stone-500">
              <tr>
                <th className="px-4 py-3">Item</th>
                <th className="px-4 py-3">Claim</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Disposition</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {listQuery.data.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-stone-500">
                    No returns match these filters.
                  </td>
                </tr>
              ) : (
                listQuery.data.map((row) => {
                  const claim = CLAIM_OPTIONS.find((item) => item.value === row.claim_category)?.label ?? row.claim_category;
                  return (
                    <tr
                      key={row.id}
                      className="cursor-pointer border-b border-sand/70 hover:bg-mist"
                      onClick={() => onSelect(row.id)}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <img src={row.front_url} alt="" className="h-12 w-12 rounded-lg object-cover" />
                          <img src={row.back_url} alt="" className="h-12 w-12 rounded-lg object-cover" />
                          <span className="font-mono text-xs">{row.id.slice(0, 8)}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">{claim}</td>
                      <td className="px-4 py-3">
                        <DecisionBadge decision={row.model_decision ?? row.decision} />
                      </td>
                      <td className="px-4 py-3">
                        <DispositionBadge disposition={row.disposition} />
                      </td>
                      <td className="px-4 py-3">{row.status.replaceAll("_", " ")}</td>
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
      await queryClient.invalidateQueries({ queryKey: ["admin-returns"] });
      await queryClient.invalidateQueries({ queryKey: ["admin-return", id] });
    },
  });

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-ink/40" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
      <button className="h-full flex-1 cursor-default" aria-label="Close drawer" onClick={onClose} />
      <aside className="h-full w-full max-w-xl overflow-y-auto bg-white p-6 shadow-soft">
        <div className="flex items-start justify-between gap-4">
          <h2 id="drawer-title" className="font-display text-3xl">
            Return detail
          </h2>
          <button type="button" className="text-sm" onClick={onClose}>
            Close
          </button>
        </div>
        {detailQuery.isLoading ? <p className="mt-6">Loading…</p> : null}
        {detailQuery.isError ? (
          <p className="mt-6 text-red-700" role="alert">
            {(detailQuery.error as Error).message}
          </p>
        ) : null}
        {detailQuery.data ? (
          <DetailBody
            row={detailQuery.data}
            form={form}
            saving={mutation.isPending}
            saveError={mutation.isError ? (mutation.error as Error).message : null}
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
  onSave,
}: {
  row: AdminReturn;
  form: ReturnType<typeof useForm<z.infer<typeof overrideSchema>>>;
  saving: boolean;
  saveError: string | null;
  onSave: () => void;
}) {
  const claim = CLAIM_OPTIONS.find((item) => item.value === row.claim_category)?.label ?? row.claim_category;
  const trail = useMemo(() => {
    const items = [
      `Submitted ${row.created_at ?? "—"}`,
      `Model decided ${row.model_decision ?? row.decision} / ${row.model_disposition ?? row.disposition}`,
    ];
    if (row.reviewer_override) {
      items.push(`Reviewer overrode to ${row.decision} / ${row.disposition}${row.reviewer_note ? ` — ${row.reviewer_note}` : ""}`);
    }
    return items;
  }, [row]);

  return (
    <div className="mt-6 space-y-6">
      <div className="grid grid-cols-2 gap-3">
        <img src={row.front_url} alt="Front of garment" className="h-48 w-full rounded-2xl object-cover" />
        <img src={row.back_url} alt="Back of garment" className="h-48 w-full rounded-2xl object-cover" />
      </div>
      <div>
        <p className="text-xs uppercase tracking-wide text-stone-500">{claim}</p>
        <p className="mt-2 whitespace-pre-wrap text-sm">{row.claim_text}</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <DecisionBadge decision={row.decision} />
        <DispositionBadge disposition={row.disposition} />
        {row.reviewer_override ? <span className="text-xs text-clay">Human override</span> : null}
      </div>
      <div className="space-y-4">
        {SCORE_KEYS.map((key) => (
          <ScoreBar key={key} label={key} value={row.scores[key] ?? 0} thresholds={row.thresholds?.[key]} />
        ))}
      </div>
      <div>
        <h3 className="font-medium">Reasons</h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-stone-700">
          {row.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </div>
      <form className="space-y-3 rounded-2xl bg-mist p-4" onSubmit={(event) => (event.preventDefault(), onSave())}>
        <h3 className="font-medium">Override</h3>
        <label className="block text-sm">
          Decision
          <select className="mt-1 w-full rounded-lg border border-sand bg-white px-2 py-2" {...form.register("decision")}>
            {(["ACCEPT", "REVIEW", "REJECT"] as Decision[]).map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-sm">
          Disposition
          <select className="mt-1 w-full rounded-lg border border-sand bg-white px-2 py-2" {...form.register("disposition")}>
            {(["RESELL_AS_NEW", "RESELL_AS_USED", "REFURBISH", "RECYCLE", "INSPECT_MANUALLY"] as Disposition[]).map(
              (value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ),
            )}
          </select>
        </label>
        <label className="block text-sm">
          Note
          <textarea className="mt-1 w-full rounded-lg border border-sand bg-white px-2 py-2" rows={3} {...form.register("note")} />
        </label>
        {form.formState.errors.note ? (
          <p className="text-sm text-red-700">{form.formState.errors.note.message}</p>
        ) : null}
        {saveError ? (
          <p className="text-sm text-red-700" role="alert">
            {saveError}
          </p>
        ) : null}
        <button type="submit" className="rounded-full bg-ink px-4 py-2 text-sm text-mist disabled:opacity-60" disabled={saving}>
          {saving ? "Saving…" : "Save override"}
        </button>
      </form>
      <div>
        <h3 className="font-medium">Audit trail</h3>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-stone-700">
          {trail.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ol>
      </div>
    </div>
  );
}
