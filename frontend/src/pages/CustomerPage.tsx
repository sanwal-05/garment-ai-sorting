import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { z } from "zod";
import { getReturnStatus, submitReturn } from "../api";
import PhotoDropzone from "../components/PhotoDropzone";
import StatusTracker from "../components/StatusTracker";
import { CLAIM_OPTIONS, type CustomerStatus } from "../types";

const schema = z.object({
  front: z.custom<File>((value) => value instanceof File, "Add a front photo of the full item."),
  back: z.custom<File>((value) => value instanceof File, "Add a back photo of the full item."),
  claim_category: z.enum([
    "arrived_stained",
    "arrived_damaged",
    "poor_quality",
    "wrong_size",
    "not_as_described",
    "changed_mind",
  ]),
  claim_text: z
    .string()
    .trim()
    .min(15, "Please tell us a little more (at least 15 characters).")
    .max(500, "Please keep this under 500 characters."),
  confirm: z.literal(true, {
    errorMap: () => ({ message: "Please confirm the photos show the item as received." }),
  }),
});

type FormValues = z.infer<typeof schema>;

const TIPS = [
  "Lay the item flat on a plain background",
  "Use good daylight, not a yellow lamp",
  "Keep the whole garment in frame",
  "Do not use filters or beauty modes",
  "Photograph the same item that was ordered",
];

export default function CustomerPage() {
  const [step, setStep] = useState(1);
  const [returnId, setReturnId] = useState<string | null>(null);
  const [frontPreview, setFrontPreview] = useState<string | null>(null);
  const [backPreview, setBackPreview] = useState<string | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      claim_text: "",
    },
    mode: "onTouched",
  });

  const front = form.watch("front");
  const back = form.watch("back");
  const claimText = form.watch("claim_text") ?? "";

  useEffect(() => {
    if (!front) {
      setFrontPreview(null);
      return;
    }
    const url = URL.createObjectURL(front);
    setFrontPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [front]);

  useEffect(() => {
    if (!back) {
      setBackPreview(null);
      return;
    }
    const url = URL.createObjectURL(back);
    setBackPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [back]);

  const mutation = useMutation({
    mutationFn: async (values: FormValues) => {
      const data = new FormData();
      data.append("front", values.front);
      data.append("back", values.back);
      data.append("claim_category", values.claim_category);
      data.append("claim_text", values.claim_text);
      return submitReturn(data);
    },
    onSuccess: (result) => setReturnId(result.id),
  });

  async function goNext() {
    if (step === 1) {
      const ok = await form.trigger(["front", "back"]);
      if (ok) setStep(2);
      return;
    }
    if (step === 2) {
      const ok = await form.trigger(["claim_category", "claim_text"]);
      if (ok) setStep(3);
    }
  }

  const claimLabel = CLAIM_OPTIONS.find((item) => item.value === form.getValues("claim_category"))?.label;

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-sage" />
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-600">Customer return</p>
      </div>
      <h1 className="mt-2 font-display text-4xl sm:text-5xl text-charcoal">Send photos. We’ll check the item fairly.</h1>
      <p className="mt-3 max-w-2xl text-stone-600 leading-relaxed">
        Three simple steps. Your photos are checked against the return reason you provide. Fast, transparent, and fair.
      </p>

      {returnId ? (
        <Confirmation returnId={returnId} onReset={() => {
          form.reset();
          setReturnId(null);
          setStep(1);
        }} />
      ) : (
        <form
          className="mt-8 grid gap-6 lg:grid-cols-[1fr_280px]"
          onSubmit={form.handleSubmit(
            (values) => mutation.mutate(values),
            (errors) => {
              if (errors.front || errors.back) {
                setStep(1);
              } else if (errors.claim_category || errors.claim_text) {
                setStep(2);
              }
            }
          )}
          noValidate
        >
          <section className="rounded-3xl bg-white p-6 shadow-soft border border-sand/50">
            <ol className="mb-6 flex gap-2 text-sm" aria-label="Form steps">
              {["Photos", "Reason", "Review"].map((label, index) => {
                const n = index + 1;
                const isCurrent = n === step;
                const isDone = n < step;
                return (
                  <li key={label} className="flex-1">
                    <button
                      type="button"
                      disabled={!isDone && !isCurrent}
                      onClick={() => {
                        if (isDone) setStep(n);
                      }}
                      className={`w-full rounded-full px-3 py-2 text-center text-xs font-semibold transition-all border ${
                        isCurrent
                          ? "bg-charcoal text-white border-charcoal shadow-sm ring-2 ring-charcoal/20"
                          : isDone
                            ? "bg-sage/20 text-[#19351E] border-sage/60 hover:bg-sage/35 cursor-pointer shadow-xs"
                            : "bg-white text-stone-600 border-sand/90 cursor-default"
                      }`}
                      aria-current={isCurrent ? "step" : undefined}
                    >
                      {isDone ? "✓ " : `${n}. `}{label}
                    </button>
                  </li>
                );
              })}
            </ol>

            {step === 1 ? (
              <div className="grid gap-5 md:grid-cols-2">
                <Controller
                  control={form.control}
                  name="front"
                  render={({ field, fieldState }) => (
                    <PhotoDropzone
                      label="Front of the full item"
                      file={field.value ?? null}
                      previewUrl={frontPreview}
                      error={fieldState.error?.message}
                      onChange={field.onChange}
                    />
                  )}
                />
                <Controller
                  control={form.control}
                  name="back"
                  render={({ field, fieldState }) => (
                    <PhotoDropzone
                      label="Back of the full item"
                      file={field.value ?? null}
                      previewUrl={backPreview}
                      error={fieldState.error?.message}
                      onChange={field.onChange}
                    />
                  )}
                />
              </div>
            ) : null}

            {step === 2 ? (
              <div className="space-y-5">
                <div>
                  <label htmlFor="claim" className="mb-2 block text-sm font-medium text-charcoal">
                    Why are you returning this?
                  </label>
                  <select
                    id="claim"
                    className="w-full rounded-xl border border-sand bg-mist px-3.5 py-3 text-sm focus:border-sage focus:ring-2 focus:ring-sage/40 transition-all"
                    {...form.register("claim_category")}
                  >
                    <option value="">Select a reason</option>
                    {CLAIM_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {form.formState.errors.claim_category ? (
                    <p className="mt-2 text-xs font-medium text-critical flex items-center gap-1" role="alert">
                      <span>⚠</span> Please choose a return reason.
                    </p>
                  ) : null}
                </div>
                <div>
                  <label htmlFor="story" className="mb-2 block text-sm font-medium text-charcoal">
                    Tell us what happened
                  </label>
                  <textarea
                    id="story"
                    rows={6}
                    maxLength={500}
                    className="w-full rounded-xl border border-sand bg-mist px-3.5 py-3 text-sm focus:border-sage focus:ring-2 focus:ring-sage/40 transition-all leading-relaxed"
                    placeholder="Where is the stain or tear? When did you notice it? Provide specific details."
                    {...form.register("claim_text")}
                  />
                  <div className="mt-1.5 flex justify-between text-xs">
                    <span className={claimText.length < 15 ? "text-warning font-medium" : "text-stone-500"}>
                      {claimText.length < 15 ? "Minimum 15 characters required" : "15–500 characters"}
                    </span>
                    <span className="tabular-nums text-stone-500">{claimText.length}/500</span>
                  </div>
                  {form.formState.errors.claim_text ? (
                    <p className="mt-2 text-xs font-medium text-critical flex items-center gap-1" role="alert">
                      <span>⚠</span> {form.formState.errors.claim_text.message}
                    </p>
                  ) : null}
                </div>
              </div>
            ) : null}

            {step === 3 ? (
              <div className="space-y-5">
                <div className="grid gap-4 sm:grid-cols-2">
                  {frontPreview ? (
                    <div className="relative rounded-2xl overflow-hidden border border-sand/70">
                      <span className="absolute top-2 left-2 rounded-md bg-charcoal/80 px-2 py-0.5 text-[11px] font-medium text-white backdrop-blur">
                        Front Photo
                      </span>
                      <img src={frontPreview} alt="Front of item" className="h-48 w-full object-cover" />
                    </div>
                  ) : null}
                  {backPreview ? (
                    <div className="relative rounded-2xl overflow-hidden border border-sand/70">
                      <span className="absolute top-2 left-2 rounded-md bg-charcoal/80 px-2 py-0.5 text-[11px] font-medium text-white backdrop-blur">
                        Back Photo
                      </span>
                      <img src={backPreview} alt="Back of item" className="h-48 w-full object-cover" />
                    </div>
                  ) : null}
                </div>
                <dl className="rounded-2xl border border-sand/60 bg-mist p-4 text-sm divide-y divide-sand/40">
                  <div className="flex justify-between items-center pb-3">
                    <dt className="text-stone-500 font-medium">Selected Reason</dt>
                    <dd className="font-semibold text-charcoal">{claimLabel ?? "—"}</dd>
                  </div>
                  <div className="pt-3">
                    <dt className="text-stone-500 font-medium">Customer Note</dt>
                    <dd className="mt-1.5 whitespace-pre-wrap text-charcoal leading-relaxed">
                      {form.getValues("claim_text")}
                    </dd>
                  </div>
                </dl>
                <label className="flex items-start gap-3 rounded-2xl border border-sand/80 bg-white p-4 hover:border-sage transition-colors cursor-pointer">
                  <input
                    type="checkbox"
                    className="mt-1 h-4 w-4 rounded border-sand text-sage focus:ring-sage"
                    {...form.register("confirm")}
                  />
                  <span className="text-sm text-charcoal font-medium">
                    I confirm these photos show the garment exactly as received without alterations.
                  </span>
                </label>
                {form.formState.errors.confirm ? (
                  <p className="text-xs font-medium text-critical flex items-center gap-1" role="alert">
                    <span>⚠</span> {form.formState.errors.confirm.message}
                  </p>
                ) : null}
                {mutation.isError ? (
                  <p className="rounded-2xl bg-critical/15 border border-critical/30 p-3.5 text-xs text-[#73291D] font-medium" role="alert">
                    {(mutation.error as Error).message}
                  </p>
                ) : null}
              </div>
            ) : null}

            <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-sand/40 pt-5">
              <button
                type="button"
                className={`rounded-full border px-5 py-2.5 text-sm font-semibold transition-all shadow-xs ${
                  step > 1 && !mutation.isPending
                    ? "border-stone-300 bg-white text-charcoal hover:bg-sand/30 hover:border-charcoal/40 cursor-pointer"
                    : "border-transparent bg-transparent text-stone-400 opacity-40 cursor-not-allowed"
                }`}
                onClick={() => setStep((n) => Math.max(1, n - 1))}
                disabled={step === 1 || mutation.isPending}
              >
                ← Back
              </button>
              {step < 3 ? (
                <button
                  type="button"
                  className="rounded-full bg-charcoal px-7 py-2.5 text-sm font-bold text-white hover:bg-black transition-all shadow-sm cursor-pointer"
                  onClick={() => void goNext()}
                >
                  Continue →
                </button>
              ) : (
                <button
                  type="submit"
                  className="rounded-full bg-sage px-7 py-2.5 text-sm font-bold text-[#142617] hover:bg-sage-dark hover:text-white border border-sage-dark/30 shadow-sm disabled:opacity-50 transition-all cursor-pointer"
                  disabled={mutation.isPending}
                >
                  {mutation.isPending ? "Sending photos…" : "Submit return"}
                </button>
              )}
            </div>
          </section>

          <aside className="h-fit rounded-3xl border border-sand/70 bg-white p-5 shadow-card">
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-secondary/20 text-secondary-dark text-xs font-bold">
                i
              </span>
              <h2 className="font-display text-xl text-charcoal">Photo tips</h2>
            </div>
            <ul className="mt-4 space-y-2.5 text-sm text-stone-600">
              {TIPS.map((tip) => (
                <li key={tip} className="flex items-start gap-2.5 leading-snug">
                  <span aria-hidden className="mt-0.5 text-sage font-bold">
                    ✓
                  </span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
            <div className="mt-5 rounded-2xl bg-secondary/10 p-3.5 text-xs text-[#274454] leading-relaxed">
              <strong>Why photos matter:</strong> Clear images speed up automated classification so your claim is processed without delay.
            </div>
          </aside>
        </form>
      )}
    </div>
  );
}

function Confirmation({ returnId, onReset }: { returnId: string; onReset: () => void }) {
  const [copied, setCopied] = useState(false);
  const query = useQuery({
    queryKey: ["return-status", returnId],
    queryFn: () => getReturnStatus(returnId),
    refetchInterval: 5000,
  });
  const status: CustomerStatus = query.data?.status ?? "received";
  const friendlyId = useMemo(() => returnId.slice(0, 8).toUpperCase(), [returnId]);

  function copyId() {
    navigator.clipboard.writeText(returnId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <section className="mt-8 max-w-3xl rounded-3xl bg-white p-7 shadow-soft border border-sand/60">
      <div className="flex items-center gap-2">
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sage/20 text-sage-dark font-bold">
          ✓
        </span>
        <span className="text-xs font-semibold uppercase tracking-wider text-sage-dark">
          Submission Successful
        </span>
      </div>
      <h2 className="mt-3 font-display text-3xl text-charcoal">Return {friendlyId}</h2>
      <p className="mt-2 text-sm text-stone-600 leading-relaxed">
        We have received your garment photos. Keep this reference number handy — our platform is vetting your claim.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <span className="rounded-xl bg-mist border border-sand px-3 py-2 font-mono text-xs text-charcoal">
          {returnId}
        </span>
        <button
          type="button"
          onClick={copyId}
          className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-xs font-semibold text-charcoal hover:bg-sand/30 hover:border-charcoal/40 shadow-xs transition-all cursor-pointer"
        >
          {copied ? "✓ Copied to clipboard" : "Copy Full ID"}
        </button>
      </div>

      <div className="mt-6">
        {query.isError ? (
          <p className="text-xs text-critical font-medium" role="alert">
            {(query.error as Error).message}
          </p>
        ) : (
          <StatusTracker status={status} />
        )}
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-4 border-t border-sand/40 pt-4 text-xs text-stone-500">
        <span>{query.isFetching ? "Checking for live updates…" : "Live updates enabled"}</span>
        <button
          type="button"
          onClick={onReset}
          className="rounded-full border border-stone-300 bg-white px-4 py-2 text-xs font-semibold text-charcoal hover:bg-charcoal hover:text-white shadow-xs transition-all cursor-pointer"
        >
          Start another return →
        </button>
      </div>
    </section>
  );
}
