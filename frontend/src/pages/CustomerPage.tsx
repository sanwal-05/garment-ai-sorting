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
      <p className="text-sm uppercase tracking-[0.2em] text-clay">Customer return</p>
      <h1 className="mt-2 font-display text-4xl sm:text-5xl">Send photos. We’ll check the item fairly.</h1>
      <p className="mt-3 max-w-2xl text-stone-600">
        Three short steps. Your photos are reviewed against the reason you give us — we never show scores or
        thresholds on this page.
      </p>

      {returnId ? (
        <Confirmation returnId={returnId} />
      ) : (
        <form
          className="mt-8 grid gap-6 lg:grid-cols-[1fr_280px]"
          onSubmit={form.handleSubmit((values) => mutation.mutate(values))}
          noValidate
        >
          <section className="rounded-3xl bg-white p-6 shadow-soft">
            <ol className="mb-6 flex gap-2 text-sm" aria-label="Form steps">
              {["Photos", "Reason", "Review"].map((label, index) => {
                const n = index + 1;
                return (
                  <li
                    key={label}
                    className={`flex-1 rounded-full px-3 py-2 text-center ${
                      n === step ? "bg-ink text-mist" : n < step ? "bg-sage/15 text-sage" : "bg-sand"
                    }`}
                    aria-current={n === step ? "step" : undefined}
                  >
                    {n}. {label}
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
                  <label htmlFor="claim" className="mb-2 block text-sm font-medium">
                    Why are you returning this?
                  </label>
                  <select
                    id="claim"
                    className="w-full rounded-xl border border-sand bg-mist px-3 py-3"
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
                    <p className="mt-2 text-sm text-red-700" role="alert">
                      Please choose a reason.
                    </p>
                  ) : null}
                </div>
                <div>
                  <label htmlFor="story" className="mb-2 block text-sm font-medium">
                    Tell us what happened
                  </label>
                  <textarea
                    id="story"
                    rows={6}
                    maxLength={500}
                    className="w-full rounded-xl border border-sand bg-mist px-3 py-3"
                    placeholder="Where is the stain or tear? When did you notice it?"
                    {...form.register("claim_text")}
                  />
                  <div className="mt-1 flex justify-between text-xs text-stone-500">
                    <span>15–500 characters</span>
                    <span>{claimText.length}/500</span>
                  </div>
                  {form.formState.errors.claim_text ? (
                    <p className="mt-2 text-sm text-red-700" role="alert">
                      {form.formState.errors.claim_text.message}
                    </p>
                  ) : null}
                </div>
              </div>
            ) : null}

            {step === 3 ? (
              <div className="space-y-5">
                <div className="grid gap-4 sm:grid-cols-2">
                  {frontPreview ? (
                    <img src={frontPreview} alt="Front of item" className="h-48 w-full rounded-2xl object-cover" />
                  ) : null}
                  {backPreview ? (
                    <img src={backPreview} alt="Back of item" className="h-48 w-full rounded-2xl object-cover" />
                  ) : null}
                </div>
                <dl className="rounded-2xl bg-mist p-4 text-sm">
                  <div className="flex justify-between gap-4">
                    <dt className="text-stone-500">Reason</dt>
                    <dd>{claimLabel ?? "—"}</dd>
                  </div>
                  <div className="mt-3">
                    <dt className="text-stone-500">Your note</dt>
                    <dd className="mt-1 whitespace-pre-wrap">{form.getValues("claim_text")}</dd>
                  </div>
                </dl>
                <label className="flex items-start gap-3 rounded-2xl border border-sand p-4">
                  <input type="checkbox" className="mt-1" {...form.register("confirm")} />
                  <span>I confirm these photos show the item exactly as received.</span>
                </label>
                {form.formState.errors.confirm ? (
                  <p className="text-sm text-red-700" role="alert">
                    {form.formState.errors.confirm.message}
                  </p>
                ) : null}
                {mutation.isError ? (
                  <p className="rounded-xl bg-red-50 px-3 py-2 text-sm text-red-800" role="alert">
                    {(mutation.error as Error).message}
                  </p>
                ) : null}
              </div>
            ) : null}

            <div className="mt-8 flex flex-wrap justify-between gap-3">
              <button
                type="button"
                className="rounded-full px-4 py-2 text-sm disabled:opacity-40"
                onClick={() => setStep((n) => Math.max(1, n - 1))}
                disabled={step === 1 || mutation.isPending}
              >
                Back
              </button>
              {step < 3 ? (
                <button
                  type="button"
                  className="rounded-full bg-ink px-5 py-2 text-sm text-mist"
                  onClick={() => void goNext()}
                >
                  Continue
                </button>
              ) : (
                <button
                  type="submit"
                  className="rounded-full bg-clay px-5 py-2 text-sm text-white disabled:opacity-60"
                  disabled={mutation.isPending}
                >
                  {mutation.isPending ? "Sending photos…" : "Submit return"}
                </button>
              )}
            </div>
          </section>

          <aside className="h-fit rounded-3xl border border-sand bg-white p-5">
            <h2 className="font-display text-2xl">Photo tips</h2>
            <ul className="mt-3 space-y-2 text-sm text-stone-600">
              {TIPS.map((tip) => (
                <li key={tip} className="flex gap-2">
                  <span aria-hidden className="text-sage">
                    ✓
                  </span>
                  {tip}
                </li>
              ))}
            </ul>
          </aside>
        </form>
      )}
    </div>
  );
}

function Confirmation({ returnId }: { returnId: string }) {
  const query = useQuery({
    queryKey: ["return-status", returnId],
    queryFn: () => getReturnStatus(returnId),
    refetchInterval: 5000,
  });
  const status: CustomerStatus = query.data?.status ?? "received";
  const friendlyId = useMemo(() => returnId.slice(0, 8).toUpperCase(), [returnId]);

  return (
    <section className="mt-8 max-w-3xl rounded-3xl bg-white p-6 shadow-soft">
      <p className="text-sage">It’s in. Thank you.</p>
      <h2 className="mt-2 font-display text-3xl">Return {friendlyId}</h2>
      <p className="mt-2 text-sm text-stone-600">
        Keep this reference if you need to follow up. We update the tracker automatically.
      </p>
      <p className="mt-4 rounded-xl bg-mist px-3 py-2 font-mono text-sm">{returnId}</p>
      <div className="mt-6">
        {query.isError ? (
          <p className="text-sm text-red-700" role="alert">
            {(query.error as Error).message}
          </p>
        ) : (
          <StatusTracker status={status} />
        )}
      </div>
      {query.isFetching ? <p className="mt-3 text-xs text-stone-500">Checking for updates…</p> : null}
    </section>
  );
}
