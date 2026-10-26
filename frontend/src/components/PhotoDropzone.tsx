import { useId, useRef, useState } from "react";

interface Props {
  label: string;
  file: File | null;
  previewUrl: string | null;
  error?: string;
  onChange: (file: File | null) => void;
}

const MAX_BYTES = 8 * 1024 * 1024;

export default function PhotoDropzone({ label, file, previewUrl, error, onChange }: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [internalError, setInternalError] = useState<string | null>(null);

  function acceptFile(next: File | null) {
    setInternalError(null);
    if (!next) {
      onChange(null);
      return;
    }
    const typeOk = next.type === "image/jpeg" || next.type === "image/png";
    if (!typeOk) {
      onChange(null);
      setInternalError("Please choose a JPEG or PNG photo.");
      return;
    }
    if (next.size > MAX_BYTES) {
      onChange(null);
      setInternalError("Photos must be smaller than 8MB.");
      return;
    }
    onChange(next);
  }

  const displayError = internalError || error;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <label htmlFor={inputId} className="block text-sm font-medium text-charcoal">
          {label}
        </label>
        {file ? (
          <button
            type="button"
            className="rounded-full border border-critical/40 bg-critical/15 px-2.5 py-0.5 text-xs font-semibold text-[#73291D] hover:bg-critical/25 transition-all cursor-pointer"
            onClick={(e) => {
              e.stopPropagation();
              onChange(null);
              setInternalError(null);
            }}
          >
            ✕ Remove
          </button>
        ) : null}
      </div>
      <div
        role="button"
        tabIndex={0}
        className={`group relative min-h-[220px] cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-200 ${
          isDragging
            ? "border-sage bg-sage/10 scale-[1.01]"
            : displayError
              ? "border-critical bg-critical/5"
              : previewUrl
                ? "border-sage/40 bg-mist"
                : "border-sand bg-mist/60 hover:border-sage hover:bg-sage/5"
        }`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          acceptFile(event.dataTransfer.files[0] ?? null);
        }}
      >
        {previewUrl ? (
          <div className="relative">
            <img src={previewUrl} alt={`${label} preview`} className="h-56 w-full object-cover" />
            <div className="absolute inset-0 flex items-center justify-center bg-charcoal/30 opacity-0 backdrop-blur-[2px] transition-opacity group-hover:opacity-100">
              <span className="rounded-full bg-white/95 px-3 py-1.5 text-xs font-semibold text-charcoal shadow-md">
                Click to change photo
              </span>
            </div>
          </div>
        ) : (
          <div className="flex h-56 flex-col items-center justify-center px-4 text-center text-sm text-stone-600">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-sage/20 text-sage-dark">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <span className="mb-1 font-semibold text-charcoal text-sm">
              {isDragging ? "Drop photo here" : "Upload garment photo"}
            </span>
            <span className="text-xs text-stone-500 mb-3">JPEG or PNG, up to 8MB</span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-stone-300 bg-white px-3.5 py-1.5 text-xs font-semibold text-charcoal shadow-xs group-hover:border-sage group-hover:bg-sage/15 transition-all">
              Choose photo
            </span>
          </div>
        )}
        <input
          ref={inputRef}
          id={inputId}
          className="sr-only"
          type="file"
          accept="image/jpeg,image/png"
          capture="environment"
          onChange={(event) => acceptFile(event.target.files?.[0] ?? null)}
        />
      </div>
      {file ? (
        <p className="mt-1.5 text-xs text-stone-600 flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-sage inline-block" />
          {file.name} · {(file.size / 1024).toFixed(0)} KB
        </p>
      ) : null}
      {displayError ? (
        <p className="mt-2 text-xs font-medium text-critical flex items-center gap-1" role="alert">
          <span>⚠</span> {displayError}
        </p>
      ) : null}
    </div>
  );
}
