import { useId, useRef } from "react";

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

  function acceptFile(next: File | null) {
    if (!next) {
      onChange(null);
      return;
    }
    const typeOk = next.type === "image/jpeg" || next.type === "image/png";
    if (!typeOk) {
      onChange(null);
      window.alert("Please choose a JPEG or PNG photo.");
      return;
    }
    if (next.size > MAX_BYTES) {
      onChange(null);
      window.alert("Photos must be smaller than 8MB.");
      return;
    }
    onChange(next);
  }

  return (
    <div>
      <label htmlFor={inputId} className="mb-2 block text-sm font-medium">
        {label}
      </label>
      <div
        role="button"
        tabIndex={0}
        className={`relative min-h-[220px] cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed bg-mist transition hover:border-clay ${
          error ? "border-red-500" : "border-sand"
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
        }}
        onDrop={(event) => {
          event.preventDefault();
          acceptFile(event.dataTransfer.files[0] ?? null);
        }}
      >
        {previewUrl ? (
          <img src={previewUrl} alt={`${label} preview`} className="h-56 w-full object-cover" />
        ) : (
          <div className="flex h-56 flex-col items-center justify-center px-4 text-center text-sm text-stone-600">
            <span className="mb-2 font-medium text-ink">Drop a photo or tap to upload</span>
            <span>JPEG or PNG, up to 8MB. On a phone, this opens the camera.</span>
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
        <p className="mt-2 text-xs text-stone-600">
          {file.name} · {(file.size / 1024).toFixed(0)} KB
        </p>
      ) : null}
      {error ? (
        <p className="mt-2 text-sm text-red-700" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
