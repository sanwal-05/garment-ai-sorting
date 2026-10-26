import type { AdminReturn, CustomerStatus, Decision, Disposition } from "./types";

const TOKEN_KEY = "returns_admin_token";

export function getAdminToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setAdminToken(token: string) {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearAdminToken() {
  sessionStorage.removeItem(TOKEN_KEY);
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item: { msg?: string }) => item.msg ?? JSON.stringify(item)).join(" ");
    }
  } catch {
    /* ignore */
  }
  return res.statusText || "Request failed";
}

export async function submitReturn(form: FormData): Promise<{ id: string; status: "received" }> {
  const res = await fetch("/api/returns", { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function getReturnStatus(id: string): Promise<{ id: string; status: CustomerStatus }> {
  const res = await fetch(`/api/returns/${id}/status`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

function authHeaders(): HeadersInit {
  const token = getAdminToken();
  if (!token) throw new Error("Admin token missing. Sign in again.");
  return { Authorization: `Bearer ${token}` };
}

export async function listReturns(filters: { status?: string; decision?: string }): Promise<AdminReturn[]> {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.decision) params.set("decision", filters.decision);
  const qs = params.toString();
  const res = await fetch(`/api/admin/returns${qs ? `?${qs}` : ""}`, { headers: authHeaders() });
  if (res.status === 401) {
    clearAdminToken();
    throw new Error("Invalid admin token.");
  }
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function getReturn(id: string): Promise<AdminReturn> {
  const res = await fetch(`/api/admin/returns/${id}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function overrideReturn(
  id: string,
  body: { decision: Decision; disposition: Disposition; note: string },
): Promise<AdminReturn> {
  const res = await fetch(`/api/admin/returns/${id}/override`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function probeAdminToken(token: string): Promise<void> {
  const res = await fetch("/api/admin/returns", {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) throw new Error("That token was not accepted.");
  if (!res.ok) throw new Error(await parseError(res));
}
