// src/lib/api.ts
const BASE = import.meta.env.VITE_API_URL ?? "";

export async function apiPost<T>(
    path: string,
    body: unknown,
    signal?: AbortSignal
): Promise<T> {
    const res = await fetch(`${BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json() as Promise<T>;
}
