// src/lib/api.ts
const BASE = import.meta.env.VITE_API_URL ?? "";

export async function apiPost<T>(
    path: string,
    body: unknown,
    signal?: AbortSignal
): Promise<T> {
    const url = `${BASE}${path}`;
    console.log("[apiPost] POST", url, "body:", body);

    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
    });

    let data: unknown;
    try {
        data = await res.json();
    } catch (err) {
        console.error("[apiPost] Failed to parse JSON response", err);
        throw new Error(`${res.status} ${res.statusText}`);
    }

    if (!res.ok) {
        console.error(
            "[apiPost] Error response",
            res.status,
            res.statusText,
            data
        );
        const detail =
            typeof data === "object" && data !== null && "detail" in data
                ? (data as { detail: string }).detail
                : `${res.status} ${res.statusText}`;
        throw new Error(detail);
    }

    console.log("[apiPost] Response from backend:", data);
    return data as T;
}

export async function apiPostForm<T>(
    path: string,
    body: FormData,
    signal?: AbortSignal
): Promise<T> {
    const url = `${BASE}${path}`;
    console.log("[apiPostForm] POST", url, "FormData entries:", [
        ...body.entries(),
    ]);

    const res = await fetch(url, {
        method: "POST",
        body,
        signal,
    });

    let data: unknown;
    try {
        data = await res.json();
    } catch (err) {
        console.error(
            "[apiPostForm] Failed to parse JSON response for",
            url,
            err
        );
        throw new Error(`${res.status} ${res.statusText}`);
    }

    if (!res.ok) {
        console.error(
            "[apiPostForm] Error response",
            res.status,
            res.statusText,
            data
        );
        const detail =
            typeof data === "object" && data !== null && "detail" in data
                ? (data as { detail: string }).detail
                : `${res.status} ${res.statusText}`;
        throw new Error(detail);
    }

    console.log("[apiPostForm] Response from backend:", data);
    return data as T;
}
