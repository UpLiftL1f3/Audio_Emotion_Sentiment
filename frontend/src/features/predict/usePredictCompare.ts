// src/features/predict/usePredictCompare.ts
import { useMutation } from "@tanstack/react-query";
import { apiPost, apiPostForm } from "../../lib/api";

// Generic model output for audio models.
// All backends return per-label probabilities plus a top label/score,
// and some (wav2vec*/hubert*) also return a human-readable message.
export type ModelOutput = {
    probs: Record<string, number>;
    top_label: string;
    top_score: number;
    message?: string;
    // Allow extra keys from specific models without breaking the type.
    [key: string]: unknown;
};

export type PredictMultiOut = {
    results: Record<string, ModelOutput[]>;
};

export function usePredictMulti() {
    return useMutation({
        mutationKey: ["predict-multi"],
        mutationFn: (payload: { text: string; models: string[] }) =>
            apiPost<PredictMultiOut>("/api/predict_multi", payload),
        // optional: centralize error toast, analytics, etc.
        onError: (err) => {
            console.error("predict_multi failed:", err);
        },
    });
}

export function usePredictMultiAudio() {
    return useMutation({
        mutationKey: ["predict-multi-audio"],
        mutationFn: async (payload: { blob: Blob; models: string[] }) => {
            const form = new FormData();
            form.append("file", payload.blob, "audio.webm");

            const query = new URLSearchParams();
            payload.models.forEach((m) => query.append("models", m));
            const path = `/api/predict_multi_audio?${query.toString()}`;

            return apiPostForm<PredictMultiOut>(path, form);
        },
        onError: (err) => {
            console.error("predict_multi_audio failed:", err);
        },
    });
}
