// src/features/predict/usePredictCompare.ts
import { useMutation } from "@tanstack/react-query";
import { apiPost, apiPostForm } from "../../lib/api";

export type ModelOutput = {
    sentiment: string;
    sentiment_probs: Record<string, number>;
    emotion: string;
    emotion_probs: Record<string, number>;
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
