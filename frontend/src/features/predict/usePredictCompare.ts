// src/features/predict/usePredictCompare.ts
import { useMutation } from "@tanstack/react-query";
import { apiPost } from "../../lib/api";

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
