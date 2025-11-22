// src/features/predict/usePredict.ts
import { useMutation } from "@tanstack/react-query";
import { apiPost } from "../../lib/api";

export type UnifiedPredictOut = {
    model: string;
    outputs: Array<Record<string, unknown>>;
};

export function usePredict() {
    return useMutation({
        mutationKey: ["predict"],
        mutationFn: (text: string) =>
            apiPost<UnifiedPredictOut>("/api/predict", {
                model: "multihead",
                text,
            }),
        onError: (err) => {
            console.error("predict failed:", err);
        },
    });
}
