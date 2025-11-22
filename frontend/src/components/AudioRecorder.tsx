import { useEffect, useRef, useState } from "react";

type AudioRecorderProps = {
    onRecordingComplete?: (blob: Blob) => void;
};

export function AudioRecorder({ onRecordingComplete }: AudioRecorderProps) {
    const [isRecording, setIsRecording] = useState(false);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    async function startRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setError("Audio recording is not supported in this browser.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
            });
            const recorder = new MediaRecorder(stream);

            chunksRef.current = [];

            recorder.ondataavailable = (event: BlobEvent) => {
                if (event.data && event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            recorder.onstop = () => {
                const blob = new Blob(chunksRef.current, {
                    type: "audio/webm",
                });

                // Revoke previous URL if any
                setAudioUrl((prev) => {
                    if (prev) URL.revokeObjectURL(prev);
                    return URL.createObjectURL(blob);
                });

                if (onRecordingComplete) {
                    onRecordingComplete(blob);
                }
            };

            mediaRecorderRef.current = recorder;
            recorder.start();
            setIsRecording(true);
            setError(null);
        } catch (err) {
            console.error("Failed to start recording:", err);
            setError("Microphone access was denied or is unavailable.");
        }
    }

    function stopRecording() {
        const recorder = mediaRecorderRef.current;
        if (!recorder || !isRecording) return;

        try {
            recorder.stop();
        } catch (err) {
            console.error("Failed to stop recording:", err);
        } finally {
            // Stop all tracks on the underlying stream
            recorder.stream.getTracks().forEach((track) => track.stop());
            setIsRecording(false);
        }
    }

    useEffect(() => {
        return () => {
            // Cleanup on unmount
            const recorder = mediaRecorderRef.current;
            if (recorder && recorder.state !== "inactive") {
                try {
                    recorder.stop();
                } catch {
                    // ignore
                }
                recorder.stream.getTracks().forEach((track) => track.stop());
            }

            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [audioUrl]);

    return (
        <div
            style={{
                width: "min(90vw, 560px)",
                alignSelf: "center",
                display: "flex",
                flexDirection: "column",
                alignItems: "stretch",
                gap: 12,
            }}
        >
            <div
                style={{
                    fontWeight: 600,
                    fontSize: 18,
                    textAlign: "center",
                    marginBottom: 4,
                }}
            >
                Record audio for analysis
            </div>

            <button
                type="button"
                onClick={isRecording ? stopRecording : startRecording}
                style={{
                    padding: "12px 16px",
                    borderRadius: 999,
                    border: "1px solid var(--border)",
                    background: isRecording
                        ? "var(--danger-bg, #ff4d4f)"
                        : "var(--button-bg)",
                    color: "var(--button-text)",
                    fontWeight: 600,
                    alignSelf: "center",
                    minWidth: 180,
                }}
            >
                {isRecording ? "Stop recording" : "Start recording"}
            </button>

            {isRecording && (
                <div
                    style={{
                        textAlign: "center",
                        color: "var(--text)",
                        opacity: 0.8,
                        fontSize: 14,
                    }}
                >
                    Listening… tap &quot;Stop recording&quot; when you&apos;re
                    done.
                </div>
            )}

            {audioUrl && (
                <div
                    style={{
                        marginTop: 8,
                        padding: 12,
                        borderRadius: 12,
                        border: "1px solid var(--border)",
                        background: "var(--card-bg)",
                        display: "flex",
                        flexDirection: "column",
                        gap: 8,
                    }}
                >
                    <div
                        style={{
                            fontWeight: 500,
                            fontSize: 14,
                            marginBottom: 4,
                        }}
                    >
                        Latest recording
                    </div>
                    <audio controls src={audioUrl} />
                </div>
            )}

            {error && (
                <div
                    style={{
                        marginTop: 8,
                        color: "#ff7a7a",
                        fontSize: 13,
                        textAlign: "center",
                    }}
                >
                    {error}
                </div>
            )}
        </div>
    );
}
