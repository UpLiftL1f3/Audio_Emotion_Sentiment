// import { useState } from "react";
// import "./App.css";

// function App() {
//     // const [count, setCount] = useState(0)
//     const [statement, setStatement] = useState("");

//     const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
//         console.log(event.target.value);
//         setStatement(event.target.value);
//     };

//     const handleSubmit = () => {
//         console.log("Submitted:", statement);
//     };

//     return (
//         <>
//             <div className="container">
//                 <input
//                     className="input"
//                     type="text"
//                     value={statement}
//                     onChange={handleChange}
//                 />
//                 {statement.trim().length > 0 && (
//                     <button className="submit-button" onClick={handleSubmit}>
//                         Submit
//                     </button>
//                 )}
//             </div>
//         </>
//     );
// }

// export default App;

// src/App.tsx
import { useLayoutEffect, useRef, useState } from "react";
import { usePredictMulti } from "./features/predict/usePredictCompare";
import { getInitialTheme, toggleTheme, type Theme } from "./theme";

export default function App() {
    const [text, setText] = useState("");
    const [theme, setThemeState] = useState<Theme>(getInitialTheme());
    const predictMulti = usePredictMulti();
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);
    const MAX_ROWS = 8;

    function adjustTextareaHeight(el: HTMLTextAreaElement, maxRows: number) {
        // Reset height to measure true scroll height
        el.style.height = "auto";
        const computed = window.getComputedStyle(el);
        let lineHeight = parseFloat(computed.lineHeight);
        if (Number.isNaN(lineHeight)) {
            const fontSize = parseFloat(computed.fontSize) || 16;
            lineHeight = fontSize * 1.4;
        }
        const paddingTop = parseFloat(computed.paddingTop) || 0;
        const paddingBottom = parseFloat(computed.paddingBottom) || 0;
        const borderTop = parseFloat(computed.borderTopWidth) || 0;
        const borderBottom = parseFloat(computed.borderBottomWidth) || 0;
        const maxHeight =
            lineHeight * maxRows +
            paddingTop +
            paddingBottom +
            borderTop +
            borderBottom;
        const newHeight = Math.min(el.scrollHeight, Math.ceil(maxHeight));
        el.style.height = `${newHeight}px`;
        el.style.overflowY = el.scrollHeight > newHeight ? "auto" : "hidden";
    }

    function formatPercent(value: number): string {
        const pct = Math.round(value * 1000) / 10; // one decimal
        return `${pct}%`;
    }

    function sortedEntries(
        probs: Record<string, number> | undefined
    ): Array<[string, number]> {
        if (!probs) return [];
        return Object.entries(probs).sort((a, b) => b[1] - a[1]);
    }

    function capFirst(s: string | undefined | null): string {
        if (!s) return "";
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    function titleFor(key: string): string {
        switch (key) {
            case "multihead":
                return "DistilBERT";
            case "twoModelHead":
                return "RoBERTa";
            case "lr":
                return "Logistic Regression";
            case "svm":
                return "Support Vector Machine";
            default:
                return key;
        }
    }

    useLayoutEffect(() => {
        if (textareaRef.current) {
            adjustTextareaHeight(textareaRef.current, MAX_ROWS);
        }
    }, [text]);

    function onSubmit(e: React.FormEvent) {
        e.preventDefault();
        const cleaned = text.trim();
        if (!cleaned) return;
        predictMulti.mutate({
            text: cleaned,
            models: ["multihead", "twoModelHead", "lr", "svm"],
        });
    }

    return (
        <div
            style={{
                width: "100vw",
                height: "100vh",
                // minHeight: "100dvh",
                display: "grid",
                placeItems: "center",
                background: "var(--bg)",
                color: "var(--text)",
            }}
        >
            <div
                style={{
                    position: "fixed",
                    top: 12,
                    right: 12,
                    display: "flex",
                    gap: 8,
                }}
            >
                <button
                    type="button"
                    onClick={() => setThemeState(toggleTheme(theme))}
                    aria-label="Toggle theme"
                    title="Toggle theme"
                    style={{
                        padding: "8px 12px",
                        borderRadius: 10,
                    }}
                >
                    {theme === "dark" ? "Light mode" : "Dark mode"}
                </button>
            </div>
            <form
                onSubmit={onSubmit}
                style={{
                    display: "flex",
                    width: "min(90vw, 1440px)",
                    justifyContent: "center",
                    flexDirection: "column",
                }}
            >
                <div
                    style={{
                        width: "min(90vw, 560px)",
                        alignSelf: "center",
                        display: "flex",
                        flexDirection: "column",
                    }}
                >
                    <textarea
                        ref={textareaRef}
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Type text for analysis…"
                        style={{
                            width: "100%",
                            padding: "14px 16px",
                            borderRadius: 12,
                            border: "1px solid var(--border)",
                            background: "var(--input-bg)",
                            color: "var(--text)",
                            resize: "none",
                        }}
                        rows={1}
                    />
                    <button
                        type="submit"
                        disabled={predictMulti.isPending || !text.trim()}
                        style={{
                            marginTop: 12,
                            padding: "10px 16px",
                            borderRadius: 10,
                            alignSelf: "center",
                            background: "var(--button-bg)",
                            color: "var(--button-text)",
                            marginBottom: 42,
                        }}
                    >
                        {predictMulti.isPending ? "Analyzing…" : "Analyze"}
                    </button>
                </div>

                {predictMulti.isError && (
                    <p style={{ marginTop: 12, color: "#ff7a7a" }}>
                        {(predictMulti.error as Error).message}
                    </p>
                )}

                {predictMulti.isSuccess && predictMulti.data && (
                    <div
                        style={{
                            marginTop: 12,
                            width: "90%",
                            alignSelf: "center",
                            border: "1px solid var(--border)",
                            borderRadius: 12,
                            overflow: "hidden",
                            // background: "red",
                            background: "var(--card-bg)",
                        }}
                    >
                        {/* 2x2 grid with a cross split */}
                        <div
                            style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 1fr",
                                gridTemplateRows: "1fr 1fr",
                            }}
                        >
                            {["multihead", "twoModelHead", "lr", "svm"].map(
                                (name, idx) => {
                                    const res =
                                        predictMulti.data.results?.[name]?.[0];
                                    const isLeftCol = idx % 2 === 0;
                                    const isTopRow = idx < 2;
                                    const bgVars = [
                                        "var(--quad-1-bg)",
                                        "var(--quad-2-bg)",
                                        "var(--quad-3-bg)",
                                        "var(--quad-4-bg)",
                                    ];
                                    const cellBg = bgVars[idx % bgVars.length];
                                    return (
                                        <div
                                            key={name}
                                            style={{
                                                padding: 11,
                                                minHeight: 198,
                                                background: cellBg,
                                                borderRight: isLeftCol
                                                    ? "1px solid var(--border)"
                                                    : undefined,
                                                borderBottom: isTopRow
                                                    ? "1px solid var(--border)"
                                                    : undefined,
                                                display: "flex",
                                                flexDirection: "column",
                                                gap: 7,
                                            }}
                                        >
                                            <div
                                                style={{
                                                    fontWeight: 700,
                                                    fontSize: 16,
                                                    letterSpacing: 0.3,
                                                    textAlign: "center",
                                                    borderBottom:
                                                        "1px solid var(--border)",
                                                    paddingBottom: 6,
                                                    marginBottom: 6,
                                                }}
                                            >
                                                {titleFor(name)}
                                            </div>
                                            {!res ? (
                                                <div
                                                    style={{
                                                        opacity: 0.7,
                                                        fontStyle: "italic",
                                                    }}
                                                >
                                                    No data
                                                </div>
                                            ) : (
                                                <>
                                                    <div
                                                        style={{
                                                            display: "flex",
                                                            gap: 8,
                                                            flexWrap: "wrap",
                                                            alignItems:
                                                                "baseline",
                                                        }}
                                                    >
                                                        <span
                                                            style={{
                                                                fontWeight: 600,
                                                            }}
                                                        >
                                                            Sentiment:
                                                        </span>
                                                        <span>
                                                            {capFirst(
                                                                res.sentiment
                                                            )}
                                                        </span>
                                                    </div>
                                                    <div
                                                        style={{
                                                            display: "flex",
                                                            flexWrap: "wrap",
                                                            // gap: 12,
                                                            justifyContent:
                                                                "flex-start",
                                                            fontSize: 13,
                                                            marginTop: 6,
                                                            marginBottom: 10,
                                                        }}
                                                    >
                                                        {sortedEntries(
                                                            res.sentiment_probs
                                                        ).map(([label, p]) => (
                                                            <div
                                                                key={`s-${label}`}
                                                                style={{
                                                                    display:
                                                                        "flex",
                                                                    flexDirection:
                                                                        "column",
                                                                    alignItems:
                                                                        "center",
                                                                    flex: "0 0 120px",
                                                                    minWidth: 0,
                                                                    border:
                                                                        p >= 0.3
                                                                            ? "2px solid var(--text)"
                                                                            : "1px solid transparent",
                                                                    borderRadius: 6,
                                                                    padding: 4,
                                                                }}
                                                            >
                                                                <span
                                                                    style={{
                                                                        fontWeight: 600,
                                                                    }}
                                                                >
                                                                    {capFirst(
                                                                        label
                                                                    )}
                                                                </span>
                                                                <span
                                                                    style={{
                                                                        opacity: 0.85,
                                                                    }}
                                                                >
                                                                    {formatPercent(
                                                                        p
                                                                    )}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>

                                                    <div
                                                        style={{
                                                            display: "flex",
                                                            gap: 8,
                                                            flexWrap: "wrap",
                                                            alignItems:
                                                                "baseline",
                                                            marginTop: 14,
                                                        }}
                                                    >
                                                        <span
                                                            style={{
                                                                fontWeight: 600,
                                                            }}
                                                        >
                                                            Emotion:
                                                        </span>
                                                        <span>
                                                            {capFirst(
                                                                res.emotion
                                                            )}
                                                        </span>
                                                    </div>
                                                    <div
                                                        style={{
                                                            display: "grid",
                                                            gridTemplateColumns:
                                                                "repeat(6, minmax(0, 8em))",
                                                            gap: 0,
                                                            fontSize: 13,
                                                            marginTop: 6,
                                                        }}
                                                    >
                                                        {sortedEntries(
                                                            res.emotion_probs
                                                        ).map(([label, p]) => (
                                                            <div
                                                                key={`e-${label}`}
                                                                style={{
                                                                    display:
                                                                        "flex",
                                                                    flexDirection:
                                                                        "column",
                                                                    alignItems:
                                                                        "center",
                                                                    flex: "0 0 120px",
                                                                    minWidth: 40,
                                                                    border:
                                                                        p >= 0.3
                                                                            ? "2px solid var(--text)"
                                                                            : "1px solid transparent",
                                                                    borderRadius: 6,
                                                                    padding: 4,
                                                                }}
                                                            >
                                                                <span
                                                                    style={{
                                                                        fontWeight: 600,
                                                                    }}
                                                                >
                                                                    {capFirst(
                                                                        label
                                                                    )}
                                                                </span>
                                                                <span
                                                                    style={{
                                                                        opacity: 0.85,
                                                                    }}
                                                                >
                                                                    {formatPercent(
                                                                        p
                                                                    )}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    );
                                }
                            )}
                        </div>
                    </div>
                )}
            </form>
        </div>
    );
}
