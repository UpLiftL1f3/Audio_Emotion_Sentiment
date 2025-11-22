export type Theme = "light" | "dark";

function getSystemTheme(): Theme {
    if (typeof window === "undefined") return "light";
    return window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
}

function applyTheme(theme: Theme) {
    if (typeof document === "undefined") return;
    document.documentElement.setAttribute("data-theme", theme);
}

export function getInitialTheme(): Theme {
    try {
        const saved = localStorage.getItem("theme") as Theme | null;
        return saved ?? getSystemTheme();
    } catch {
        return getSystemTheme();
    }
}

export function setTheme(theme: Theme) {
    try {
        localStorage.setItem("theme", theme);
    } catch {
        /* ignore write errors */
    }
    applyTheme(theme);
}

export function initTheme() {
    applyTheme(getInitialTheme());
}

export function toggleTheme(current: Theme): Theme {
    const next: Theme = current === "light" ? "dark" : "light";
    setTheme(next);
    return next;
}
