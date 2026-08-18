import type { Config } from "tailwindcss";

// General Intelligence Company editorial design system — a warm, literary light theme.
// The robotics data-category hues (perception/grounding/motor/ok) are retained but
// retuned for legible contrast on the parchment canvas.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // GIC palette
        parchment: "#fefffc",
        paper: "#ffffff",
        linen: "#f9faf7",
        "ink-black": "#171717",
        graphite: "#2c2c2c",
        charcoal: "#444141",
        ash: "#646464",
        fog: "#b4b8b4",
        mist: "#dee2de",
        twilight: "#282834",
        dusk: "#1f1f29",
        signal: "#41a1cf",
        cerulean: "#0081c0",

        // Backwards-compatible aliases so existing markup adopts the light theme.
        ink: "#171717", // darkest text
        panel: "#ffffff", // paper card surface
        edge: "#dee2de", // mist hairline border
        muted: "#646464", // ash muted text
        accent: "#41a1cf", // signal blue

        // Failure-surface data categories, darkened for light-canvas readability.
        perception: "#a86611",
        grounding: "#6d45c9",
        motor: "#cf3f52",
        ok: "#2e9e63",
      },
      fontFamily: {
        serif: ["var(--font-ppmondwest)", "Fraunces", "Georgia", "serif"],
        sans: [
          "var(--font-af)",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        display: ["54px", { lineHeight: "1.1", letterSpacing: "-0.02em" }],
        "heading-lg": ["48px", { lineHeight: "1.1", letterSpacing: "-0.02em" }],
        heading: ["40px", { lineHeight: "1.1", letterSpacing: "-0.02em" }],
        "heading-sm": ["27px", { lineHeight: "1.25", letterSpacing: "-0.03em" }],
        subheading: ["18px", { lineHeight: "1.3", letterSpacing: "-0.01em" }],
        "body-sm": ["15px", { lineHeight: "1.5", letterSpacing: "-0.01em" }],
        caption: ["13px", { lineHeight: "1.3", letterSpacing: "-0.008em" }],
      },
      borderRadius: {
        nav: "50px",
      },
      boxShadow: {
        nav: "rgba(0, 0, 0, 0.15) 0px 2px 6px 0px",
        subtle:
          "rgba(0, 0, 0, 0.08) 0px 1px 1px 0px, rgba(0, 0, 0, 0.08) 0px 4px 5px 0px",
        diagram: "rgba(0, 0, 0, 0.05) 0px 1px 8px 0px",
        atmospheric:
          "rgba(0, 0, 0, 0.06) 0px 2px 2px 0px, rgba(0, 0, 0, 0.04) 0px 0px 0px 5px",
      },
    },
  },
  plugins: [],
};

export default config;
