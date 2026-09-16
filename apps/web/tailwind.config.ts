import type { Config } from "tailwindcss";

// Operate — "botanist's data terminal". A near-monochrome green system on a sage-paper
// canvas, where the page itself reads as a chart: hairline borders instead of shadows,
// compact instrument-like type, and data marks sitting directly on the canvas.
//
// Only the tokens below exist. The previous editorial palette (parchment/graphite/signal/…)
// was removed rather than aliased, so a stale utility class fails visibly instead of
// silently resolving to an off-scale colour.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ── Operate core palette ──────────────────────────────────────────
        "forest-ink": "#09352e", // primary text, icons, chart strokes
        "bone-white": "#ffffff", // card surfaces, data-point fills
        "sage-paper": "#e0e0e0", // the page canvas
        "ash-gray": "#e5e5e5", // recessed/nested surface
        "muted-sage": "#77aa83", // soft washes and halos
        lichen: "#cad3d2", // hairlines, gridlines, axes
        "slate-smoke": "#6c7a79", // secondary metadata, captions
        "charcoal-bark": "#29211e", // deep text, rare dark panel
        moss: "#85c093", // THE single filled action colour
        "deep-fern": "#007010", // link / emphasis text accent
        pine: "#117025", // alternate text accent
        emerald: "#008023", // emphasis chips, data-mark fill
        "indigo-accent": "#433787", // one chromatic exception per page

        // ── Failure-surface categories ────────────────────────────────────
        // Validated with the dataviz palette checker against the #e0e0e0 canvas:
        // lightness band, chroma floor, CVD separation (ΔE 13.5) and normal-vision
        // separation (ΔE 15.4) all pass. The brand's own Emerald sits too dark to leave
        // room for three separable steps, so the mid tone is opened up one stop.
        //
        // The lightest step falls under 3:1 against the canvas, which the checker flags
        // as needing relief — every surface mark therefore also carries a 0.5px Forest Ink
        // ring, a legend, and a direct label. Identity is never colour-alone.
        perception: "#7ac98a",
        grounding: "#00a63a",
        motor: "#0b6b45",

        // Reserved status colours — never reused as a data category.
        alarm: "#8a3324", // errors and destructive states
        caution: "#8a6a24", // contested / low-confidence warnings
      },
      fontFamily: {
        // denim — the workhorse: body copy, headings, UI text.
        denim: ["var(--font-denim)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["var(--font-denim)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        // muoto — compact instrument sans for labels, captions, axis ticks (11–13px).
        muoto: ["var(--font-muoto)", "IBM Plex Mono", "ui-monospace", "monospace"],
        // cinetype — ALL-CAPS decorative labels only, at wide tracking.
        cinetype: ["var(--font-cinetype)", "JetBrains Mono", "ui-monospace", "monospace"],
        mono: ["var(--font-muoto)", "IBM Plex Mono", "ui-monospace", "monospace"],
      },
      fontSize: {
        caption: ["11px", { lineHeight: "1.38", letterSpacing: "-0.012em" }],
        body: ["14px", { lineHeight: "1.5", letterSpacing: "0.012em" }],
        // Retained alias: the codebase says `body-sm` in many places and it is the same
        // 14px step as `body` in this system.
        "body-sm": ["14px", { lineHeight: "1.5", letterSpacing: "0.012em" }],
        "body-lg": ["16px", { lineHeight: "1.44", letterSpacing: "0.012em" }],
        subheading: ["18px", { lineHeight: "1.4", letterSpacing: "0" }],
        "heading-sm": ["20px", { lineHeight: "1.35", letterSpacing: "0" }],
        heading: ["32px", { lineHeight: "1.17", letterSpacing: "-0.01em" }],
        display: ["48px", { lineHeight: "1.11", letterSpacing: "-0.018em" }],
      },
      letterSpacing: {
        // The system's visual signature: extreme tracking on uppercase specimen labels.
        specimen: "0.30em",
        instrument: "-0.01em",
      },
      borderRadius: {
        tag: "4px",
        card: "12px",
        btn: "12px",
        panel: "18px",
        hero: "24px",
      },
      boxShadow: {
        // Hairlines, not elevation. 0.5px inset rings are the whole depth model.
        hairline: "rgb(202, 211, 210) 0px 0px 0px 0.5px inset",
        "hairline-ink": "rgb(64, 68, 66) 0px 0px 0px 0.5px inset",
        underline: "rgb(224, 229, 229) 0px -0.5px 0px 0px inset",
        lift: "rgba(9, 53, 45, 0.05) 0px 5px 6px -4px, rgba(174, 202, 197, 0.08) 0px 1px 1px 0.5px",
        panel:
          "rgba(8, 18, 17, 0.01) 0px 0px 16px -1px, rgba(8, 18, 17, 0.04) 0px 0px 5.5px -0.5px, rgb(224, 229, 229) 0px 0px 0px 0.5px inset",
      },
      spacing: {
        section: "64px",
      },
      maxWidth: {
        page: "1280px",
      },
    },
  },
  plugins: [],
};

export default config;
