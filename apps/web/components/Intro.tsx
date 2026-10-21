"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";

// WebGL, browser-only. Without WebGL2 the component renders nothing and the plum ground shows.
const MicroSlats = dynamic(() => import("@/components/MicroSlats/MicroSlats"), { ssr: false });

export const INTRO_SEEN_KEY = "aperture-intro-seen";

/**
 * Inline script for <head>: marks the document before first paint when the intro has already
 * been seen this session, so CSS can hide the overlay with no flash.
 */
export const INTRO_HEAD_SCRIPT = `try{if(sessionStorage.getItem("${INTRO_SEEN_KEY}"))document.documentElement.dataset.introSeen="1"}catch(e){}`;

/**
 * The intro screen: an animated slat field with the product name across the middle.
 *
 * Shown once per browser session — people who live in this dashboard should not sit through it
 * on every navigation — and dismissed with the Enter button or the Enter/Space/Escape keys.
 */
export function Intro() {
  const [open, setOpen] = useState(true);
  const [leaving, setLeaving] = useState(false);
  const enterRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let seen = false;
    try {
      seen = !!sessionStorage.getItem(INTRO_SEEN_KEY);
    } catch {
      /* storage blocked: show it, and simply show it again next time */
    }
    if (seen) {
      setOpen(false);
      return;
    }
    enterRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === "Escape" || e.key === " ") {
        e.preventDefault();
        dismiss();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function dismiss() {
    try {
      sessionStorage.setItem(INTRO_SEEN_KEY, "1");
    } catch {
      /* ignore */
    }
    setLeaving(true);
    window.setTimeout(() => setOpen(false), 450);
  }

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Welcome to Aperture"
      className={`intro-overlay fixed inset-0 z-[100] bg-deep-plum transition-opacity duration-500 ${
        leaving ? "pointer-events-none opacity-0" : "opacity-100"
      }`}
    >
      <div className="absolute inset-0">
        <MicroSlats
          preset="swell"
          color="#e894ff"
          glintColor="#ffffff"
          backgroundColor="#23002b"
          slatWidth={10}
          slatHeight={25}
          gap={3}
          roundness={0.75}
          interactive
          cursorStrength={1}
          cursorSize={40}
          swirl={0}
          trail={1.4}
          lean={0}
          intro
        />
      </div>

      {/* A soft plum pool behind the name keeps it readable over the brightest crests. */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 55% 38% at 50% 50%, rgba(35,0,43,0.92) 0%, rgba(35,0,43,0.55) 55%, rgba(35,0,43,0) 100%)",
        }}
        aria-hidden
      />

      <div className="relative flex h-full flex-col items-center justify-center gap-8 px-6 text-center">
        <h1
          className="text-[clamp(88px,19vw,280px)] leading-[0.8] tracking-[0.01em] text-white"
          style={{ textShadow: "0 0 48px rgba(35,0,43,0.9)" }}
        >
          Aperture
        </h1>
        <p className="max-w-md text-body-lg text-slate-smoke">
          Find out why your robot policies fail — and fix it.
        </p>
        <button ref={enterRef} type="button" onClick={dismiss} className="btn-filled px-8">
          Enter
        </button>
      </div>
    </div>
  );
}
