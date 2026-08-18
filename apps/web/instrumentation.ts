// Sentry server/edge init (build doc §14 — free tier). No-ops until SENTRY_DSN is set.
import * as Sentry from "@sentry/nextjs";

export async function register() {
  const dsn = process.env.SENTRY_DSN;
  if (!dsn) return;
  if (process.env.NEXT_RUNTIME === "nodejs" || process.env.NEXT_RUNTIME === "edge") {
    Sentry.init({ dsn, tracesSampleRate: 0.1 });
  }
}

export const onRequestError = Sentry.captureRequestError;
