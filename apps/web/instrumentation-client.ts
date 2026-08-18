// Sentry client init (build doc §14 — free tier). No-ops until NEXT_PUBLIC_SENTRY_DSN is set,
// so local/dev runs need no Sentry account. Set the DSN in Vercel to activate error capture.
import * as Sentry from "@sentry/nextjs";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
if (dsn) {
  Sentry.init({
    dsn,
    tracesSampleRate: 0.1,
    enabled: true,
  });
}

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
