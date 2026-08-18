import { withSentryConfig } from "@sentry/nextjs";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

// Only wrap the build with Sentry when it is actually configured (DSN or CI auth token present).
// Locally there is no DSN, so we skip the wrapper entirely — it otherwise adds heavy build-time
// instrumentation that slows/stalls `next dev`. Vercel sets these env vars, activating Sentry
// (source-map upload, release tracking) in production.
const sentryEnabled = Boolean(
  process.env.SENTRY_DSN || process.env.NEXT_PUBLIC_SENTRY_DSN || process.env.SENTRY_AUTH_TOKEN,
);

export default sentryEnabled
  ? withSentryConfig(nextConfig, {
      silent: true,
      // org/project/authToken come from env in CI/Vercel; omitted locally so builds stay offline-safe.
    })
  : nextConfig;
