import { withSentryConfig } from "@sentry/nextjs";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

// Sentry wrapping is inert without a DSN + SENTRY_AUTH_TOKEN (no source-map upload locally).
// Set those in Vercel to activate. Free tier (build doc §14).
export default withSentryConfig(nextConfig, {
  silent: true,
  disableLogger: true,
  // org/project/authToken come from env in CI/Vercel; omitted locally so builds stay offline-safe.
});
