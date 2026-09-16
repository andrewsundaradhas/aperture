# Aperture dashboard (Next.js).
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# The API key is attached server-side at request time (app/api/aperture/[...path]/route.ts), so
# no credential is needed at build time and none can be baked into the bundle.
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup -g 10001 aperture && adduser -u 10001 -G aperture -D aperture

COPY --from=build --chown=aperture:aperture /app/.next ./.next
COPY --from=build --chown=aperture:aperture /app/public ./public
COPY --from=build --chown=aperture:aperture /app/package.json ./package.json
COPY --from=deps --chown=aperture:aperture /app/node_modules ./node_modules

USER aperture
EXPOSE 3000
CMD ["npm", "run", "start"]
