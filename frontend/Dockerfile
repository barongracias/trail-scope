# syntax=docker/dockerfile:1.7

FROM node:20-slim AS base
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1
WORKDIR /app

FROM base AS deps
COPY package*.json ./
RUN npm install --include=dev

FROM deps AS build
COPY . ./
RUN npm run build

FROM base AS runtime
WORKDIR /app
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/package.json ./package.json
COPY --from=deps /app/node_modules ./node_modules

EXPOSE 3000
CMD ["npm", "run", "start"]
