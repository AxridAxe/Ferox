# FeroxAi

A secure gateway that aggregates free-tier AI models from providers like
OpenRouter and serves them behind one consistent assistant persona — so the
model backing a response can change day to day without the experience
changing.

## Why

Free-tier model availability shifts constantly and every provider/model has
a different voice. FeroxAi sits in front of that churn: it discovers which
free models are currently available, routes each request to one, and
injects a shared system prompt so responses stay consistent regardless of
which underlying model answered.

## Architecture

```
client → FeroxAi gateway → provider registry → free model → response
              │
              ├─ auth (API key)
              ├─ rate limiting
              └─ shared persona (system prompt)
```

- `src/providers/` — one module per upstream provider. Each implements the
  `Provider` interface (`listFreeModels`, `chat`). Only OpenRouter is wired
  up today; adding a provider means adding one file and registering it.
- `src/router/` — picks an available free model and forwards the chat
  request.
- `src/persona/` — the shared system prompt injected into every request so
  output tone stays consistent across models.
- `src/middleware/` — API key auth, rate limiting, error handling.

## Setup

```bash
npm install
cp .env.example .env   # fill in your OpenRouter API key
npm run dev
```

## Security

- All upstream API keys live in environment variables only — never
  committed, never logged. `.env` is gitignored; `.env.example` documents
  the shape.
- Every request to this gateway requires an `x-api-key` header, checked
  with a timing-safe comparison against `FEROX_API_KEYS`.
- `helmet` sets standard security headers; CORS is locked to
  `ALLOWED_ORIGINS`.
- Per-IP and per-key rate limiting via `express-rate-limit`.
- Request bodies are validated (`zod`) before touching any provider —
  no unbounded prompt sizes, no unexpected fields.
- Errors returned to clients are generic; stack traces and provider error
  detail are logged server-side only.

## Status

Early scaffold. Chat routing and the OpenRouter free-model provider work;
additional providers, response normalization, and persistent model-health
tracking are next.
