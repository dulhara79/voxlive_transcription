# Applying these changes

Everything here was written against `develop/AWS_Migration_authentication` at
the commit you linked, and verified in a container: 60 backend tests pass, the
frontend builds, and the assembled FastAPI app boots and serves a real login.

## Option A — apply the patch

```bash
git checkout -b feature/auth-and-language-lock develop/AWS_Migration_authentication
git apply --3way deploy/voxlive-changes.patch
```

## Option B — copy the files

The `backend/` and `frontend/` trees mirror the repository layout, so they can
be copied over the top. `deploy/` is new.

---

## Install and run

```bash
cd backend

# ALWAYS install torch from the CPU index first -- see requirements.txt and
# deploy/AWS_COST.md for why this is worth ~2 GB of container image.
pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2" "torchaudio>=2.2"
pip install -r requirements.txt

cp .env.example .env          # then fill in GEMINI / Vertex settings
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Sign in with any seeded account — password `voxlive-dev-password`:

| email | organization | role |
|---|---|---|
| `alice@acme.example` | ACME Media (Business) | admin |
| `bob@acme.example` | ACME Media | member |
| `carol@acme.example` | ACME Media | viewer — signs in, cannot record |
| `dave@globex.example` | Globex (Starter, 5 concurrent) | owner |
| `erin@initech.example` | Initech — **suspended** | owner |

Or use **Create an organization** to make a fresh tenant with yourself as owner.

### Seeing the organization chooser

It only appears when one email + password matches accounts in more than one
organization, which is rare but must work. To reproduce: sign up twice with the
same email and password under two different organization names, then sign in.

---

## New environment variables

| Variable | Where | Notes |
|---|---|---|
| `AUTH_JWT_SECRET` | backend | **Required outside development.** `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Every task must share it. |
| `AUTH_ACCESS_TTL_SEC` | backend | Default 43200 (12 h). Keep short — the token rides in the WebSocket URL. |
| `AUTH_ORG_SELECT_TTL_SEC` | backend | Default 300. |
| `MAX_CHARS_PER_SEC` | backend | Default 28. The Sinhala/Tamil half of the rate guard. |
| `VITE_API_URL` | frontend | Backend HTTP origin for `/auth/*`. |

`VITE_ORG_ID` / `VITE_USER_ID` are now blank by default. They remain only as a
load-harness escape hatch and are ignored unless `APP_ENV=development`.

---

## Two things that will bite you if you skip them

**CORS.** The frontend now makes real HTTP calls to `/auth/*` before any
WebSocket opens. If `CORS_ORIGINS` does not include the origin serving the
frontend, the browser blocks them before FastAPI ever sees them, and the
symptom is a network error on the sign-in button rather than anything in the
backend log.

**ALB access logs.** The token is in the WebSocket query string, because the
browser WebSocket API cannot set an Authorization header. Confirm your load
balancer is not recording query strings before this reaches a real deployment.

---

## What is deliberately not built

- **Password reset and invitations.** Both need outbound email — a domain, a
  DKIM record, a bounce policy. A half-built version would leave a live
  endpoint minting reset tokens nobody can deliver. Until then, an owner
  creates members and sets the initial password.
- **`Plan.max_monthly_minutes` enforcement.** Needs the `sessions` table to be
  real, because it means summing completed durations per organization per
  month. The index for that query is already in `docs_schema.sql`. Set a GCP
  billing alarm in the meantime — see `AWS_COST.md`.
- **`SqlTenantRepository`.** The interface is extended and
  `InMemoryTenantRepository` now enforces the same per-organization email
  uniqueness PostgreSQL will, so a test passing here should pass against the
  database. Swapping it is still one line in `main.py`.
