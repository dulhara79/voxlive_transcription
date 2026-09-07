-- ---------------------------------------------------------------------------
-- VoxLive multi-tenant schema (PostgreSQL)
--
-- This is the DESIGN deliverable for step 1. It is not executed by the
-- application yet: the PostgreSQL commit turns this into an Alembic migration
-- and a SqlTenantRepository. It is committed now so the model can be reviewed
-- before any code depends on it.
--
--   organizations
--         |
--         +-- users
--         |
--         +-- sessions
--                 |
--                 +-- speakers
--                 +-- transcript_segments
--
-- THE ONE RULE
-- ------------
-- organization_id is carried on sessions, speakers and transcript_segments
-- even though it is derivable by joining upward. That denormalisation is
-- deliberate: it lets every query filter on organization_id directly, and it
-- is what PostgreSQL row-level security policies attach to. Without it, a
-- forgotten join becomes a cross-tenant read; with it, the tenant filter is
-- available on the table being read.
-- ---------------------------------------------------------------------------

CREATE TABLE organizations (
    id                      TEXT PRIMARY KEY,          -- org_<ms>_<hex>
    name                    TEXT        NOT NULL,
    plan_code               TEXT        NOT NULL DEFAULT 'starter',
    status                  TEXT        NOT NULL DEFAULT 'trial',
    -- NULL means "use the plan's limit". A value here is a negotiated
    -- override for one customer, so plan definitions stay shared.
    max_concurrent_sessions INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT organizations_status_ck
        CHECK (status IN ('active','trial','suspended','closed'))
);

CREATE TABLE users (
    id              TEXT PRIMARY KEY,                  -- usr_<ms>_<hex>
    organization_id TEXT        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email           TEXT        NOT NULL,
    name            TEXT        NOT NULL DEFAULT '',
    role            TEXT        NOT NULL DEFAULT 'member',
    status          TEXT        NOT NULL DEFAULT 'active',
    -- The Cognito subject. Globally unique across the pool, which is why the
    -- lookup that ESTABLISHES tenancy is the only unscoped query in the system.
    cognito_sub     TEXT UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT users_role_ck   CHECK (role   IN ('owner','admin','member','viewer')),
    CONSTRAINT users_status_ck CHECK (status IN ('active','invited','disabled')),
    -- Email is unique WITHIN an organization, not globally: the same person
    -- may legitimately belong to two customers (a contractor, a consultant).
    CONSTRAINT users_email_per_org_uq UNIQUE (organization_id, email)
);
CREATE INDEX users_org_idx ON users (organization_id);

CREATE TABLE sessions (
    id              TEXT PRIMARY KEY,                  -- sess_<ms>_<hex>
    organization_id TEXT        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id         TEXT        NOT NULL REFERENCES users(id),
    status          TEXT        NOT NULL DEFAULT 'active',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    duration_sec    NUMERIC(10,2),
    audio_sec       NUMERIC(10,2),
    segment_count   INTEGER     NOT NULL DEFAULT 0,
    speaker_count   INTEGER     NOT NULL DEFAULT 0,
    languages       TEXT[],
    -- Which ECS task served it. Invaluable when one task misbehaves and you
    -- need to know which sessions were affected.
    task_id         TEXT,
    recording_s3_key TEXT,
    CONSTRAINT sessions_status_ck
        CHECK (status IN ('active','completed','failed','abandoned'))
);
CREATE INDEX sessions_org_started_idx ON sessions (organization_id, started_at DESC);
-- Partial index for the monthly-minutes quota query, which only ever looks at
-- finished sessions.
CREATE INDEX sessions_org_completed_idx ON sessions (organization_id, ended_at)
    WHERE status = 'completed';

CREATE TABLE speakers (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    organization_id TEXT        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    -- The engine's display index (1, 2, 3...), stable within a session only.
    speaker_label   INTEGER     NOT NULL,
    display_name    TEXT,                              -- user-assigned later
    total_seconds   NUMERIC(10,2) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT speakers_session_label_uq UNIQUE (session_id, speaker_label)
);
CREATE INDEX speakers_org_idx ON speakers (organization_id);

CREATE TABLE transcript_segments (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    organization_id TEXT        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    -- TranscriptStore's stable chunk id. ASR completes OUT OF ORDER, so this
    -- is the ordering key; a plain autoincrement would record arrival order,
    -- which is not the order the words were spoken.
    sequence        INTEGER     NOT NULL,
    speaker_label   INTEGER,
    language        TEXT,
    text            TEXT        NOT NULL,
    start_time      NUMERIC(10,3) NOT NULL,
    end_time        NUMERIC(10,3) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT transcript_session_sequence_uq UNIQUE (session_id, sequence)
);
CREATE INDEX transcript_session_time_idx
    ON transcript_segments (session_id, start_time);
CREATE INDEX transcript_org_idx ON transcript_segments (organization_id);

-- ---------------------------------------------------------------------------
-- Optional second line of defence: PostgreSQL row-level security.
--
-- The application already scopes every query by organization_id. RLS makes a
-- forgotten WHERE clause return zero rows instead of another tenant's data —
-- the difference between a bug and a breach. Enable it once the repository
-- layer sets `app.current_organization_id` per connection.
-- ---------------------------------------------------------------------------
--
-- ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY sessions_tenant_isolation ON sessions
--     USING (organization_id = current_setting('app.current_organization_id', true));