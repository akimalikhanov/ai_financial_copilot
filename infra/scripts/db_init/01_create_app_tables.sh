#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# 001_init_core.sh
#
# This script is meant to be mounted into a Postgres container at:
#   /docker-entrypoint-initdb.d
# e.g. in docker-compose:
#   - ../db_init:/docker-entrypoint-initdb.d
#
# The official Postgres image will run *.sh (executable) and *.sql files in that
# directory on first initialization of the database volume.
# ------------------------------------------------------------------------------

set -euo pipefail

echo ">> Initializing core DB schema (users, conversations, llm_requests, messages)..."

# Use APP_DB environment variable (defaults to 'app' if not set)
APP_DB="${APP_DB:-app}"

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${APP_DB}" <<'SQL'
-- ============================================================================
-- Extensions
-- ============================================================================

-- pgcrypto provides gen_random_uuid() for UUID primary keys
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- citext provides case-insensitive text (great for unique emails)
CREATE EXTENSION IF NOT EXISTS citext;

-- ============================================================================
-- Enums / Types
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_role') THEN
    CREATE TYPE message_role AS ENUM (
      'system',    -- system message / instruction
      'user',      -- end-user message
      'assistant', -- assistant response
      'tool'       -- tool call or tool output (optional, if you store those)
    );
  END IF;
END $$;

COMMENT ON TYPE message_role IS
  'Role of a message in a conversation: system/user/assistant/tool.';

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_status') THEN
    CREATE TYPE message_status AS ENUM (
      'completed',    -- final persisted message
      'in_progress',  -- optional: partial/streaming state persisted
      'cancelled',    -- generation was cancelled
      'error'         -- generation failed
    );
  END IF;
END $$;

COMMENT ON TYPE message_status IS
  'Lifecycle state of a message: completed/in_progress/cancelled/error.';

-- ============================================================================
-- Utility trigger: auto-update updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION set_updated_at() IS
  'Trigger function to automatically set updated_at=now() on UPDATE.';

-- ============================================================================
-- Table: users
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  -- Primary identifier for the user. UUID keeps IDs opaque and easy to shard.

  email             citext NOT NULL UNIQUE,
  -- User email (case-insensitive). Unique constraint prevents duplicates.

  display_name      text,
  -- Optional UI name / nickname.

  auth_provider     text NOT NULL DEFAULT 'local',
  -- Authentication provider label: local/google/github/etc.

  auth_subject      text,
  -- Provider subject / user id (e.g., OAuth "sub"). Lets you map external accounts.

  email_verified_at timestamptz,
  -- Timestamp when email was verified (NULL if not verified).

  is_active         boolean NOT NULL DEFAULT true,
  -- Soft account disable flag.

  created_at        timestamptz NOT NULL DEFAULT now(),
  -- Creation timestamp.

  updated_at        timestamptz NOT NULL DEFAULT now(),
  -- Last update timestamp (maintained by trigger).

  last_seen_at      timestamptz,
  -- Optional: last activity timestamp for analytics / session management.

  metadata          jsonb NOT NULL DEFAULT '{}'::jsonb
  -- Flexible JSON for user prefs, plan flags, etc. (avoid schema churn).
);

COMMENT ON TABLE users IS
  'Application users. Stores identity/auth info and stable UUID primary key.';

COMMENT ON COLUMN users.id IS 'Primary key UUID (gen_random_uuid()).';
COMMENT ON COLUMN users.email IS 'Case-insensitive unique email (citext).';
COMMENT ON COLUMN users.display_name IS 'Optional display name shown in UI.';
COMMENT ON COLUMN users.auth_provider IS 'Auth provider key (local/google/etc.).';
COMMENT ON COLUMN users.auth_subject IS 'External provider subject/user id.';
COMMENT ON COLUMN users.email_verified_at IS 'When email was verified (NULL otherwise).';
COMMENT ON COLUMN users.is_active IS 'Whether account is active (soft disable).';
COMMENT ON COLUMN users.created_at IS 'Row creation time.';
COMMENT ON COLUMN users.updated_at IS 'Row last updated time (trigger managed).';
COMMENT ON COLUMN users.last_seen_at IS 'Most recent activity time.';
COMMENT ON COLUMN users.metadata IS 'Arbitrary JSON metadata for future fields.';

CREATE INDEX IF NOT EXISTS users_auth_lookup_idx
  ON users (auth_provider, auth_subject);
COMMENT ON INDEX users_auth_lookup_idx IS
  'Lookup index for (auth_provider, auth_subject) during OAuth sign-in.';

CREATE INDEX IF NOT EXISTS users_updated_at_idx
  ON users (updated_at DESC);
COMMENT ON INDEX users_updated_at_idx IS
  'Supports sorting users by recent updates (admin/ops use).';

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================================================
-- Table: conversations
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversations (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  -- Primary key UUID for the conversation.

  user_id            uuid,
  -- Owner of this conversation. Nullable for skip-auth approach.
  -- ForeignKey("users.id", ondelete="CASCADE") - commented out until users table exists

  title              text,
  -- Optional conversation title (UI sidebar). Can be generated.

  created_at         timestamptz NOT NULL DEFAULT now(),
  -- Conversation creation time.

  updated_at         timestamptz NOT NULL DEFAULT now(),
  -- Conversation update time (trigger managed).

  last_message_at    timestamptz,
  -- Denormalized last message time for fast sidebar ordering.

  last_message_id    uuid,
  -- Denormalized last message id for fast preview fetch (FK added after messages table).

  message_count      integer,
  -- Denormalized count (nullable, can be computed via COUNT(*) or async worker).

  last_seq           bigint,
  -- Last sequence number (guarded by WHERE new_seq > last_seq to prevent race conditions).

  pinned             boolean NOT NULL DEFAULT false,
  -- UI pin (keep at top).

  archived_at        timestamptz,
  -- Archive timestamp (NULL means not archived).

  deleted_at         timestamptz,
  -- Soft-delete timestamp (NULL means visible/active).

  summary            text,
  -- Optional rolling summary to keep context short for LLM prompts.

  summary_updated_at timestamptz,
  -- When summary was last refreshed.

  settings           jsonb NOT NULL DEFAULT '{}'::jsonb,
  -- Per-conversation settings (system prompt override, retrieval prefs, etc).

  metadata           jsonb NOT NULL DEFAULT '{}'::jsonb
  -- Extra flexible fields (tags, UI state, etc).
);

COMMENT ON TABLE conversations IS
  'User conversations. Includes denormalized last_message_* fields for fast listing.';

COMMENT ON COLUMN conversations.id IS 'Primary key UUID.';
COMMENT ON COLUMN conversations.user_id IS 'Owner user id. FK to users.';
COMMENT ON COLUMN conversations.title IS 'Conversation title (optional).';
COMMENT ON COLUMN conversations.created_at IS 'Creation timestamp.';
COMMENT ON COLUMN conversations.updated_at IS 'Last update timestamp (trigger managed).';
COMMENT ON COLUMN conversations.last_message_at IS 'Timestamp of last message for ordering.';
COMMENT ON COLUMN conversations.last_message_id IS 'Last message id for preview (FK added later).';
COMMENT ON COLUMN conversations.message_count IS 'Denormalized count (nullable, computed via COUNT(*) or async worker).';
COMMENT ON COLUMN conversations.last_seq IS 'Last sequence number (guarded updates prevent race conditions).';
COMMENT ON COLUMN conversations.pinned IS 'Pinned conversations appear at top in UI.';
COMMENT ON COLUMN conversations.archived_at IS 'Archive timestamp; NULL means active.';
COMMENT ON COLUMN conversations.deleted_at IS 'Soft-delete timestamp; NULL means active.';
COMMENT ON COLUMN conversations.summary IS 'Optional rolling summary for long chats.';
COMMENT ON COLUMN conversations.summary_updated_at IS 'When summary was last updated.';
COMMENT ON COLUMN conversations.settings IS 'Conversation-level configuration JSON.';
COMMENT ON COLUMN conversations.metadata IS 'Extra metadata JSON.';

CREATE INDEX IF NOT EXISTS conversations_user_last_idx
  ON conversations (user_id, COALESCE(last_message_at, created_at) DESC);
COMMENT ON INDEX conversations_user_last_idx IS
  'Primary sidebar query: list conversations by last activity per user.';

CREATE INDEX IF NOT EXISTS conversations_user_active_idx
  ON conversations (user_id)
  WHERE deleted_at IS NULL;
COMMENT ON INDEX conversations_user_active_idx IS
  'Filters active (non-deleted) conversations per user.';

DROP TRIGGER IF EXISTS trg_conversations_updated_at ON conversations;
CREATE TRIGGER trg_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================================================
-- Table: llm_requests
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_requests (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  -- Primary key UUID for the LLM request.

  conversation_id    uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  -- Conversation this request belongs to. Cascades delete with the conversation.

  user_id            uuid,
  -- User who initiated this request. Denormalized for quick authorization checks. Nullable for skip-auth approach.
  -- ForeignKey("users.id", ondelete="CASCADE") - commented out until users table exists

  provider           text NOT NULL,
  -- LLM provider (openai/gemini/etc).

  model              text NOT NULL,
  -- Model id/name used for inference.

  request_params     jsonb NOT NULL DEFAULT '{}'::jsonb,
  -- Request parameters (temperature, max_tokens, etc.) stored as JSON.

  prompt_tokens      integer,
  -- Token usage accounting (prompt).

  completion_tokens  integer,
  -- Token usage accounting (completion).

  reasoning_tokens   integer,
  -- Token usage accounting (reasoning).

  total_tokens       integer,
  -- Total tokens (if provided/derived).

  cost_usd           numeric(12,6),
  -- Cost in USD for this request (numeric for precision).

  latency_ms         integer,
  -- End-to-end latency in milliseconds.

  ttft_ms            integer,
  -- Time to first token in milliseconds.

  tps                integer,
  -- Tokens per second.

  error_code         text,
  -- Optional error code for failed requests.

  error_message      text,
  -- Optional error details (keep non-sensitive).

  created_at         timestamptz NOT NULL DEFAULT now(),
  -- Request creation timestamp.

  updated_at         timestamptz NOT NULL DEFAULT now(),
  -- Last update timestamp (trigger managed).

  user_message_id    uuid,
  -- Anchor point for the user message that triggered this request (FK added after messages exists).

  snapshot_seq       bigint,
  -- Sequence number snapshot when request started.

  client_request_id  text,
  -- Idempotency key from client (unique per conversation).

  included_message_ids jsonb,
  -- Array of message IDs included in LLM context.

  status             text,
  -- Request status: 'pending', 'streaming', 'completed', 'cancelled', 'failed'.

  assistant_message_id uuid
  -- Pre-created assistant message placeholder (FK added after messages exists).
);

COMMENT ON TABLE llm_requests IS
  'LLM request tracking. Stores request-level statistics separate from message content.';

COMMENT ON COLUMN llm_requests.id IS 'Primary key UUID.';
COMMENT ON COLUMN llm_requests.conversation_id IS 'FK to conversations (cascade on delete).';
COMMENT ON COLUMN llm_requests.user_id IS 'Denormalized FK to users for auth/audit.';
COMMENT ON COLUMN llm_requests.provider IS 'LLM provider name.';
COMMENT ON COLUMN llm_requests.model IS 'LLM model identifier.';
COMMENT ON COLUMN llm_requests.request_params IS 'Request parameters JSON (temperature, max_tokens, etc.).';
COMMENT ON COLUMN llm_requests.prompt_tokens IS 'Prompt token count.';
COMMENT ON COLUMN llm_requests.completion_tokens IS 'Completion token count.';
COMMENT ON COLUMN llm_requests.reasoning_tokens IS 'Reasoning token count.';
COMMENT ON COLUMN llm_requests.total_tokens IS 'Total token count.';
COMMENT ON COLUMN llm_requests.cost_usd IS 'Approximate cost in USD.';
COMMENT ON COLUMN llm_requests.latency_ms IS 'Generation latency in ms.';
COMMENT ON COLUMN llm_requests.ttft_ms IS 'Time to first token in ms.';
COMMENT ON COLUMN llm_requests.tps IS 'Tokens per second.';
COMMENT ON COLUMN llm_requests.error_code IS 'Error code if failed.';
COMMENT ON COLUMN llm_requests.error_message IS 'Error message/details if failed.';
COMMENT ON COLUMN llm_requests.created_at IS 'Row creation time.';
COMMENT ON COLUMN llm_requests.updated_at IS 'Row last update time (trigger managed).';
COMMENT ON COLUMN llm_requests.user_message_id IS 'Anchor point for the user message that triggered this request.';
COMMENT ON COLUMN llm_requests.snapshot_seq IS 'Sequence number snapshot when request started.';
COMMENT ON COLUMN llm_requests.client_request_id IS 'Idempotency key from client (unique per conversation).';
COMMENT ON COLUMN llm_requests.included_message_ids IS 'Array of message IDs included in LLM context.';
COMMENT ON COLUMN llm_requests.status IS 'Request status: pending/streaming/completed/cancelled/failed.';
COMMENT ON COLUMN llm_requests.assistant_message_id IS 'Pre-created assistant message placeholder.';

CREATE INDEX IF NOT EXISTS llm_requests_conv_idx
  ON llm_requests (conversation_id);
COMMENT ON INDEX llm_requests_conv_idx IS
  'Fast lookup of LLM requests by conversation.';

CREATE INDEX IF NOT EXISTS llm_requests_user_idx
  ON llm_requests (user_id, created_at DESC);
COMMENT ON INDEX llm_requests_user_idx IS
  'Supports user-level analytics and cost tracking.';

CREATE INDEX IF NOT EXISTS llm_requests_created_idx
  ON llm_requests (created_at DESC);
COMMENT ON INDEX llm_requests_created_idx IS
  'Supports time-based queries and analytics.';

CREATE UNIQUE INDEX IF NOT EXISTS llm_requests_conv_client_req_idx
  ON llm_requests (conversation_id, client_request_id)
  WHERE client_request_id IS NOT NULL;
COMMENT ON INDEX llm_requests_conv_client_req_idx IS
  'Idempotency constraint: prevent duplicate requests with same client_request_id per conversation.';

DROP TRIGGER IF EXISTS trg_llm_requests_updated_at ON llm_requests;
CREATE TRIGGER trg_llm_requests_updated_at
BEFORE UPDATE ON llm_requests
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================================================
-- Table: messages
-- ============================================================================

CREATE TABLE IF NOT EXISTS messages (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  -- Primary key UUID for the message.

  conversation_id    uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  -- Conversation this message belongs to. Cascades delete with the conversation.

  user_id            uuid,
  -- Denormalized user_id for quick authorization checks & auditing. Nullable for skip-auth approach.
  -- ForeignKey("users.id", ondelete="CASCADE") - commented out until users table exists

  role               message_role NOT NULL,
  -- Message role: system/user/assistant/tool.

  status             message_status NOT NULL DEFAULT 'completed',
  -- Message lifecycle: completed/in_progress/cancelled/error.

  seq                bigint NOT NULL,
  -- Stable per-conversation ordering number (use instead of created_at for pagination).

  content            text NOT NULL,
  -- Message text (assistant output or user input). Keep as text to avoid bloat.

  content_format     text NOT NULL DEFAULT 'text/markdown',
  -- Content type hint (text/plain, text/markdown, etc).

  metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
  -- Flexible JSON: citations, tool info, UI flags, structured payloads.

  client_msg_id      text,
  -- Optional idempotency key from client. Unique per conversation to prevent duplicates on retries.

  request_id         uuid REFERENCES llm_requests(id) ON DELETE SET NULL,
  -- FK to llm_requests table. Links message to the LLM request that generated it.

  created_at         timestamptz NOT NULL DEFAULT now(),
  -- Creation timestamp.

  updated_at         timestamptz NOT NULL DEFAULT now(),
  -- Updated timestamp (trigger managed).

  UNIQUE (conversation_id, seq),
  -- Prevent two messages from claiming the same sequence number.

  UNIQUE (conversation_id, client_msg_id)
  -- If client_msg_id is provided, enforce idempotency within a conversation.
);

COMMENT ON TABLE messages IS
  'Conversation transcript. Uses stable per-conversation seq for deterministic ordering.';

COMMENT ON COLUMN messages.id IS 'Primary key UUID.';
COMMENT ON COLUMN messages.conversation_id IS 'FK to conversations (cascade on delete).';
COMMENT ON COLUMN messages.user_id IS 'Denormalized FK to users for auth/audit.';
COMMENT ON COLUMN messages.role IS 'Role: system/user/assistant/tool.';
COMMENT ON COLUMN messages.status IS 'Status: completed/in_progress/cancelled/error.';
COMMENT ON COLUMN messages.seq IS 'Stable per-conversation ordering number.';
COMMENT ON COLUMN messages.content IS 'Message body (text).';
COMMENT ON COLUMN messages.content_format IS 'Content format hint (e.g., text/markdown).';
COMMENT ON COLUMN messages.metadata IS 'JSON metadata: citations, tool payloads, etc.';
COMMENT ON COLUMN messages.client_msg_id IS 'Client idempotency id (unique per conversation).';
COMMENT ON COLUMN messages.request_id IS 'FK to llm_requests table. Links message to LLM request.';
COMMENT ON COLUMN messages.created_at IS 'Row creation time.';
COMMENT ON COLUMN messages.updated_at IS 'Row last update time (trigger managed).';

CREATE INDEX IF NOT EXISTS messages_conv_seq_idx
  ON messages (conversation_id, seq);
COMMENT ON INDEX messages_conv_seq_idx IS
  'Primary pagination index: fetch messages by conversation_id ordered by seq.';

CREATE INDEX IF NOT EXISTS messages_conv_created_idx
  ON messages (conversation_id, created_at);
COMMENT ON INDEX messages_conv_created_idx IS
  'Secondary index for time-based queries / debugging.';

CREATE INDEX IF NOT EXISTS messages_request_id_idx
  ON messages (request_id)
  WHERE request_id IS NOT NULL;
COMMENT ON INDEX messages_request_id_idx IS
  'Fast lookup of messages by LLM request id.';

CREATE INDEX IF NOT EXISTS messages_metadata_gin_idx
  ON messages USING gin (metadata);
COMMENT ON INDEX messages_metadata_gin_idx IS
  'Supports querying/filtering inside JSON metadata (citations, tool data).';

DROP TRIGGER IF EXISTS trg_messages_updated_at ON messages;
CREATE TRIGGER trg_messages_updated_at
BEFORE UPDATE ON messages
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================================================
-- Add llm_requests table FK constraint (after llm_requests exists)
-- ============================================================================

-- Note: llm_requests table is created before messages, so the FK in messages
-- references llm_requests correctly. No additional constraint needed here.

-- ============================================================================
-- Add conversations.last_message_id FK (after messages exists)
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'conversations_last_message_fk'
  ) THEN
    ALTER TABLE conversations
      ADD CONSTRAINT conversations_last_message_fk
      FOREIGN KEY (last_message_id) REFERENCES messages(id)
      DEFERRABLE INITIALLY DEFERRED;
  END IF;
END $$;

COMMENT ON CONSTRAINT conversations_last_message_fk ON conversations IS
  'Optional FK to last message for fast conversation preview. Deferrable for safe transactions.';

-- ============================================================================
-- Add llm_requests FKs to messages (after messages exists)
-- ============================================================================

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'llm_requests_user_message_fk'
  ) THEN
    ALTER TABLE llm_requests
      ADD CONSTRAINT llm_requests_user_message_fk
      FOREIGN KEY (user_message_id) REFERENCES messages(id) ON DELETE SET NULL;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'llm_requests_assistant_message_fk'
  ) THEN
    ALTER TABLE llm_requests
      ADD CONSTRAINT llm_requests_assistant_message_fk
      FOREIGN KEY (assistant_message_id) REFERENCES messages(id) ON DELETE SET NULL;
  END IF;
END $$;

COMMENT ON CONSTRAINT llm_requests_user_message_fk ON llm_requests IS
  'FK to user message that triggered this request.';
COMMENT ON CONSTRAINT llm_requests_assistant_message_fk ON llm_requests IS
  'FK to pre-created assistant message placeholder.';

SQL

echo ">> Granting permissions to application user..."

# Grant permissions to application user (using variable substitution)
psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${APP_DB}" <<SQL
-- Grant usage on schema (public is default)
GRANT USAGE ON SCHEMA public TO ${APP_DB_USER};

-- Grant all privileges on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${APP_DB_USER};

-- Grant privileges on sequences (for SERIAL columns, though we use UUIDs)
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${APP_DB_USER};

-- Grant privileges on future tables and sequences (for tables created later)
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO ${APP_DB_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO ${APP_DB_USER};
SQL

echo ">> Core schema init completed."
