#!/usr/bin/env python
"""Build-side Postgres provisioning for the pgvector retrieval backend.

Runs as the Postgres SUPERUSER (root) -- admin only. It creates, idempotently:

  - a dedicated database (default ``herobench_kg``) for the belief-node corpus,
  - two least-privilege login roles:
      * ``ps_loader``   read-write, build-side seeding identity,
      * ``ps_agent_ro`` READ-ONLY, the model/agent runtime identity,
  - the ``vector`` extension, a ``kg`` schema, and ``kg.belief_nodes``
    (id, text, metadata, embedding vector(N)) with an HNSW cosine index,
  - grants so the runtime role can only SELECT, never write or reach DDL.

The two scoped DSNs are written to a chmod-600, gitignored ``.env.local`` --
NEVER the repo, never stdout. The root password is read from $POSTGRESQL_PASSWORD
and is never used by the runtime.

Run:  uv run --extra pgvector python scripts/bootstrap_pg.py
      uv run --extra pgvector python scripts/bootstrap_pg.py --rotate   # reset pws
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

import psycopg
from psycopg import sql

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env.local"


def gen_password() -> str:
    # URL-safe -> no DSN-escaping needed.
    return secrets.token_urlsafe(24)


def role_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (name,))
    return cur.fetchone() is not None


def db_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
    return cur.fetchone() is not None


def ensure_role(cur, name: str, password: str | None) -> None:
    """Create or (when password given) reset a locked-down login role.

    Always enforces NOSUPERUSER/NOCREATEDB/NOCREATEROLE so the role can never
    escalate or create databases.
    """
    attrs = sql.SQL("LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION")
    ident = sql.Identifier(name)
    if not role_exists(cur, name):
        stmt = sql.SQL("CREATE ROLE {} WITH {} PASSWORD {}").format(
            ident, attrs, sql.Literal(password)
        )
        cur.execute(stmt)
        print(f"  + created role {name}")
    else:
        # enforce attributes; only touch the password when rotating
        cur.execute(sql.SQL("ALTER ROLE {} WITH {}").format(ident, attrs))
        if password is not None:
            cur.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(ident, sql.Literal(password)))
            print(f"  ~ reset password for role {name}")
        else:
            print(f"  = role {name} already exists (password unchanged)")


def admin_phase(root_dsn: str, args, loader_pw, agent_pw) -> None:
    """Role + database + database-level grants. Needs autocommit (CREATE DATABASE)."""
    with psycopg.connect(root_dsn, autocommit=True) as conn, conn.cursor() as cur:
        ensure_role(cur, args.loader_role, loader_pw)
        ensure_role(cur, args.agent_role, agent_pw)

        if not db_exists(cur, args.db):
            cur.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(args.db), sql.Identifier(args.loader_role)
                )
            )
            print(f"  + created database {args.db}")
        else:
            print(f"  = database {args.db} already exists")

        dbid = sql.Identifier(args.db)
        loader = sql.Identifier(args.loader_role)
        agent = sql.Identifier(args.agent_role)
        # Lock our DB: only our roles may connect (does not touch other DBs).
        cur.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(dbid))
        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}, {}").format(dbid, loader, agent))
        # Per-DB search_path so unqualified table names resolve to kg.
        for r in (loader, agent):
            cur.execute(
                sql.SQL("ALTER ROLE {} IN DATABASE {} SET search_path = kg, public").format(r, dbid)
            )
    print(f"  admin phase done (db={args.db}, roles={args.loader_role},{args.agent_role})")


def schema_phase(db_dsn: str, args) -> None:
    """Extension, schema, table, index, grants -- run while connected to the DB."""
    loader = sql.Identifier(args.loader_role)
    agent = sql.Identifier(args.agent_role)
    with psycopg.connect(db_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS kg AUTHORIZATION {}").format(loader))
        # Hygiene: nothing reaches kg via PUBLIC.
        cur.execute("REVOKE ALL ON SCHEMA kg FROM PUBLIC")
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS kg.belief_nodes ("
                "  id text PRIMARY KEY,"
                "  text text NOT NULL,"
                "  metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,"
                "  embedding vector({}) "
                ")"
            ).format(sql.Literal(args.dim))
        )
        cur.execute(sql.SQL("ALTER TABLE kg.belief_nodes OWNER TO {}").format(loader))
        # HNSW cosine index (works on an empty table; pgvector >= 0.5).
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS belief_nodes_embedding_hnsw "
                "ON kg.belief_nodes USING hnsw (embedding vector_cosine_ops)"
            )
        except psycopg.Error as exc:
            print(f"  ! HNSW index skipped ({exc.diag.message_primary}); add after seeding")

        # Read-only runtime role: USAGE + SELECT only, now and for future tables.
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA kg TO {}").format(agent))
        cur.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA kg TO {}").format(agent))
        cur.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA kg GRANT SELECT ON TABLES TO {}"
            ).format(loader, agent)
        )
    print("  schema phase done (kg.belief_nodes ready, agent role is SELECT-only)")


def verify(agent_dsn: str, args) -> None:
    """Prove the runtime role is genuinely read-only and unprivileged."""
    with psycopg.connect(agent_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kg.belief_nodes")
        n = cur.fetchone()[0]
        cur.execute("SHOW is_superuser")
        is_super = cur.fetchone()[0]
        assert is_super == "off", "runtime role must not be superuser!"

        def must_fail(action_sql: str, label: str) -> None:
            try:
                cur.execute(action_sql)
            except psycopg.errors.InsufficientPrivilege:
                conn.rollback()
                print(f"  ok: {label} correctly DENIED")
            else:
                raise AssertionError(f"SECURITY: {label} was ALLOWED for read-only role!")

        must_fail("INSERT INTO kg.belief_nodes (id, text) VALUES ('x','y')", "INSERT")
        must_fail("CREATE TABLE kg.should_not_exist (i int)", "CREATE TABLE")
    print(f"  verify done: read-only role can SELECT (rows={n}), cannot write or DDL")


def write_env(updates: dict[str, str]) -> None:
    """Merge DSNs into a chmod-600 .env.local (never committed)."""
    existing: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                existing[k] = v
    existing.update(updates)
    body = "# PumpkinSpice scoped DB credentials -- gitignored, do not commit.\n"
    body += "\n".join(f"{k}={v}" for k, v in sorted(existing.items())) + "\n"
    ENV_FILE.write_text(body)
    ENV_FILE.chmod(0o600)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    ap.add_argument("--port", default=os.environ.get("PGPORT", "5432"))
    ap.add_argument("--superuser", default=os.environ.get("PGUSER", "postgres"))
    ap.add_argument("--db", default="herobench_kg", help="corpus database name")
    ap.add_argument("--loader-role", default="ps_loader")
    ap.add_argument("--agent-role", default="ps_agent_ro")
    ap.add_argument("--dim", type=int, default=768, help="embedding dimension")
    ap.add_argument(
        "--rotate", action="store_true", help="reset role passwords and rewrite .env.local"
    )
    args = ap.parse_args(argv)

    root_pw = os.environ.get("POSTGRESQL_PASSWORD")
    if not root_pw:
        print("error: $POSTGRESQL_PASSWORD not set (root creds, admin-only).", file=sys.stderr)
        return 1

    base = f"host={args.host} port={args.port} user={args.superuser} password={root_pw}"
    root_dsn = f"{base} dbname=postgres"
    db_dsn = f"{base} dbname={args.db}"

    # Decide whether we are (re)issuing passwords. On first creation we must.
    with psycopg.connect(root_dsn, autocommit=True) as conn, conn.cursor() as cur:
        loader_is_new = not role_exists(cur, args.loader_role)
        agent_is_new = not role_exists(cur, args.agent_role)

    loader_pw = gen_password() if (loader_is_new or args.rotate) else None
    agent_pw = gen_password() if (agent_is_new or args.rotate) else None

    print("provisioning pgvector backend...")
    admin_phase(root_dsn, args, loader_pw, agent_pw)
    schema_phase(db_dsn, args)

    # Build the scoped DSNs we know the password for; write what we can.
    env_updates: dict[str, str] = {}
    if loader_pw is not None:
        env_updates["PUMPKINSPICE_PG_LOADER_DSN"] = (
            f"postgresql://{args.loader_role}:{loader_pw}@{args.host}:{args.port}/{args.db}"
        )
    if agent_pw is not None:
        env_updates["PUMPKINSPICE_PG_DSN"] = (
            f"postgresql://{args.agent_role}:{agent_pw}@{args.host}:{args.port}/{args.db}"
        )
    if env_updates:
        write_env(env_updates)
        print(f"  wrote {sorted(env_updates)} to {ENV_FILE} (chmod 600)")
    else:
        print("  (roles existed; passwords unchanged -- pass --rotate to reissue + rewrite env)")

    # Verify using the agent DSN if we have it; else connect with root-known check skipped.
    if agent_pw is not None:
        agent_dsn = env_updates["PUMPKINSPICE_PG_DSN"]
        verify(agent_dsn, args)
    else:
        print("  (skipping verify: agent password unknown this run; use --rotate to refresh)")

    print("done. Source the DSN for a run:  set -a; . .env.local; set +a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
