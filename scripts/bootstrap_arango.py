#!/usr/bin/env python
"""Build-side ArangoDB provisioning for the arango retrieval backend.

Runs as the Arango root user (admin only; password from $ARANGO_PASSWORD). It
creates, idempotently:

  - a dedicated database ``herobench_kg`` for the belief-node corpus,
  - a ``belief_nodes`` document collection,
  - two least-privilege users:
      * ``ps_loader``   read-write on herobench_kg (build-side seeding),
      * ``ps_agent_ro`` READ-ONLY on herobench_kg (the model runtime),
    both with **default database access "none"** so they cannot see or touch any
    other database on this shared server.

Scoped credentials are written to the chmod-600, gitignored ``.env.local``.
The root password is never used by the runtime.

Run:  uv run --extra arango python scripts/bootstrap_arango.py
      uv run --extra arango python scripts/bootstrap_arango.py --rotate
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

from arango import ArangoClient

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env.local"


def gen_password() -> str:
    return secrets.token_urlsafe(24)


def write_env(updates: dict[str, str]) -> None:
    """Merge values into a chmod-600 .env.local (never committed)."""
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


def ensure_user(sys_db, name, password):
    if sys_db.has_user(name):
        if password is not None:
            sys_db.update_user(username=name, password=password, active=True)
            print(f"  ~ reset password for user {name}")
        else:
            print(f"  = user {name} exists (password unchanged)")
    else:
        sys_db.create_user(username=name, password=password, active=True)
        print(f"  + created user {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("ARANGO_URL", "http://localhost:8529"))
    ap.add_argument("--root-user", default=os.environ.get("ARANGO_ROOT_USER", "root"))
    ap.add_argument("--db", default="herobench_kg")
    ap.add_argument("--collection", default="belief_nodes")
    ap.add_argument("--loader-user", default="ps_loader")
    ap.add_argument("--agent-user", default="ps_agent_ro")
    ap.add_argument("--rotate", action="store_true")
    args = ap.parse_args(argv)

    root_pw = os.environ.get("ARANGO_PASSWORD")
    if not root_pw:
        print("error: $ARANGO_PASSWORD not set (root creds, admin-only).", file=sys.stderr)
        return 1

    client = ArangoClient(hosts=args.url)
    sys_db = client.db("_system", username=args.root_user, password=root_pw)

    loader_new = not sys_db.has_user(args.loader_user)
    agent_new = not sys_db.has_user(args.agent_user)
    loader_pw = gen_password() if (loader_new or args.rotate) else None
    agent_pw = gen_password() if (agent_new or args.rotate) else None

    print("provisioning arango backend...")
    ensure_user(sys_db, args.loader_user, loader_pw)
    ensure_user(sys_db, args.agent_user, agent_pw)

    if not sys_db.has_database(args.db):
        sys_db.create_database(args.db)
        print(f"  + created database {args.db}")
    else:
        print(f"  = database {args.db} exists")

    # Least privilege: default access "none" to every database, then grant only
    # this one. This blocks the user from every other DB on the shared server.
    for user, level in ((args.loader_user, "rw"), (args.agent_user, "ro")):
        sys_db.update_permission(username=user, permission="none", database="*")
        sys_db.update_permission(username=user, permission=level, database=args.db)
    print(f"  set default access=none; {args.loader_user}=rw, {args.agent_user}=ro on {args.db}")

    db = client.db(args.db, username=args.root_user, password=root_pw)
    if not db.has_collection(args.collection):
        db.create_collection(args.collection)
        print(f"  + created collection {args.collection}")
    else:
        print(f"  = collection {args.collection} exists")

    env_updates: dict[str, str] = {}
    if loader_pw is not None:
        env_updates["ARANGO_LOADER_USER"] = args.loader_user
        env_updates["ARANGO_LOADER_PASSWORD"] = loader_pw
    if agent_pw is not None:
        env_updates["ARANGO_AGENT_USER"] = args.agent_user
        env_updates["ARANGO_AGENT_PASSWORD"] = agent_pw
    if env_updates:
        write_env(env_updates)
        print(f"  wrote {sorted(env_updates)} to {ENV_FILE} (chmod 600)")

    if agent_pw is not None:
        verify(client, args, agent_pw)
    else:
        print("  (skipping verify: agent password unknown this run; use --rotate)")

    print("done.")
    return 0


def verify(client, args, agent_pw) -> None:
    """Prove the runtime user is read-only on its DB and locked out of others."""
    agent_db = client.db(args.db, username=args.agent_user, password=agent_pw)
    n = agent_db.collection(args.collection).count()

    # write must be denied
    try:
        agent_db.collection(args.collection).insert({"_key": "_probe", "text": "x"})
    except Exception:
        print(f"  ok: write to {args.collection} correctly DENIED")
    else:
        raise AssertionError("SECURITY: read-only user could WRITE!")

    # access to another database must be denied (default access = none)
    other = next(
        (
            d
            for d in client.db(
                "_system", username=args.root_user, password=os.environ["ARANGO_PASSWORD"]
            ).databases()
            if d not in (args.db, "_system")
        ),
        None,
    )
    if other:
        try:
            client.db(other, username=args.agent_user, password=agent_pw).collections()
        except Exception:
            print(f"  ok: access to other DB '{other}' correctly DENIED")
        else:
            raise AssertionError(f"SECURITY: read-only user reached other DB '{other}'!")
    print(f"  verify done: read-only user can read (docs={n}), cannot write or reach other DBs")


if __name__ == "__main__":
    raise SystemExit(main())
