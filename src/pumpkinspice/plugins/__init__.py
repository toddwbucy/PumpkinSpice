"""Built-in plugins, each registered via an entry point in pyproject.toml.

These ship with the core for the first vertical slice. Backends added later
(arango KG, and -- last, as an explicitly-flagged non-control run -- HADES) can
live here or in separate packages; the kernel discovers them the same way.
"""
