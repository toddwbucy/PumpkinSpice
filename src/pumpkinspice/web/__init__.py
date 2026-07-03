"""Web frontend backend for PumpkinSpice.

A FastAPI app (decoder-agnostic) that drives the SAME harness the CLI does:
a decoder playground, HeroBench run launching + live turn streaming, and a
capture browser. It is a UI layer only -- it does not change any experiment
invariant (read-only DB role, conventional retrieval, no hades).
"""
