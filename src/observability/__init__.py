"""Observability package.

Purges the Prometheus multiprocess shard directory on first import. This module
body runs before ``src.observability.metrics`` can be imported, which makes it the
one hook that is guaranteed to land before the first metric write in every
entrypoint — uvicorn, the Celery workers, eval scripts and tests alike — without
depending on import order inside those entrypoints.

See :mod:`src.observability.multiproc` for why the purge is needed at all.
"""

from __future__ import annotations

from src.observability.multiproc import purge_multiproc_dir

purge_multiproc_dir()
