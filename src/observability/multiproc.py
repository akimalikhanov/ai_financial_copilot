"""Startup housekeeping for the ``PROMETHEUS_MULTIPROC_DIR`` shard directory.

A delta gauge (``.inc()`` on start, ``.dec()`` on finish) is only correct in
multiprocess mode if the directory is empty when the process group starts. A
child killed by SIGKILL — the OOM killer, most often — never runs its ``.dec()``
and never reaches ``worker_process_shutdown``, so its ``+1`` stays on disk.

Two things then keep the stale value alive across a restart:

* the K8s volume is an ``emptyDir``, which is *pod*-scoped. An OOMKill restarts
  the container in place, so the directory survives.
* shard files are named only by PID. A fresh container restarts PIDs from a low
  number, so a new child reopens its dead predecessor's file and *inherits* the
  leaked value rather than starting at zero.

``make worker`` already did ``rm -rf`` before launching; this is that same step
for every entrypoint, container ones included.
"""

from __future__ import annotations

import glob
import logging
import os

logger = logging.getLogger(__name__)

_purged = False


def purge_multiproc_dir() -> None:
    """Delete stale metric shards. Call before the first metric write in the process.

    Must run in the parent, before any child is forked and before
    ``prometheus_client`` records anything — deleting a shard that a live process
    has already mmap'd drops that process's values silently. Prefork children
    inherit the imported module rather than re-running it, so the package-body call
    site fires once per process tree; the ``_purged`` guard covers the rest (an
    explicit second call, a spawned child that re-imports, a reloader).

    A no-op when ``PROMETHEUS_MULTIPROC_DIR`` is unset (single-process mode).
    """
    global _purged
    if _purged:
        return
    _purged = True

    mpdir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not mpdir:
        return

    try:
        os.makedirs(mpdir, exist_ok=True)
    except OSError:
        logger.warning("multiproc.purge_mkdir_failed", extra={"dir": mpdir}, exc_info=True)
        return

    removed = 0
    for path in glob.glob(os.path.join(mpdir, "*.db")):
        try:
            os.remove(path)
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            # Never let metrics housekeeping stop the process from booting.
            logger.warning("multiproc.purge_unlink_failed", extra={"path": path}, exc_info=True)

    logger.info("multiproc.purged", extra={"dir": mpdir, "removed": removed})
