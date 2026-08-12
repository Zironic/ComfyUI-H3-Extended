"""Finalization shim for router-characterization-only MoBA probe runs.

The historical probe considered a run empty when it had neither exact attention
records nor latent-dynamics records.  The characterization branch can now have
useful lightweight router records even when no expensive snapshot was selected,
so finalization must recognize those records as first-class output.
"""

import logging

from . import moba_capture, moba_report


_ORIGINAL_END = moba_capture.MobaProbeSession.end


def _end_with_router_records(self):
    run = self.run
    self.run = None
    if run is None:
        return None

    run.dynamics_tracker.close()
    router_records = list(getattr(run, "router_dynamics", ()) or ())
    if not run.records and not run.latent_dynamics and not router_records:
        logging.warning("[H3 MoBA3D probe] run finished with no captures")
        return None

    path = moba_report.write_run(run)
    logging.info(
        "[H3 MoBA3D probe] %d attention records, %d dynamics records, %d router records -> %s",
        len(run.records),
        len(run.latent_dynamics),
        len(router_records),
        path,
    )
    return path


if not getattr(moba_capture.MobaProbeSession.end, "_h3_router_records", False):
    _end_with_router_records._h3_router_records = True
    _end_with_router_records._h3_original = _ORIGINAL_END
    moba_capture.MobaProbeSession.end = _end_with_router_records
