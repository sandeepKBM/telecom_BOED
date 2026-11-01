"""Synthetic CLEAN deployable: reads the shipped catalog via paths; accepts task in **kwargs without reading it."""
import json
from telecom_boed import paths


def make_agent(*, task=None, **kwargs):
    del task
    tasks_path, _ = paths.telecom_catalog_paths()
    return {"catalog": tasks_path, "n_extra": len(kwargs), "dump": json.dumps({"x": 1})}


def make_agent2(task, **kw):  # task accepted and never read
    return kw
