"""[telecom_boed] Task-split freeze and contamination guards (owner: W1-S3). Test-40 ids are allowed here only."""
from __future__ import annotations

import json
from pathlib import Path

from telecom_boed import paths
from telecom_boed.testing.leakage_scan import branch_pool_task_ids, parse_task_id

FIXTURE = Path(__file__).parent / "fixtures" / "W1-S3" / "test40_ids.json"


def test_split_md5_pinned():
    paths.assert_catalog_pins()


def test_test40_frozen(splits):
    frozen = json.loads(FIXTURE.read_text())
    assert frozen == sorted(frozen) and len(frozen) == len(set(frozen)) == 40
    assert sorted(splits["test"]) == frozen


def test_split_algebra(splits, catalog):
    ids = [t["id"] for t in catalog]
    assert len(ids) == len(set(ids)) == 2285
    small, train, test, base, full = (set(splits[k]) for k in ("small", "train", "test", "base", "full"))
    assert {k: len(splits[k]) for k in ("small", "train", "test", "base", "full")} == {
        "small": 20, "train": 74, "test": 40, "base": 114, "full": 2285}
    assert not train & test
    assert base == train | test
    assert small < full
    assert full == set(ids)
    assert (small | train | test | base) <= set(ids)  # every split id exists in tasks.json


def test_branch_pool_excludes_test40(catalog, test40_ids):
    pool = branch_pool_task_ids(catalog, test40_ids)
    assert not set(pool) & test40_ids
    banned = {parse_task_id(i)[1] for i in test40_ids}
    assert all(parse_task_id(i)[1] not in banned for i in pool)
    loose = branch_pool_task_ids(catalog, test40_ids, intent_keyed=True)
    assert set(pool) <= set(loose) and not set(loose) & test40_ids
    print(f"\n[W1-S3] pool(fault-set exclusion)={len(pool)} pool((intent,faults) exclusion)={len(loose)} "
          f"extra kept by intent-keyed={len(loose) - len(pool)} of {len(catalog)}")
