"""[telecom_boed] Repair-relevant BOED + reach-avoid layer on τ²-bench telecom.

Import rule: this package must NOT import tau2 at import time. tau2 reads TAU2_DATA_DIR once, when
`tau2.utils.utils` is first imported, so the data dir has to be set first. Entry points (tests/conftest.py,
tau2_ext/launch.py) call `telecom_boed.paths.prepare_tau2_data_dir(...)` and then
`telecom_boed.env.tau2_patches.apply()` (W1-S1) before any tau2 import.
"""
