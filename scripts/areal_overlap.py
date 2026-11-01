#!/usr/bin/env python3
"""Check AReaL tau2 telecom training rows for overlap with the frozen test-40 split.

Stdlib only. Run as:
  python3 -I scripts/areal_overlap.py --areal DIR_OR_FILES --tasks tasks.json \
      --split split_tasks.json --out report.md [--source URL ...]

Task ids have the form "[intent]fault1|fault2|...[PERSONA:x]". A "key" is (intent, faults).

Mapping rules
- SFT rows: metadata.task_id is parsed directly. Rows without task_id are non-telecom and skipped.
- RL rows: they carry no task id (ids look like telecom_N). The row's initial_state (preferred)
  or user_scenario is matched by exact canonical JSON against tasks.json. The row gets a key only
  if all matching catalog tasks agree on it. Ticket text is NOT used: tasks.json has only a few
  distinct ticket strings, so it cannot identify a task.

Overlap criteria, per AReaL row, against the test ids:
  a) exact task id (SFT task_id; RL row whose match resolves to exactly one catalog id)
  b) same (intent, fault set)
  c) same fault set under any intent
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import sys

ID_RE = re.compile(r"^\[([^\]]+)\](.*)\[PERSONA:([^\]]*)\]$")
BAD = object()


def parse_id(tid):
    if not isinstance(tid, str):
        return None
    m = ID_RE.match(tid.strip())
    if not m:
        return None
    intent = m.group(1).strip()
    faults = tuple(sorted(f for f in m.group(2).split("|") if f))
    return intent, faults


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def find_files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.endswith((".jsonl", ".json")):
                        out.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            out.append(p)
        else:
            sys.exit(f"error: not found: {p}")
    return sorted(set(out))


def iter_rows(path):
    """Yield parsed rows; yields BAD for undecodable lines. JSON files must be arrays."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                yield BAD
                return
        if isinstance(data, list):
            yield from data
        return  # non-list JSON (e.g. DB snapshots) holds no training rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield BAD


def sha256_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def row_kind(row):
    if not isinstance(row, dict):
        return None
    if isinstance(row.get("metadata"), dict) and "messages" in row:
        return "sft"
    if "user_scenario" in row and "id" in row:
        return "rl"
    return None


def is_telecom_rl(row):
    if str(row.get("id", "")).lower().startswith("telecom"):
        return True
    us = row.get("user_scenario") or {}
    instr = us.get("instructions") if isinstance(us, dict) else None
    return isinstance(instr, dict) and instr.get("domain") == "telecom"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areal", nargs="+", required=True, help="AReaL dir(s) or file(s)")
    ap.add_argument("--tasks", required=True, help="telecom tasks.json (full catalog)")
    ap.add_argument("--split", required=True, help="telecom split_tasks.json")
    ap.add_argument("--split-key", default="test")
    ap.add_argument("--out", required=True, help="markdown report path")
    ap.add_argument("--source", action="append", default=[], help="source URL to cite (repeatable)")
    args = ap.parse_args()

    # Catalog and split
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)
    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)
    test_ids = list(split[args.split_key])
    test_set = set(test_ids)

    cat_key = {}
    init_idx = collections.defaultdict(list)
    us_idx = collections.defaultdict(list)
    tickets = set()
    for t in tasks:
        tid = t["id"]
        cat_key[tid] = parse_id(tid)
        init_idx[canon(t.get("initial_state"))].append(tid)
        us_idx[canon(t.get("user_scenario"))].append(tid)
        tickets.add(canon(t.get("ticket")))
    catalog_intents = {k[0] for k in cat_key.values() if k}
    missing_test = [t for t in test_ids if t not in cat_key]

    test_key = {t: parse_id(t) for t in test_ids}
    test_key_index = collections.defaultdict(list)
    test_fault_index = collections.defaultdict(list)
    for t, k in test_key.items():
        if k:
            test_key_index[k].append(t)
            test_fault_index[k[1]].append(t)

    # AReaL rows
    files = find_files(args.areal)
    file_rows = collections.Counter()
    file_hash = {}
    counts = collections.Counter()
    infos = []

    def handle_sft(row):
        tid = row["metadata"].get("task_id")
        if tid is None:
            return None  # airline/retail SFT rows carry no telecom task id
        p = parse_id(tid)
        if p is None:
            return {"kind": "sft", "exact": None, "key": None, "fset": None,
                    "method": "task_id (unparsed)", "cand_test": False}
        return {"kind": "sft", "exact": tid, "key": p, "fset": p[1],
                "method": "task_id", "cand_test": tid in test_set}

    def handle_rl(row):
        if not is_telecom_rl(row):
            return None
        init = row.get("initial_state")
        us = row.get("user_scenario")
        cands, method = [], None
        if init is not None and init_idx.get(canon(init)):
            cands, method = init_idx[canon(init)], "initial_state"
        elif us is not None and us_idx.get(canon(us)):
            cands, method = us_idx[canon(us)], "user_scenario"
        if not cands:
            return {"kind": "rl", "exact": None, "key": None, "fset": None,
                    "method": "unmatched", "cand_test": False}
        keys = {cat_key[c] for c in cands if cat_key.get(c)}
        faults = {f for _, f in keys}
        key = next(iter(keys)) if len(keys) == 1 else None
        fset = next(iter(faults)) if len(faults) == 1 else None
        exact = cands[0] if len(cands) == 1 else None
        return {"kind": "rl", "exact": exact, "key": key, "fset": fset,
                "method": method, "cand_test": any(c in test_set for c in cands)}

    for path in files:
        base = os.path.basename(path)
        file_hash[base] = sha256_prefix(path)
        for row in iter_rows(path):
            if row is BAD:
                counts["parse_errors"] += 1
                continue
            counts["rows_total"] += 1
            kind = row_kind(row)
            file_rows[(base, kind or "other")] += 1
            if kind == "sft":
                info = handle_sft(row)
                if info is None:
                    counts["sft_nontelecom"] += 1
                    continue
                counts["sft_telecom"] += 1
            elif kind == "rl":
                info = handle_rl(row)
                if info is None:
                    counts["rl_nontelecom"] += 1
                    continue
                counts["rl_telecom"] += 1
            else:
                continue
            info["file"] = base
            infos.append(info)

    # Overlap per row
    hit = {"a": collections.Counter(), "b": collections.Counter(), "c": collections.Counter()}
    rows_hit = {k: collections.Counter() for k in ("a", "b", "c")}
    tests_hit = {k: {"sft": set(), "rl": set()} for k in ("a", "b", "c")}
    per_test = collections.defaultdict(lambda: collections.Counter())
    cand_test_rows = collections.Counter()
    coverage = collections.Counter()
    for info in infos:
        kind = info["kind"]
        if info["key"]:
            state = "key"
        elif info["fset"] is not None:
            state = "fault_set_only"
        elif info["method"] == "unmatched":
            state = "no_match"
        else:
            state = "ambiguous"  # catalog matches exist but their fault sets disagree
        coverage[(kind, state)] += 1
        if info["cand_test"]:
            cand_test_rows[kind] += 1
        a_ids = [info["exact"]] if info["exact"] in test_set else []
        b_ids = test_key_index.get(info["key"], []) if info["key"] else []
        c_ids = test_fault_index.get(info["fset"], []) if info["fset"] is not None else []
        for crit, ids in (("a", a_ids), ("b", b_ids), ("c", c_ids)):
            if ids:
                rows_hit[crit][kind] += 1
                for t in ids:
                    tests_hit[crit][kind].add(t)
                    per_test[t][f"{kind}_{crit}"] += 1

    def total(crit):
        return rows_hit[crit]["sft"] + rows_hit[crit]["rl"]

    unmapped_telecom = sum(coverage[(k, s)] for k in ("sft", "rl") for s in ("ambiguous", "no_match"))
    any_ab = total("a") + total("b")
    if any_ab > 0:
        verdict = (f"contaminated: {len(tests_hit['a']['sft'] | tests_hit['a']['rl'])} test-40 ids match exactly "
                   f"and {len(tests_hit['b']['sft'] | tests_hit['b']['rl'])} match by (intent, fault set)")
    elif total("c") > 0:
        verdict = "undetermined: overlap only at fault-set level (c), no exact or intent-level match"
    elif unmapped_telecom > 0:
        verdict = (f"undetermined: no overlap among mapped rows, but {unmapped_telecom} telecom rows "
                   f"could not be mapped to a key")
    else:
        verdict = "clean: no overlap under (a), (b), or (c); all telecom rows mapped"

    # Report
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    L = []
    L.append("# AReaL tau2 telecom data vs frozen test-40 split")
    L.append("")
    L.append(f"Generated {datetime.date.today().isoformat()} by `scripts/areal_overlap.py` (stdlib only).")
    L.append(f"Catalog: `{args.tasks}` ({len(tasks)} tasks). Split: `{args.split}` key `{args.split_key}` "
             f"({len(test_ids)} ids; not in catalog: {len(missing_test)}).")
    L.append("")
    L.append("## Sources")
    for s in args.source or ["(none given; pass --source)"]:
        L.append(f"- {s}")
    L.append("")
    L.append("## Input files")
    L.append("| file | sha256 (prefix) | kind | rows |")
    L.append("|---|---|---|---|")
    for (base, kind), n in sorted(file_rows.items()):
        L.append(f"| {base} | {file_hash.get(base, '')} | {kind} | {n} |")
    L.append("")
    L.append(f"Telecom rows found: SFT {counts['sft_telecom']}, RL {counts['rl_telecom']} "
             f"(non-telecom skipped: SFT {counts['sft_nontelecom']}, RL {counts['rl_nontelecom']}; "
             f"parse errors {counts['parse_errors']}).")
    L.append("")
    L.append("## Matching method and coverage")
    L.append("- SFT: `metadata.task_id` parsed as `[intent]faults[PERSONA:x]`; key and exact id come directly.")
    L.append("- RL: no task id. Row's `initial_state` (else `user_scenario`) matched by exact canonical JSON "
             f"against {len(tasks)} catalog tasks. Key assigned only if all matches agree.")
    L.append(f"- Ticket text not used: tasks.json has {len(tickets)} distinct ticket strings.")
    L.append("")
    L.append("| kind | telecom rows | mapped to (intent, faults) | fault set only | ambiguous (catalog matches disagree) | no catalog match |")
    L.append("|---|---|---|---|---|---|")
    for kind in ("sft", "rl"):
        n = counts[f"{kind}_telecom"]
        L.append(f"| {kind.upper()} | {n} | {coverage[(kind, 'key')]} | {coverage[(kind, 'fault_set_only')]} "
                 f"| {coverage[(kind, 'ambiguous')]} | {coverage[(kind, 'no_match')]} |")
    L.append("")
    L.append(f"RL rows whose matched catalog set contains a test-40 id (candidate level, not exact): "
             f"{cand_test_rows['rl']}. SFT rows with an unparsable task_id: "
             f"{sum(1 for i in infos if i['kind']=='sft' and i['key'] is None)}.")
    L.append("")
    L.append("## Overlap counts (rows | distinct test-40 tasks)")
    L.append("| criterion | SFT rows | SFT tasks | RL rows | RL tasks | total rows |")
    L.append("|---|---|---|---|---|---|")
    labels = {"a": "(a) exact task id", "b": "(b) same intent + fault set", "c": "(c) same fault set, any intent"}
    for crit in ("a", "b", "c"):
        L.append(f"| {labels[crit]} | {rows_hit[crit]['sft']} | {len(tests_hit[crit]['sft'])} "
                 f"| {rows_hit[crit]['rl']} | {len(tests_hit[crit]['rl'])} | {total(crit)} |")
    L.append("")
    L.append("Distinct test-40 tasks overall (any criterion): "
             f"{len(set().union(*(tests_hit[c]['sft'] | tests_hit[c]['rl'] for c in 'abc')))} of {len(test_ids)}.")
    L.append("")
    L.append("## Overlapping test-40 task ids")
    if per_test:
        L.append("Per-id row counts: a = exact, b = intent+faults, c = faults only.")
        L.append("| test id | SFT a/b/c | RL a/b/c |")
        L.append("|---|---|---|")
        for t in test_ids:
            if t in per_test:
                pt = per_test[t]
                L.append(f"| `{t}` | {pt['sft_a']}/{pt['sft_b']}/{pt['sft_c']} | {pt['rl_a']}/{pt['rl_b']}/{pt['rl_c']} |")
    else:
        L.append("None.")
    L.append("")
    L.append("## Verdict")
    L.append(f"**{verdict.split(':')[0]}.** {verdict.split(':', 1)[1].strip() if ':' in verdict else ''}")
    L.append("")
    L.append("## Limitations")
    L.append("- RL rows have synthetic ids; their key is inferred from exact catalog matches, not from the row's own id.")
    L.append("- A row matched by initial_state or user_scenario is treated as the catalog task's fault set; "
             "rows with ambiguous or no exact match are counted as unmapped, not as clean. "
             "user_scenario is a weak key (shared by up to hundreds of catalog tasks), so it is used only "
             "when it resolves to a single fault set.")
    L.append("- Only telecom rows are checked; airline and retail rows are excluded.")
    L.append("")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
