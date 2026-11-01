"""[telecom_boed] Static leakage scanner + task-split helpers (owner: W1-S3). Never imports or executes scanned code.

Rule enforced: deployable code (DEPLOYABLE_MODULES) must never read the current episode's `task`, `user_db`,
user tool results, or any `oracle_*` module. Reading the shipped catalog (tasks.json as prior/support) is allowed.
Everything here is AST/text based; see the blind spots listed in the W1-S3 report (dynamic code is only flagged
heuristically).
"""
from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PKG = "telecom_boed"
DEPLOYABLE_MODULES: tuple[str, ...] = (
    "telecom_boed.tau2_ext.agents",
    "telecom_boed.tau2_ext.candidates",
    "telecom_boed.tau2_ext.belief_script",
    "telecom_boed.belief.parsers",
    "telecom_boed.belief.filter",  # oracle mode receives true observations from the caller; must not import oracle_*
)
FORBIDDEN_ATTRS = frozenset({"user_db", "initial_state", "evaluation_criteria", "user_tools", "task"})
FORBIDDEN_STRS = frozenset({"user_db", "initial_state", "evaluation_criteria", "user_tools"})
_REFLECT = frozenset({"getattr", "setattr", "hasattr", "delattr"})
_DYNAMIC = frozenset({"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "import_module"})


# --------------------------------------------------------------------------- task ids / splits
_ID_RE = re.compile(r"\[([^\]]*)\](.*)\[PERSONA:([^\]]*)\]")


def parse_task_id(task_id: str) -> tuple[str, frozenset[str], str]:
    """`[intent]f1|f2[PERSONA:x]` -> (intent, frozenset(faults), persona)."""
    m = _ID_RE.fullmatch(task_id)
    if not m:
        raise ValueError(f"unparseable telecom task id: {task_id!r}")
    intent, faults, persona = m.groups()
    return intent, frozenset(f for f in faults.split("|") if f), persona


def branch_pool_task_ids(catalog: list[dict], test40_ids, *, intent_keyed: bool = False) -> list[str]:
    """Catalog tasks usable as branch/rollout pool: fault set differs from every test-40 fault set.

    Default ignores intent (conservative: a test fault set is excluded under ANY intent). `intent_keyed=True`
    gives the weaker (intent, faults) exclusion, only for sizing the conservatism cost.
    """
    test = set(test40_ids)
    key = (lambda i: parse_task_id(i)[:2]) if intent_keyed else (lambda i: parse_task_id(i)[1])
    banned = {key(t) for t in test}
    return [t["id"] for t in catalog if t["id"] not in test and key(t["id"]) not in banned]


# --------------------------------------------------------------------------- own-source scan
@dataclass(frozen=True)
class Violation:
    module: str  # file path or module name as given
    lineno: int
    kind: str  # oracle_import | forbidden_attr | task_read | task_rebind | forbidden_str | dynamic
    detail: str

    def __str__(self) -> str:
        return f"{self.module}:{self.lineno} [{self.kind}] {self.detail}"


# Privileged rollout modules (the scripted teacher reads the hidden task): deployable code may not import them either, even
# if a future one is not named oracle_*. oracle_agent is additionally caught by the oracle_ name rule.
PRIVILEGED_PREFIXES: tuple[str, ...] = ("telecom_boed.rollout.oracle_agent", "telecom_boed.rollout.privileged")


def _is_oracle(dotted: str) -> bool:
    return any(p.startswith("oracle_") for p in dotted.split(".")) or any(
        dotted == q or dotted.startswith(q + ".") for q in PRIVILEGED_PREFIXES)


class _Scanner(ast.NodeVisitor):
    def __init__(self, label: str):
        self.label, self.out = label, []

    def _add(self, node: ast.AST, kind: str, detail: str) -> None:
        self.out.append(Violation(self.label, getattr(node, "lineno", 0), kind, detail))

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if _is_oracle(a.name):
                self._add(node, "oracle_import", a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = ("." * node.level) + (node.module or "")
        if _is_oracle(base):
            self._add(node, "oracle_import", base)
        for a in node.names:
            if _is_oracle(a.name):
                self._add(node, "oracle_import", f"{base}.{a.name}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self._add(node, "forbidden_attr", f".{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "task":
            if isinstance(node.ctx, ast.Load):
                self._add(node, "task_read", "reads `task` (only `del task` / leaving it unused is allowed)")
            elif isinstance(node.ctx, ast.Store):
                self._add(node, "task_rebind", "binds a local named `task` (rename it)")
        elif node.id in _DYNAMIC and isinstance(node.ctx, ast.Load):
            self._add(node, "dynamic", f"{node.id} (cannot be analysed statically)")

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name in _REFLECT:
            lit = node.args[1] if len(node.args) > 1 else None
            if isinstance(lit, ast.Constant) and isinstance(lit.value, str):
                if lit.value in FORBIDDEN_STRS | {"task"}:
                    self._add(node, "forbidden_str", f"{name}(..., {lit.value!r})")
            else:
                self._add(node, "dynamic", f"{name} with non-literal attribute name")
        if name in ("get", "pop") and node.args:  # kwargs.get("task"), state.get("user_db")
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and a0.value in FORBIDDEN_STRS | {"task"}:
                self._add(node, "forbidden_str", f".{name}({a0.value!r})")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        s = node.slice
        if isinstance(s, ast.Constant) and s.value in FORBIDDEN_STRS | {"task"}:
            self._add(node, "forbidden_str", f"[{s.value!r}]")
        self.generic_visit(node)


def scan_module_source(path: str | Path) -> list[Violation]:
    """Flag leakage patterns in this module's OWN source (imports are checked for oracle_* names only)."""
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return [Violation(str(path), e.lineno or 0, "dynamic", f"unparseable: {e.msg}")]
    sc = _Scanner(str(path))
    sc.visit(tree)
    return sorted(sc.out, key=lambda v: (v.lineno, v.kind))


# --------------------------------------------------------------------------- import closure
def _src_root(package_root: str | Path) -> Path:
    p = Path(package_root)
    return p.parent if p.name == PKG else p


def _module_file(name: str, root: Path) -> Path | None:
    rel = Path(*name.split("."))
    for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _direct_imports(name: str, root: Path) -> set[str]:
    """telecom_boed.* modules imported (anywhere in the file, incl. inside functions / TYPE_CHECKING) by `name`.

    Parent packages are included, since importing a.b.c executes a/__init__ and a/b/__init__.
    """
    f = _module_file(name, root)
    if f is None:
        return set()
    tree = ast.parse(f.read_text(), filename=str(f))
    pkg = name if f.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                up = pkg.split(".")
                up = up[: len(up) - (node.level - 1)]
                base = ".".join(up + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            targets = [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]
        else:
            continue
        for t in targets:
            if t == PKG or t.startswith(PKG + "."):
                # `from pkg import name` -> `pkg.name` only counts if it is a module (or already oracle-named)
                if _module_file(t, root) or _is_oracle(t) or t in targets[:1]:
                    found.update(_parents(t))
    found.discard(name)
    return found


def _closure(module_name: str, root: Path) -> dict[str, str | None]:
    """BFS closure: module -> predecessor (None for the start). Includes missing-but-referenced modules."""
    seen: dict[str, str | None] = {module_name: None}
    q = deque([module_name])
    while q:
        m = q.popleft()
        for d in sorted(_direct_imports(m, root)):
            if d not in seen:
                seen[d] = m
                q.append(d)
    return seen


def transitive_imports(module_name: str, package_root: str | Path) -> set[str]:
    """Static import closure of `module_name` within telecom_boed (excludes the module itself).

    `package_root` = the dir containing the `telecom_boed` package (e.g. <repo>/src); the package dir itself also works.
    """
    c = _closure(module_name, _src_root(package_root))
    c.pop(module_name)
    return set(c)


# --------------------------------------------------------------------------- deployable check
@dataclass
class ModuleReport:
    module: str
    violations: list[Violation] = field(default_factory=list)
    oracle_chains: dict[str, list[str]] = field(default_factory=dict)  # oracle module -> import chain from `module`
    closure_violations: list[Violation] = field(default_factory=list)  # only with scan_closure=True

    @property
    def clean(self) -> bool:
        return not (self.violations or self.oracle_chains or self.closure_violations)


@dataclass
class DeployableReport:
    modules: dict[str, ModuleReport] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)  # named but not yet on disk (W2/W3 create them)

    @property
    def ok(self) -> bool:
        return all(m.clean for m in self.modules.values())

    def problems(self) -> list[str]:
        out: list[str] = []
        for m in self.modules.values():
            out += [str(v) for v in m.violations + m.closure_violations]
            out += [f"{m.module} reaches {o} via {' -> '.join(c)}" for o, c in m.oracle_chains.items()]
        return out


def check_deployable(module_names=DEPLOYABLE_MODULES, package_root: str | Path | None = None, *, scan_closure: bool = False) -> DeployableReport:
    """Own-source violations + oracle_* reachability for each existing module; missing modules go to `pending`.

    `scan_closure=True` additionally runs the own-source scan over every telecom_boed module in the closure
    (stricter: flags helpers that read task/user_db even if they are not oracle_*-named).
    """
    if package_root is None:
        package_root = Path(__file__).resolve().parents[2]  # .../src
    root = _src_root(package_root)
    rep = DeployableReport()
    for name in module_names:
        f = _module_file(name, root)
        if f is None:
            rep.pending.append(name)
            continue
        mr = ModuleReport(name, scan_module_source(f))
        clos = _closure(name, root)
        for mod in clos:
            if mod != name and _is_oracle(mod):
                chain, cur = [], mod
                while cur is not None:
                    chain.append(cur)
                    cur = clos[cur]
                mr.oracle_chains[mod] = chain[::-1]
        if scan_closure:
            for mod in clos:
                g = _module_file(mod, root)
                if mod != name and g is not None and not _is_oracle(mod):
                    mr.closure_violations += scan_module_source(g)
        rep.modules[name] = mr
    return rep
