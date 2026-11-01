"""[telecom_boed] Runtime poisoning: stand-ins for the episode `task` / `user_db` that raise on ANY use.

W3 passes `PoisonTask()` as `task=` to agent factories and `PoisonDB()` as user_db proxies, then asserts the
agent's output is identical to a run with the real objects. `LeakageError` derives from BaseException so a
sloppy `except Exception:` inside agent code cannot swallow a leak; every attempt is also appended to LEAK_LOG,
and `poisoned()` fails on exit if anything was touched, even if the exception was caught.
Only `__class__` and `__repr__` (-> "<PoisonTask>" / "<PoisonDB>") are usable.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

LEAK_LOG: list[str] = []


class LeakageError(BaseException):
    """Deployable code touched hidden simulator state (task / user_db)."""


def _raiser(name: str) -> Callable[..., Any]:
    def method(self: Any, *a: Any, **k: Any) -> Any:
        _trip(type(self), f"<{name}>")

    method.__name__ = name
    return method


def _trip(cls: type, what: str) -> None:
    msg = f"{cls._label}: forbidden access {what}"
    LEAK_LOG.append(msg)
    raise LeakageError(msg)


# Every special method whose use would read, compare, convert, iterate, call or copy the object.
_DUNDERS = (
    "__str__ __format__ __bytes__ __bool__ __len__ __length_hint__ __iter__ __next__ __reversed__ __contains__ "
    "__getitem__ __setitem__ __delitem__ __missing__ __call__ __hash__ __eq__ __ne__ __lt__ __le__ __gt__ __ge__ "
    "__int__ __float__ __complex__ __index__ __round__ __trunc__ __floor__ __ceil__ __neg__ __pos__ __abs__ __invert__ "
    "__add__ __sub__ __mul__ __matmul__ __truediv__ __floordiv__ __mod__ __divmod__ __pow__ __lshift__ __rshift__ "
    "__and__ __xor__ __or__ __radd__ __rsub__ __rmul__ __rmatmul__ __rtruediv__ __rfloordiv__ __rmod__ __rdivmod__ "
    "__rpow__ __rlshift__ __rrshift__ __rand__ __rxor__ __ror__ __iadd__ __isub__ __imul__ __itruediv__ __ior__ "
    "__enter__ __exit__ __aenter__ __aexit__ __await__ __aiter__ __anext__ __dir__ __sizeof__ __copy__ "
    "__deepcopy__ __reduce__ __reduce_ex__ __getstate__ __setstate__ __getnewargs__ __getnewargs_ex__ "
    "__setattr__ __delattr__ __fspath__ __instancecheck__ __subclasscheck__"
).split()


class _Poison:
    __slots__ = ()
    _label = "Poison"

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":
            return type(self)
        _trip(type(self), f".{name}")

    def __repr__(self) -> str:
        return f"<{type(self)._label}>"


for _n in _DUNDERS:
    setattr(_Poison, _n, _raiser(_n))
del _n


class PoisonTask(_Poison):
    """Stands in for the episode's tau2 Task (initial_state, evaluation_criteria, user_scenario, ...)."""

    __slots__ = ()
    _label = "PoisonTask"


class PoisonDB(_Poison):
    """Stands in for the user device DB / user-side tool results."""

    __slots__ = ()
    _label = "PoisonDB"


@contextmanager
def poisoned(**names: type[_Poison]) -> Iterator[dict[str, _Poison]]:
    """Yield fresh poison objects, e.g. `with poisoned(task=PoisonTask, user_db=PoisonDB) as p: f(**p)`.

    Defaults to {task: PoisonTask, user_db: PoisonDB}. Fails on exit if any access was attempted (even if swallowed).
    """
    n = len(LEAK_LOG)
    objs = {k: v() for k, v in (names or {"task": PoisonTask, "user_db": PoisonDB}).items()}
    yield objs
    if len(LEAK_LOG) > n:
        raise LeakageError("poison touched (swallowed?): " + "; ".join(LEAK_LOG[n:]))


def assert_identical_under_poison(fn: Callable[..., Any], real: dict[str, Any], poison: dict[str, type[_Poison]]) -> Any:
    """Run fn(**real) and fn(**{real | poison objects}); assert equal results; return the result.

    `poison` maps kwarg name -> poison class replacing real[name]. Compare with ==, so return plain data.
    """
    ref = fn(**real)
    with poisoned(**poison) as p:
        got = fn(**{**real, **p})
    assert got == ref, f"output changed when {sorted(poison)} were poisoned: {got!r} != {ref!r}"
    return got
