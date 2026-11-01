"""Synthetic LEAKS: direct oracle import, task read, kwargs['task'], dynamic getattr, eval."""
import telecom_boed.oracle_hidden


def f(task, name, **kw):
    x = kw["task"]
    y = kw.get("user_db")
    z = getattr(kw, name)
    return str(task), eval("1")
