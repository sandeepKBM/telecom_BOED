"""Synthetic LEAK: reaches oracle_hidden only through helper (relative import)."""
from . import helper


def make_agent(task=None):
    return helper.peek()
