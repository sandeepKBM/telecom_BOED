"""Synthetic LEAK: reads task.initial_state."""


def make_agent(task, **kwargs):
    return task.initial_state
