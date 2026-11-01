"""Synthetic LEAK: touches .user_db and a getattr string."""


def act(env):
    a = env.user_db
    b = getattr(env, "evaluation_criteria")
    return a, b
