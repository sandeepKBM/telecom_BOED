"""[telecom_boed] Model internals (H7): hidden-state capture of the Qwen3-8B agent at every decision point.

prompts.py : rebuild the agent's exact prompt (system + visible history + 13 tool schemas) per assistant message.
capture.py : tokenize, check token counts against the recorded vLLM usage, forward, store last-token states.
Import rule: neither module imports tau2 (or torch) at import time.
"""
