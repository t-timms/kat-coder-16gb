"""Preflight the agent's model path through LITELLM, exactly as the agent uses it.

The first pilot burned its entire run on `400 Bad Request` for every call while a
raw-curl preflight passed cleanly. The bug lived in the layer curl skipped:
mini-swe-agent's default model class (models/litellm_model.py:69) always sends
`tools=[BASH_TOOL]` with tool_choice auto, and the server had no tool-call parser.

Hence this preflight sends the SAME bash tool the agent sends, through litellm,
with the same registry. Checks, in the order of what silently ruins a run:
  1. the call succeeds at all
  2. tool_calls come back PARSED (route A viability - does qwen3_xml understand
     this model's <tool_call><function=..><parameter=..> dialect?)
  3. content is not None and the reasoning trace has not leaked into it
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "LITELLM_MODEL_REGISTRY_PATH", str(Path.home() / "kat_swebench/registry.json")
)

import litellm  # noqa: E402

litellm.suppress_debug_info = True

# Mirrors minisweagent.models.litellm_model.BASH_TOOL
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run"}},
            "required": ["command"],
        },
    },
}

MODEL = "hosted_vllm/kat-16gb"
KWARGS = {
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 3072,
    "drop_params": True,
}

MESSAGES = [
    {"role": "system", "content": "You interact with a computer shell to solve tasks."},
    {"role": "user", "content": "List the files in the current directory. Use the bash tool."},
]

print("registry :", os.environ["LITELLM_MODEL_REGISTRY_PATH"])

try:
    resp = litellm.completion(
        model=MODEL, messages=MESSAGES, tools=[BASH_TOOL], tool_choice="auto", **KWARGS
    )
except Exception as e:  # noqa: BLE001
    print(f"!! litellm call FAILED: {type(e).__name__}")
    print("   " + str(e)[:600])
    sys.exit(1)

choice = resp.choices[0]  # type: ignore[union-attr]
msg = choice.message
content = msg.content
reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
tool_calls = getattr(msg, "tool_calls", None) or []

print("finish_reason :", choice.finish_reason)
print("reasoning     :", f"{len(reasoning)} chars" if reasoning else "none")
print("content       :", "NULL" if content is None else f"{len(content)} chars")
print("tool_calls    :", len(tool_calls))

fail = False

if tool_calls:
    for tc in tool_calls:
        fn = tc.function
        print(f"  -> {fn.name}({fn.arguments!r})")
        try:
            args = json.loads(fn.arguments)
        except (TypeError, json.JSONDecodeError):
            print("     !! arguments are not valid JSON; the agent cannot use this")
            fail = True
            continue
        if "command" not in args:
            print("     !! no 'command' key; the agent expects one")
            fail = True
    print("\nROUTE A VIABLE: qwen3_xml parsed this model's tool calls." if not fail
          else "\nROUTE A BROKEN: tool calls came back malformed.")
else:
    print("\nROUTE A NOT VIABLE: no tool_calls parsed out of the response.")
    print("Use ROUTE B instead: --model-class litellm_textbased with")
    print("swebench_backticks.yaml, which needs no server-side tool parser.")
    if content:
        print("\n--- raw content (is the XML sitting here unparsed? ---")
        print(content[:600])
    fail = True

if content and ("<think>" in content or "</think>" in content):
    print("!! reasoning trace leaked into content; parsing will break")
    fail = True

if choice.finish_reason == "length":
    print("!! finish_reason=length: truncated before completing. Budget too small.")
    fail = True

sys.exit(1 if fail else 0)
