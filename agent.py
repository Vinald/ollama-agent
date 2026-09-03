#!/usr/bin/env python3
"""
A tiny agentic coding layer over Ollama — a minimal "Claude Code"-style loop.

Pieces:
  1. A system prompt that defines the tools and the exact call format.
  2. An agent loop: model emits one action -> we run it -> feed the result
     back -> repeat until the model calls `finish`.
  3. Tool implementations, path-sandboxed to a project root.

Memory:
  - Project memory : AGENT.md at the project root, read into the system prompt.
  - Session history: full transcript saved under .agent/history/, resumable.
  - Self notes     : a `run_note` tool appends to .agent/memory.md, which is
                     also read into the system prompt on the next run.

Action format is deliberately NOT "code inside JSON" — small models cannot
reliably escape quotes and newlines. Control goes in a one-line JSON object;
file/code payloads go in fenced blocks after it (heredoc style, no escaping).

Stdlib only. Python 3.9+.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-coder-7b-32k")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "40"))
MAX_CONSECUTIVE_PARSE_FAILS = 4
# Rough char budget (~4 chars/token) before we trim old tool output.
CONTEXT_CHAR_BUDGET = int(os.environ.get("AGENT_CONTEXT_CHARS", str(24_000 * 4)))

AGENT_DIR_NAME = ".agent"
PROJECT_MEMORY_FILE = "AGENT.md"
SELF_NOTES_FILE = "memory.md"


class AgentError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #


class Sandbox:
    """Confines all file paths to a single project root."""

    def __init__(self, root: pathlib.Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise AgentError(f"project root is not a directory: {self.root}")
        self.agent_dir = self.root / AGENT_DIR_NAME
        (self.agent_dir / "history").mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> pathlib.Path:
        rel = (rel or "").strip()
        if not rel or rel in (".", "./"):
            return self.root
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise AgentError(f"path escapes project root: {rel!r}")
        return p

    def display(self, p: pathlib.Path) -> str:
        try:
            return str(p.relative_to(self.root)) or "."
        except ValueError:
            return str(p)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

TOOLS_DOC = textwrap.dedent(
    """\
    read_file   — {"tool": "read_file", "args": {"path": "rel/path.py"}}
        Returns the file with line numbers.

    list_dir    — {"tool": "list_dir", "args": {"path": "."}}
        Lists entries in a directory.

    write_file  — {"tool": "write_file", "args": {"path": "rel/path.py"}}
        Then a fenced `content` block with the WHOLE new file:
        ```content
        <entire file text, verbatim>
        ```
        Creates or overwrites. Needs confirmation.

    edit_file   — {"tool": "edit_file", "args": {"path": "rel/path.py"}}
        Then a `search` block and a `replace` block:
        ```search
        <exact text that currently exists, unique in the file>
        ```
        ```replace
        <text to put in its place>
        ```
        Prefer this over write_file for small changes. Needs confirmation.

    run_bash    — {"tool": "run_bash", "args": {"command": "python -m pytest -q"}}
        Runs a shell command from the project root. Needs confirmation.

    run_note    — {"tool": "run_note", "args": {"note": "build uses make, not npm"}}
        Saves a durable note to memory for future runs.

    finish      — {"tool": "finish", "args": {"summary": "what you did"}}
        Call when the task is complete or you are blocked.
    """
)


class Tools:
    def __init__(self, sandbox: Sandbox, auto_approve: bool):
        self.sb = sandbox
        self.auto_approve = auto_approve

    # -- confirmation --------------------------------------------------- #

    def _confirm(self, action: str, detail: str) -> bool:
        if self.auto_approve:
            return True
        print(f"\n\033[33m! {action}\033[0m\n{textwrap.indent(detail, '  ')}")
        try:
            ans = input("  proceed? [y/N] ").strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")

    # -- dispatch ----------------------------------------------------- #

    def run(self, name: str, args: dict) -> str:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return f"ERROR: unknown tool {name!r}. Valid: {', '.join(self.names())}"
        try:
            return fn(args)
        except KeyError as e:
            return f"ERROR: missing required arg {e}"
        except AgentError as e:
            return f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001 — surface everything to the model
            return f"ERROR: {type(e).__name__}: {e}"

    @staticmethod
    def names() -> list[str]:
        return ["read_file", "list_dir", "write_file", "edit_file", "run_bash", "run_note", "finish"]

    # -- implementations ------------------------------------------- #

    def _t_read_file(self, args: dict) -> str:
        p = self.sb.resolve(args["path"])
        if not p.is_file():
            raise AgentError(f"no such file: {self.sb.display(p)}")
        text = p.read_text(errors="replace")
        all_lines = text.splitlines()
        lines = all_lines[:800]
        width = len(str(len(lines)))
        body = "\n".join(f"{i + 1:>{width}}  {ln}" for i, ln in enumerate(lines))
        if len(all_lines) > 800:
            body += f"\n... ({len(all_lines) - 800} more lines)"
        return body

    def _t_list_dir(self, args: dict) -> str:
        p = self.sb.resolve(args.get("path", "."))
        if not p.is_dir():
            raise AgentError(f"not a directory: {self.sb.display(p)}")
        rows = []
        for entry in sorted(p.iterdir()):
            if entry.name == AGENT_DIR_NAME:
                continue
            rows.append(f"{'dir ' if entry.is_dir() else 'file'}  {self.sb.display(entry)}")
        return "\n".join(rows) or "(empty)"

    def _t_write_file(self, args: dict) -> str:
        p = self.sb.resolve(args["path"])
        if "content" not in args:
            raise AgentError("write_file needs a fenced ```content block after the JSON")
        content = args["content"]
        verb = "overwrite" if p.is_file() else "create"
        preview = content if len(content) < 1200 else content[:1200] + "\n... (truncated)"
        if not self._confirm(f"{verb} {self.sb.display(p)}  ({len(content)} bytes)", preview):
            return "SKIPPED: user declined the write"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: wrote {self.sb.display(p)} ({len(content)} bytes)"

    def _t_edit_file(self, args: dict) -> str:
        p = self.sb.resolve(args["path"])
        if not p.is_file():
            raise AgentError(f"no such file: {self.sb.display(p)}")
        if "old" not in args or "new" not in args:
            raise AgentError("edit_file needs a ```search block and a ```replace block after the JSON")
        old, new = args["old"], args["new"]
        text = p.read_text()
        count = text.count(old)
        if count == 0:
            raise AgentError("search text not found (must match the file exactly, incl. indentation)")
        if count > 1:
            raise AgentError(f"search text is not unique ({count} matches); include more surrounding lines")
        diff = "\n".join(
            [f"\033[31m- {l}\033[0m" for l in old.splitlines()]
            + [f"\033[32m+ {l}\033[0m" for l in new.splitlines()]
        )
        if not self._confirm(f"edit {self.sb.display(p)}", diff):
            return "SKIPPED: user declined the edit"
        p.write_text(text.replace(old, new, 1))
        return f"OK: edited {self.sb.display(p)}"

    def _t_run_bash(self, args: dict) -> str:
        cmd = args["command"]
        if not self._confirm(f"run: {cmd}", f"cwd: {self.sb.root}"):
            return "SKIPPED: user declined the command"
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.sb.root,
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 600s"
        out = ((proc.stdout or "") + (proc.stderr or "")).strip() or "(no output)"
        if len(out) > 6000:
            out = out[:6000] + "\n... (output truncated)"
        return f"exit {proc.returncode}\n{out}"

    def _t_run_note(self, args: dict) -> str:
        note = args["note"].strip()
        if not note:
            return "ERROR: empty note"
        f = self.sb.agent_dir / SELF_NOTES_FILE
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        with f.open("a") as fh:
            fh.write(f"- ({stamp}) {note}\n")
        return f"OK: saved note to {AGENT_DIR_NAME}/{SELF_NOTES_FILE}"

    def _t_finish(self, args: dict) -> str:
        return "__FINISH__"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM_TEMPLATE = textwrap.dedent(
    """\
    You are a coding agent working inside the project at {root}.
    You make progress by taking ONE action per reply.

    ACTION FORMAT — every reply is a single JSON object on its own line:

        {{"tool": "<name>", "args": {{...}}}}

    NEVER put file contents or code inside the JSON. For write_file and
    edit_file, args holds only the path, and the code goes in fenced blocks
    right after the JSON (no escaping needed inside a fenced block):

        {{"tool": "write_file", "args": {{"path": "hello.py"}}}}
        ```content
        print("hello")
        ```

        {{"tool": "edit_file", "args": {{"path": "hello.py"}}}}
        ```search
        print("hello")
        ```
        ```replace
        print("hello world")
        ```

    Optionally include a short "thought" string in the JSON. No other prose.

    TOOLS:
    {tools}

    RULES:
    - One action per reply, then wait for the result.
    - read_file before you edit_file. Use exact text for the search block.
    - edit_file WITHOUT both a ```search and a ```replace block will fail. If you
      cannot form a unique search block, use write_file with the whole file.
    - After editing code, run the tests / relevant command with run_bash.
    - Do not ask the user questions; keep working until done, then call finish.

    WORKED EXAMPLE (user: "make greet() use the name arg"):

    reply 1:
    {{"tool": "read_file", "args": {{"path": "greet.py"}}}}

    reply 2 (after seeing the file):
    {{"tool": "edit_file", "args": {{"path": "greet.py"}}}}
    ```search
    def greet(name):
        print("hello")
    ```
    ```replace
    def greet(name):
        print(f"hello {{name}}")
    ```

    reply 3 (after "OK: edited greet.py"):
    {{"tool": "run_bash", "args": {{"command": "python -m pytest -q"}}}}

    reply 4 (after tests pass):
    {{"tool": "finish", "args": {{"summary": "greet() now uses the name arg; tests pass"}}}}

    {memory}
    """
)


def build_system_prompt(sb: Sandbox) -> str:
    parts = []
    proj = sb.root / PROJECT_MEMORY_FILE
    if proj.is_file():
        parts.append(f"PROJECT MEMORY ({PROJECT_MEMORY_FILE}):\n{proj.read_text().strip()}")
    notes = sb.agent_dir / SELF_NOTES_FILE
    if notes.is_file() and notes.stat().st_size:
        parts.append(
            f"NOTES FROM PAST RUNS ({AGENT_DIR_NAME}/{SELF_NOTES_FILE}):\n{notes.read_text().strip()}"
        )
    memory = "\n\n".join(parts) if parts else "(no saved memory yet)"
    return SYSTEM_TEMPLATE.format(root=sb.root, tools=TOOLS_DOC, memory=memory)


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #


def ollama_chat(model: str, messages: list[dict], temperature: float = 0.0) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "stream": False,
         "options": {"temperature": temperature}}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise AgentError(f"cannot reach Ollama at {OLLAMA_URL}: {e}") from e
    if "error" in data:
        raise AgentError(f"Ollama error: {data['error']}")
    return data.get("message", {}).get("content", "")


# --------------------------------------------------------------------------- #
# Action parsing
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```([a-zA-Z_]*)[^\S\n]*\n(.*?)\n?```", re.DOTALL)
_KNOWN_TOOLS = {"read_file", "list_dir", "write_file", "edit_file", "run_bash", "run_note", "finish"}


def _balanced_json(text: str) -> dict | None:
    """First balanced {...} in `text` that parses as a JSON object."""
    for start in (i for i, c in enumerate(text) if c == "{"):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
    return None


def parse_action(text: str) -> dict:
    """Return {"tool", "args", "thought"} from a model reply.

    Accepts several shapes small models drift into:
      - a bare JSON object   {"tool": "read_file", "args": {"path": "x"}}
      - a ```tool fenced block whose first line is the tool name, followed by
        an args JSON object or a bare path
    File/code payloads are read from ```content / ```search / ```replace blocks.
    Raises AgentError with a message the model can recover from.
    """
    fences = {tag.lower(): body for tag, body in _FENCE_RE.findall(text)}

    name = None
    args: dict = {}
    thought = None

    # Shape 1: ```tool block
    if "tool" in fences:
        first, _, rest = fences["tool"].strip().partition("\n")
        name = first.strip().strip('"').strip("`")
        rest = rest.strip()
        if rest:
            parsed = _balanced_json(rest)
            if parsed is not None:
                args = parsed.get("args") if isinstance(parsed.get("args"), dict) else parsed
            elif "\n" not in rest:
                args = {"path": rest.strip().strip('"'), "command": rest.strip(), "note": rest.strip()}

    # Shape 2: bare / fenced JSON object
    if name is None:
        masked = _FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        obj = _balanced_json(masked) or _balanced_json(fences.get("json", ""))
        if obj is None:
            raise AgentError(
                'no action found. Reply with ONE line: {"tool": "<name>", "args": {...}}'
            )
        name = obj.get("tool") or obj.get("name") or obj.get("action")
        raw_args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
        args = raw_args if isinstance(raw_args, dict) else {}
        thought = obj.get("thought")

    if not name:
        raise AgentError('missing "tool" name')
    if name not in _KNOWN_TOOLS:
        raise AgentError(f"unknown tool {name!r}. Valid: {', '.join(sorted(_KNOWN_TOOLS))}")

    if name == "write_file" and "content" not in args and "content" in fences:
        args["content"] = fences["content"]
    if name == "edit_file":
        args.setdefault("old", fences["search"]) if "search" in fences else None
        args.setdefault("new", fences["replace"]) if "replace" in fences else None

    # Drop the catch-all keys we speculatively set for bare-path tool blocks.
    if name in ("read_file", "list_dir", "write_file", "edit_file"):
        args.pop("command", None)
        args.pop("note", None)
    if name != "run_bash":
        args.pop("command", None)
    if name != "run_note":
        args.pop("note", None)

    return {"tool": name, "args": args, "thought": thought}


# --------------------------------------------------------------------------- #
# Context trimming
# --------------------------------------------------------------------------- #


def trim_context(messages: list[dict]) -> None:
    while sum(len(m["content"]) for m in messages) > CONTEXT_CHAR_BUDGET:
        for i in range(2, len(messages) - 4):
            if messages[i]["role"] == "tool" and not messages[i]["content"].startswith("[trimmed]"):
                messages[i]["content"] = "[trimmed] earlier tool output removed to save context"
                break
        else:
            return


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def save_history(sb: Sandbox, messages: list[dict], path: pathlib.Path | None) -> pathlib.Path:
    if path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = sb.agent_dir / "history" / f"{stamp}.json"
    path.write_text(json.dumps(messages, indent=2))
    return path


def latest_history(sb: Sandbox) -> pathlib.Path | None:
    files = sorted((sb.agent_dir / "history").glob("*.json"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #


def run_agent(task, sb, model, tools, messages=None, history_path=None):
    if messages is None:
        messages = [{"role": "system", "content": build_system_prompt(sb)}]
    messages.append({"role": "user", "content": task})

    fails = 0
    last_result = None
    stall = 0
    for step in range(1, MAX_STEPS + 1):
        trim_context(messages)
        reply = ollama_chat(model, messages)
        messages.append({"role": "assistant", "content": reply})

        try:
            action = parse_action(reply)
            fails = 0
        except AgentError as e:
            fails += 1
            print(f"\033[31m[{step}] unparseable reply ({fails}/{MAX_CONSECUTIVE_PARSE_FAILS})\033[0m")
            if fails >= MAX_CONSECUTIVE_PARSE_FAILS:
                print("\033[31m✗ model cannot produce a valid action — stopping\033[0m")
                print(textwrap.indent(_clip(reply, 800), "    "))
                break
            messages.append({"role": "tool", "content": f"ERROR: {e}"})
            continue

        name, args = action["tool"], action["args"]
        if action.get("thought"):
            print(f"\033[90m[{step}] {action['thought']}\033[0m")
        print(f"\033[36m[{step}] {name}({_fmt_args(args)})\033[0m")

        if name == "finish":
            print(f"\n\033[32m✓ {args.get('summary', '(no summary)')}\033[0m")
            return save_history(sb, messages, history_path)

        result = tools.run(name, args)
        print(textwrap.indent(_clip(result, 1500), "    "))

        if result == last_result and result.startswith(("ERROR", "SKIPPED")):
            stall += 1
        else:
            stall = 0
        last_result = result
        if stall == 2:
            result += (
                "\n\nYou have repeated this failing action. Change approach: if "
                "edit_file keeps failing, call write_file with the ENTIRE file in a "
                "```content block instead. If you are blocked, call finish."
            )
        elif stall >= 3:
            print("\n\033[31m✗ stalled on the same failing action — stopping\033[0m")
            messages.append({"role": "tool", "content": result})
            break

        messages.append({"role": "tool", "content": result})
        save_history(sb, messages, history_path)

    else:
        print("\n\033[31m✗ hit step limit without finishing\033[0m")
    return save_history(sb, messages, history_path)


def _fmt_args(args: dict) -> str:
    out = []
    for k, v in args.items():
        s = str(v).replace("\n", "\\n")
        out.append(f"{k}={s[:50]}" + ("…" if len(s) > 50 else ""))
    return ", ".join(out)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"\n... ({len(s) - n} more chars)"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="Tiny agentic coding layer over Ollama.")
    ap.add_argument("task", nargs="*", help="the task; omit for an interactive REPL")
    ap.add_argument("-C", "--root", default=".", help="project root (default: cwd)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    ap.add_argument("--yolo", action="store_true", help="skip confirmation for writes and commands")
    ap.add_argument("--continue", dest="cont", action="store_true", help="resume the latest session")
    ap.add_argument("--resume", metavar="FILE", help="resume a specific history JSON file")
    args = ap.parse_args()

    try:
        sb = Sandbox(pathlib.Path(args.root))
    except AgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    tools = Tools(sb, auto_approve=args.yolo)

    messages = None
    history_path = None
    resume = args.resume or (str(latest_history(sb)) if args.cont and latest_history(sb) else None)
    if resume:
        history_path = pathlib.Path(resume)
        messages = json.loads(history_path.read_text())
        print(f"\033[90mresumed {history_path.name} ({len(messages)} messages)\033[0m")

    print(f"\033[90mroot={sb.root}  model={args.model}  "
          f"{'YOLO (no confirms)' if args.yolo else 'confirm on writes/commands'}\033[0m")

    if args.task:
        run_agent(" ".join(args.task), sb, args.model, tools, messages, history_path)
        return 0

    print("interactive mode — type a task, or 'exit'. Tasks continue the same session.")
    while True:
        try:
            task = input("\n\033[1magent>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if task in ("exit", "quit", ":q", ""):
            break
        if messages is None:
            messages = [{"role": "system", "content": build_system_prompt(sb)}]
            history_path = None
        history_path = run_agent(task, sb, args.model, tools, messages, history_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
