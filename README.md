# ollama-agent

A tiny agentic coding layer over [Ollama](https://ollama.com) — a minimal
"Claude Code"-style loop you run in your terminal, pointed at a local model.

It is one file, stdlib only, ~450 lines. Not a Claude replacement — a small,
hackable base you can read in full and tune for your model's quirks.

## What it does

- **Terminal tool.** One-shot (`agent.py "task"`) or an interactive REPL.
- **Local model.** Talks to Ollama at `localhost:11434`. Fully offline.
- **File access, sandboxed.** Reads/writes/lists only inside one project root.
  `run_bash` runs from that root. Writes and commands need confirmation
  (`--yolo` to skip).
- **Memory, three ways:**
  - *Project memory* — `AGENT.md` at the project root, loaded into the prompt.
  - *Session history* — every run saved to `.agent/history/*.json`, resumable
    with `--continue` / `--resume`.
  - *Self notes* — the `run_note` tool appends to `.agent/memory.md`, which is
    loaded into the prompt on later runs.

## Tools the model can call

`read_file`, `list_dir`, `write_file`, `edit_file` (exact search/replace),
`run_bash`, `run_note`, `finish`.

Actions are a one-line JSON object; file/code payloads go in fenced blocks
after it (heredoc style) so the model never has to escape quotes or newlines
inside JSON — the main failure mode for small models.

## Setup

```bash
# a model that is decent at instruction-following; bump its context window
cat > Modelfile <<'EOF'
FROM qwen2.5-coder:7b
PARAMETER num_ctx 32768
EOF
ollama create qwen2.5-coder-7b-32k -f Modelfile
```

No Python dependencies.

### Install as a terminal command

```bash
ln -sf "$PWD/agent.py" ~/.local/bin/ollama-agent
ln -sf "$PWD/agent.py" ~/.local/bin/oa   # short alias
```

`~/.local/bin` must be on your `PATH`. Then from any project directory:

```bash
oa --yolo "add a docstring to slugify in text.py"
```

The loop is stateless between calls; Ollama keeps the model warm on its own
(`OLLAMA_KEEP_ALIVE`), so repeated calls stay fast without a resident daemon.

## Usage

```bash
# one-shot, from inside your project
./agent.py -C . "add a docstring to every function in utils.py and run pytest"

# interactive; tasks continue the same session
./agent.py -C .

# skip confirmation prompts
./agent.py -C . --yolo "fix the failing test in test_parser.py"

# resume where you left off
./agent.py -C . --continue "now also handle the empty-input case"

# pick a stronger model for trickier work
./agent.py -C . -m qwen2.5-coder-32b-16k "refactor the config loader"
```

### Options

| flag | meaning |
|------|---------|
| `-C, --root DIR` | project root the agent is confined to (default: cwd) |
| `-m, --model NAME` | Ollama model (default: `qwen2.5-coder-7b-32k`, or `$AGENT_MODEL`) |
| `--yolo` | don't ask before writes / shell commands |
| `--continue` | resume the most recent session in this project |
| `--resume FILE` | resume a specific `.agent/history/*.json` |

### Environment

| var | default |
|-----|---------|
| `OLLAMA_URL` | `http://localhost:11434` |
| `AGENT_MODEL` | `qwen2.5-coder-7b-32k` |
| `AGENT_MAX_STEPS` | `40` |
| `AGENT_CONTEXT_CHARS` | `96000` (~24k tokens before old tool output is trimmed) |

## `AGENT.md` example

```markdown
# Project memory
- Python library. Run tests with: python -m pytest -q
- Style: every public function needs a one-line docstring.
- Do not touch anything under vendor/.
```

## Limits

- A 7B model will skip steps, occasionally hallucinate that it did something,
  and need more retries than a hosted model. `qwen2.5-coder:32b` is markedly
  better at multi-step tool adherence if you can spare the RAM.
- No streaming, no parallel tool calls, no git integration.
- `run_bash` is only sandboxed by working directory — it runs with your full
  user permissions. Keep confirmation on for anything you don't trust.

## License

MIT
