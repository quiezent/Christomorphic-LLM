# Scripts

This folder contains runnable Tinker sampling utilities for the archived public checkpoints. The script has been checked against installed `tinker==0.23.4`.

## Interactive Chat

[chat_qa.py](chat_qa.py) opens a Tinker sampling client and runs a multi-turn terminal chat.

```powershell
python -m pip install -r requirements.txt
$env:TINKER_API_KEY="your-api-key"
python script/chat_qa.py
```

Default checkpoint:

```text
gpt-r38-20b
```

Switch to V6R43:

```powershell
$env:CHECKPOINT_ALIAS="gpt-v6r43-120b"
python script/chat_qa.py
```

## Environment Variables

| Variable | Meaning |
|---|---|
| `TINKER_API_KEY` | Required Tinker API key |
| `CHECKPOINT_ALIAS` | `gpt-r38-20b` or `gpt-v6r43-120b` |
| `MODEL_PATH` | Override sampler or saved-weights URI |
| `BASE_MODEL` | Used when `MODEL_PATH` is blank |
| `SYSTEM_PROMPT` | Optional public system prompt |

## Boundary

This script is for sampling and research inspection. It is not a production chat service and does not include full safety orchestration, audit logging, rate limits, human escalation, or deployment controls.

The system prompt used by this convenience script is an inference scaffold. Output sampled with that prompt is not bare-prompt formation evidence. Set `SYSTEM_PROMPT` explicitly and report it when comparing results.

The archived checkpoints predate the current prospective causal standard. Sampling them can reproduce historical behavioral evidence, but cannot retroactively create fresh common starts, matched controls, immutable hosted-base identity, or hidden-state proof.
