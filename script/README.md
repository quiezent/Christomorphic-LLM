# Scripts

This folder contains runnable Tinker sampling utilities.

## Interactive Chat

[chat_qa.py](chat_qa.py) opens a Tinker sampling client and runs a multi-turn terminal chat.

```powershell
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

This script is for sampling and research inspection. It is not a production chat service and does not include full safety orchestration, logging, or deployment controls.
