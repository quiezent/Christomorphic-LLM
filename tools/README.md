# Repository Tools

`validate_repository.py` performs the public repository's offline integrity checks:

- local Markdown links resolve;
- JSON files parse;
- public prompt counts and IDs remain valid;
- Python files compile.

Run it from the repository root:

```powershell
python tools/validate_repository.py
```

It makes no Tinker, model, or network calls.
