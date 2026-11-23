"""Talk to the trained models yourself, base and adapters side by side (Clayton, 12 Nov 2026: "how can I infer these
models that you have trained to test them").

    .venv\\Scripts\\python.exe -B chat.py                                    # Ministral, base then links, for each prompt
    .venv\\Scripts\\python.exe -B chat.py --key ministral3_8b --arms base links wrong tinystories
    .venv\\Scripts\\python.exe -B chat.py --raw                              # plain text, as training read it: give a verse, see what follows
    .venv\\Scripts\\python.exe -B chat.py --prompt "What does Isaiah 11:2 say rests on the Branch?"   # one prompt, then exit
    .venv\\Scripts\\python.exe -B chat.py --file my_prompts.txt              # one prompt per line (or .jsonl with "prompt"), then exit
    .venv\\Scripts\\python.exe -B chat.py --file my_prompts.txt --conversation --draws 3   # the prompts as one conversation, three times

Models (--key): ministral3_8b, olmo3_7b, gemma4_e4b. Arms (--arms): base, links, links_s2, wrong, wrong_s2, scrambled,
scrambled_s2, tinystories, whichever the model has (runs/train/<key>_<arm>/adapter). The base model is loaded once, in 4-bit
as in training, and each adapter is switched in for its turn.

In the conversation, type a prompt over one or more lines and send it with an empty line. Commands, each on its own line:
    :q  quit         :reset  forget the conversation          :raw / :chat  plain text or chat turns
    :arms base links  which arms answer                        :t 0.7  temperature     :n 600  most new tokens

Sampling is as for the replies already read: temperature 0.7, top_p 1, no top-k, no repetition penalty. Each prompt gets
one seed, the same for every arm, so differences between arms come from the adapters, not the dice. Chat turns keep
each arm's own conversation until :reset. Every exchange is saved to runs/chat/<key>_<date>.jsonl.

With --file or --prompt the arms run one after another, with one adapter on the card at a time, so any number of arms fits.
Each prompt is a fresh conversation unless --conversation makes them one, as if typed in turn. --draws N repeats it all
with N seeds; draw 0 has the seed an interactive session would use, so it repeats what was seen there.

Run nothing else on the card at the same time: the 8 GB card holds one model.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

from measure_base import in_venv, load_model

HERE = Path(__file__).parent
TRAIN = HERE / "runs" / "train"
OUT = HERE / "runs" / "chat"
MODELS = {"olmo3_7b": ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
          "ministral3_8b": ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "f6fae9795746f63c9be8344932f01275f3c63734"),
          "gemma4_e4b": ("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c")}
ARMS = ("base", "links", "links_s2", "wrong", "wrong_s2", "scrambled", "scrambled_s2", "tinystories",
        "pairs_links", "pairs_links_s2", "pairs_scrambled", "pairs_scrambled_s2", "pairs_tinystories",  # round 2 (14 Nov 2026)
        "tongues_true", "tongues_true_s2", "tongues_false", "tongues_false_s2",  # form B at the word (tongues.py, 21 Nov 2026)
        "tongues_true_w", "tongues_true_w_s2", "tongues_false_w", "tongues_false_w_s2")  # the same with the loss on the exact words beside it


def adapter(key, arm):
    return TRAIN / f"{key}_{arm}" / "adapter"


def load(key, arms, wanted=None):
    """The base model once, with every requested adapter attached under its arm's name. wanted: all the arms that will
    answer, checked here so a missing one stops the run before anything is generated."""
    missing = [a for a in (wanted or arms) if a != "base" and not (adapter(key, a) / "adapter_config.json").exists()]
    if missing:
        have = [a for a in ARMS if a == "base" or (adapter(key, a) / "adapter_config.json").exists()]
        raise SystemExit(f"{key} has no adapter for {', '.join(missing)}; it has: {', '.join(have)}")
    model_id, revision = MODELS[key]
    paths = {a: adapter(key, a) for a in arms if a != "base"}
    first = next(iter(paths), None)
    model, tok = load_model(model_id, revision, True, str(paths[first]) if first else None, cpu_words=True)  # 1 GB more room for long conversations; the forward pass is bit-identical (11 Nov)
    names = {}
    if first:  # PEFT names the first adapter "default"
        names[first] = "default"
        for a, p in paths.items():
            if a != first:
                model.load_adapter(str(p), adapter_name=a)
                names[a] = a
    return model, tok, names


def swap(model, key, names, arm):
    """Put arm's adapter on the card in place of the one there now (the file runs keep one adapter loaded)."""
    if arm == "base" or arm in names:
        return
    import torch
    model.load_adapter(str(adapter(key, arm)), adapter_name=arm)
    model.set_adapter(arm)
    for a in list(names):
        model.delete_adapter(names.pop(a))
    names[arm] = arm
    torch.cuda.empty_cache()


def strength(model, scale):
    """Every loaded adapter at this strength (1: as trained), set outright so that it cannot compound."""
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict):
            for k in m.scaling:
                m.scaling[k] = scale * m.lora_alpha[k] / m.r[k]


def seed_for(key, turn, prompt, draw=0):
    """One seed per prompt and turn, the same for every arm; draw 0 is the one an interactive session uses."""
    return zlib.crc32(f"{key}:{turn}:{prompt}{f':{draw}' if draw else ''}".encode())


def encode(tok, history, prompt, raw):
    if raw:
        return tok(prompt, add_special_tokens=True, return_tensors="pt")
    msgs = history + [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except Exception:  # Gemma's template wants content parts
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in msgs]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)


def reply(model, tok, names, arm, enc, seed, temperature, max_tokens, stream=True):
    import contextlib

    import torch
    from transformers import TextStreamer
    if arm == "base":
        ctx = model.disable_adapter() if names else contextlib.nullcontext()
    else:
        model.set_adapter(names[arm])
        ctx = contextlib.nullcontext()
    ids, mask = enc["input_ids"].to(0), enc["attention_mask"].to(0)
    torch.manual_seed(seed)
    t0 = time.time()
    with ctx, torch.inference_mode():
        seq = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                             repetition_penalty=1.0, max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
                             streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None)
    new = seq[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), len(new), round(time.time() - t0, 1)


def main():
    in_venv()
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="ministral3_8b", choices=MODELS)
    p.add_argument("--arms", nargs="+", default=["base", "links"], choices=ARMS)
    p.add_argument("--raw", action="store_true", help="plain text, no chat template")
    p.add_argument("--prompt", help="one prompt, then exit")
    p.add_argument("--file", help="prompts, one per line (.txt) or with a 'prompt' field (.jsonl), then exit")
    p.add_argument("--conversation", action="store_true", help="with --file: the prompts are one conversation, as if typed in turn")
    p.add_argument("--draws", type=int, default=1, help="with --file or --prompt: run it all this many times, each with its own seeds")
    p.add_argument("--first-draw", type=int, default=0, help="number the draws from here: 4 with --draws 4 gives draws 4 to 7, on seeds no earlier run used")
    p.add_argument("--log", help="where to save the exchanges (default runs/chat/<key>_<date>.jsonl)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--scale", type=float, default=1.0, help="the adapters' strength: 1 as trained, 0.25 a quarter (base is not changed)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from transformers.utils import logging
    logging.set_verbosity_error()  # no max_length notice before every reply
    batch = bool(args.prompt or args.file)
    first = [next((a for a in args.arms if a != "base"), "base")] if batch else args.arms  # a file run swaps adapters in one at a time
    print(f"Loading {args.key} with {', '.join(args.arms)} ...", flush=True)
    model, tok, names = load(args.key, first, args.arms)
    if args.scale != 1:
        strength(model, args.scale)
    log = Path(args.log) if args.log else OUT / f"{args.key}_{time.strftime('%Y%m%d')}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    state = {"raw": args.raw, "arms": list(args.arms), "t": args.temperature, "n": args.max_tokens}
    history = {a: [] for a in ARMS}

    def answer(arm, prompt, hist, turn, seed, draw=0):
        print(f"\n===== {arm} =====" + (f" (draw {draw})" if args.draws > 1 else ""), flush=True)
        enc = encode(tok, [] if state["raw"] else hist, prompt, state["raw"])
        text, n, secs = reply(model, tok, names, arm, enc, seed, state["t"], state["n"])
        if n >= state["n"]:
            print(f"[cut off at {state['n']} tokens]", flush=True)
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "key": args.key, "arm": arm, "mode": "raw" if state["raw"] else "chat",
                                "turn": turn, "draw": draw, "prompt": prompt, "reply": text, "tokens": n, "cut_off": n >= state["n"],
                                "seconds": secs, "temperature": state["t"], "seed": seed, **({"scale": args.scale} if args.scale != 1 and arm != "base" else {})}, ensure_ascii=False) + "\n")
        return text

    if batch:
        if args.prompt:
            prompts = [args.prompt]
        elif args.file.endswith(".jsonl"):
            prompts = [json.loads(l)["prompt"] for l in open(args.file, encoding="utf-8") if l.strip()]
        else:
            prompts = [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
        for arm in args.arms:
            swap(model, args.key, names, arm)
            if args.scale != 1:
                strength(model, args.scale)
            for d in range(args.first_draw, args.first_draw + args.draws):
                hist = []
                for q in prompts:
                    turn = len(hist) // 2
                    print(f"\n>>> {q}", flush=True)
                    text = answer(arm, q, hist, turn, seed_for(args.key, turn, q, d), d)
                    if args.conversation and not state["raw"]:
                        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
        print(f"\n(saved to {log})", flush=True)
        return
    print("Type a prompt and send it with an empty line. :q quits, :reset forgets, :raw / :chat switch, :arms ..., :t ..., :n ...")
    while True:
        lines = []
        try:
            while True:
                line = input("> " if not lines else "  ")
                if not lines and line.startswith(":"):  # a command takes effect at once
                    lines.append(line)
                    break
                if not line.strip():
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        q = "\n".join(lines).strip()
        if not q:
            continue
        if q == ":q":
            return
        if q == ":reset":
            history = {a: [] for a in ARMS}
            print("(conversation forgotten)")
            continue
        if q in (":raw", ":chat"):
            state["raw"] = q == ":raw"
            print(f"({'plain text' if state['raw'] else 'chat turns'})")
            continue
        if q.startswith(":arms"):
            want = q.split()[1:]
            bad = [a for a in want if a != "base" and a not in names]
            if bad or not want:
                print(f"(loaded: {', '.join(['base'] + list(names))}; restart with --arms to load others)")
            else:
                state["arms"] = want
            continue
        if q.startswith(":t "):
            state["t"] = float(q.split()[1])
            continue
        if q.startswith(":n "):
            state["n"] = int(q.split()[1])
            continue
        turn = len(history[state["arms"][0]]) // 2
        seed = seed_for(args.key, turn, q)
        for arm in state["arms"]:
            text = answer(arm, q, history[arm], turn, seed)
            if not state["raw"]:
                history[arm] += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
        print(f"\n(saved to {log.relative_to(HERE) if log.is_relative_to(HERE) else log})", flush=True)


if __name__ == "__main__":
    main()
