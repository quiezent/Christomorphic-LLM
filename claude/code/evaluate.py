"""Sample models on Clayton's probes (or GSM8K) and write responses plus a blinded reading packet.

Every model gets the same prompt: one user turn, rendered either "bare" (no system message, the
format R38 and v6r43 were trained on) or "native" (the standard Harmony system message with
"Reasoning: medium" and a fixed date). Each model writes its own analysis channel; sampling defaults to
OpenAI's recommended temperature 1.0 and top_p 1.0 with a generous token cap. Packets show only
the final channel, with labels shuffled per probe and the key in a separate file.

    python evaluate.py --out runs/audit --set full --format bare --format native --samples 2 \
        --models base20,r38,base120,v6r43
    python evaluate.py --out runs/x/eval --models base20,r38 --model cand=tinker://.../sampler_weights/final
"""
import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

PROJECT_ID = "dab755b5-88e7-4876-9829-e9ecc1063e39"
KNOWN = {
    "base20": "base:openai/gpt-oss-20b",
    "r38": "tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final",
    "base120": "base:openai/gpt-oss-120b",
    "v6r43": "tinker://8ad467bc-72eb-51c2-bbe3-417bf8940b43:train:0/sampler_weights/final",
}
HERE = Path(__file__).parent
FINAL_MARK = "<|channel|>final<|message|>"
DATE = "2026-10-13"


def load_set(name):
    """Return [(id, prompt, gold)] for 'subset', 'full' or 'gsm8k:N'."""
    if name.startswith("gsm8k"):
        n = int(name.split(":")[1])
        rows = [json.loads(l) for l in open(HERE / "bench" / "gsm8k" / "test.jsonl", encoding="utf-8")][:n]
        return [(f"gsm8k_{i:04d}", r["question"], r["answer"].split("####")[-1].strip().replace(",", ""))
                for i, r in enumerate(rows)]
    spec = json.loads((HERE / "eval_probes.json").read_text(encoding="utf-8"))
    suite = json.loads(Path(spec["suite"]).read_text(encoding="utf-8"))
    by_id = {p["prompt_id"]: p["prompt"] for c in suite["categories"] for p in c["prompts"]}
    ids = spec["ids"] if name == "subset" else list(by_id)
    return [(pid, by_id[pid], None) for pid in ids]


def make_renderer(fmt, tokenizer):
    if fmt == "bare":
        return renderers.get_renderer("gpt_oss_no_sysprompt", tokenizer)
    return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium", current_date=DATE)


def split_channels(renderer, tokens):
    """Return (analysis, final, complete). A reply cut off by the token cap keeps any partial final."""
    msg, termination = renderer.parse_response(tokens)
    content = msg["content"]
    if isinstance(content, list) and termination == renderers.ParseTermination.STOP_SEQUENCE:
        analysis = "\n".join(p["thinking"] for p in content if p["type"] == "thinking")
        return analysis, renderers.get_text_content(msg), True
    head, _, tail = renderer.tokenizer.decode(tokens).partition(FINAL_MARK)
    return head, tail, False


def numbers(text):
    return [n.replace(",", "").rstrip(".") for n in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def is_correct(final, gold):
    """Boxed answer if there is one; otherwise the last bold number, the first sentence's last number
    or the reply's last number may match. The same rule applies to every model."""
    t = final.replace("{,}", ",").replace("\\$", "").replace("$", "")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", t)
    if boxed:
        cands = numbers(boxed[-1])[-1:]
    else:
        bold = [n for span in re.findall(r"\*\*([^*]+)\*\*", t) for n in numbers(span)]
        first = numbers(re.split(r"(?<=[.!?])\s", t.strip(), maxsplit=1)[0])
        cands = bold[-1:] + first[-1:] + numbers(t)[-1:]
    try:
        return any(float(c) == float(gold) for c in cands)
    except ValueError:
        return False


def write_packet(out, title, items, rows, models):
    by_key = {(r["id"], r["model"]): r for r in rows}
    key, lines = {}, [f"# {title}", "",
                      f"{len(items)} prompts, {len(models)} replies each. Labels are shuffled for every prompt; "
                      "the key is in `key.json`. Only the final answer is shown.", ""]
    for n, (pid, prompt, _) in enumerate(items, 1):
        order = list(models)
        random.Random(f"christomorphic:{pid}").shuffle(order)
        key[pid] = {chr(65 + i): m for i, m in enumerate(order)}
        lines += [f"## {n}. {pid}", "", "**Prompt:** " + prompt, ""]
        for i, m in enumerate(order):
            r = by_key[(pid, m)]
            text = r["final"] or "*(no final answer)*"
            if not r["complete"]:
                text += f"\n\n*(cut off at {r['tokens']} tokens)*"
            lines += [f"### {chr(65 + i)}", "", text, ""]
        lines += ["---", ""]
    (out / "packet.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="subset", help="subset | full | gsm8k:N")
    ap.add_argument("--models", default="base20,r38", help=f"comma list from {sorted(KNOWN)}")
    ap.add_argument("--model", action="append", default=[], help="extra name=tinker://... sampler path")
    ap.add_argument("--format", action="append", choices=["bare", "native"], help="default: bare")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0, help="OpenAI recommends 1.0 for gpt-oss")
    ap.add_argument("--temp", action="append", default=[], help="per-model override, e.g. r38=0 (its best setting)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
    ap.add_argument("--title", default="Christomorphic review packet")
    args = ap.parse_args()

    items = load_set(args.set)[: args.limit]
    formats = args.format or ["bare"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer("openai/gpt-oss-20b")  # gpt-oss-20b and 120b share the o200k_harmony tokenizer
    rends = {f: make_renderer(f, tokenizer) for f in formats}
    specs = {m: KNOWN[m] for m in args.models.split(",") if m} | dict(m.split("=", 1) for m in args.model)
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    clients = {}
    for name, spec in specs.items():
        clients[name] = await (service.create_sampling_client_async(base_model=spec[5:]) if spec.startswith("base:")
                               else service.create_sampling_client_async(model_path=spec))

    gate = asyncio.Semaphore(args.concurrency)
    temps = {k: float(v) for k, v in (t.split("=") for t in args.temp)}

    async def one(name, fmt, s, n, pid, prompt, gold):
        r = rends[fmt]
        params = types.SamplingParams(max_tokens=args.max_tokens, temperature=temps.get(name, args.temperature), top_p=1.0,
                                      seed=1000 * s + n, stop=r.get_stop_sequences())
        model_input = r.build_generation_prompt([{"role": "user", "content": prompt}])
        async with gate:
            resp = await clients[name].sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        seq = resp.sequences[0]
        analysis, final, complete = split_channels(r, seq.tokens)
        row = {"id": pid, "model": name, "spec": specs[name], "format": fmt, "temperature": params.temperature,
               "sample": s, "prompt": prompt,
               "analysis": analysis, "final": final.strip(), "complete": complete,
               "stop_reason": seq.stop_reason, "tokens": len(seq.tokens)}
        if gold is not None:
            row |= {"gold": gold, "correct": is_correct(final, gold)}
        return row

    jobs = [(name, fmt, s, n, pid, prompt, gold) for n, (pid, prompt, gold) in enumerate(items)
            for name in clients for fmt in formats for s in range(args.samples)]
    rows = await asyncio.gather(*(one(*j) for j in jobs))
    with open(out / "responses.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    first = [r for r in rows if r["format"] == formats[0] and r["sample"] == 0]
    write_packet(out, args.title, items, first, list(clients))

    for name in clients:
        for fmt in formats:
            mine = [r for r in rows if r["model"] == name and r["format"] == fmt]
            line = (f"{name:8s} {fmt:6s} complete {sum(r['complete'] for r in mine)}/{len(mine)}, "
                    f"mean tokens {sum(r['tokens'] for r in mine) / len(mine):.0f}, "
                    f"mean final chars {sum(len(r['final']) for r in mine) / len(mine):.0f}")
            if "correct" in mine[0]:
                line += f", correct {sum(r['correct'] for r in mine)}/{len(mine)}"
            print(line)
    print(f"sampled tokens: {sum(r['tokens'] for r in rows)}; wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
