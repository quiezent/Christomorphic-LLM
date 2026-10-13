"""LoRA-train gpt-oss-20b so that its first movement is Scripture, recalled in its private analysis channel.

Each example renders as
    <|start|>user<|message|>{a person's words from Scripture}<|end|><|start|>assistant<|channel|>analysis<|message|>
    {Book c:v-v (ESV|NKJV)}\n{the whole scene, exact verse text}\n\n
Loss falls only on the recalled reference and scene. Everything up to <|message|> has weight 0, and the
example stops after the paragraph break: no <|end|>, no final channel, no <|return|>. The model's own
reasoning and its final answer are never trained, so it keeps them.

    python train.py --name fm1 --dry      # print datum 0 token by token, count tokens, no Tinker calls
    python train.py --name fm1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.tokenizer_utils import get_tokenizer

PROJECT_ID = "dab755b5-88e7-4876-9829-e9ecc1063e39"
BASE_MODEL = "openai/gpt-oss-20b"
ANALYSIS = [200005, 35644, 200008]  # <|channel|>analysis<|message|>
HERE = Path(__file__).parent


def build(rows, tok, renderer, batch):
    data = []
    for r in rows:
        head = renderer.build_generation_prompt([{"role": "user", "content": r["user"]}]).to_ints()
        assert head[-2:] == [200006, 173781], "generation prompt must end with <|start|>assistant"
        head += ANALYSIS
        target = tok.encode(r["analysis"] + "\n\n", add_special_tokens=False)
        tokens = head + target
        weights = [0.0] * (len(head) - 1) + [1.0 / (len(target) * batch)] * len(target)  # each step sums to 1
        data.append(types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]),
                                loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights}))
    return data


def show(datum, tok):
    ids = datum.model_input.to_ints()
    w = datum.loss_fn_inputs["weights"].tolist()
    t = datum.loss_fn_inputs["target_tokens"].tolist()
    print(f"input {len(ids)} tokens; weighted {sum(x > 0 for x in w)}")
    for i in range(len(t)):
        print(f"{i:4d} {'W' if w[i] > 0 else '.'} {t[i]:7d} {tok.decode([t[i]])!r}")


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(HERE / "runs" / "data" / "train.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--ttl-days", type=int, default=14)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    tok = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer("gpt_oss_no_sysprompt", tok)
    data = build(rows, tok, renderer, args.batch)
    per_epoch = len(data) // args.batch
    total = per_epoch * args.epochs
    n_tok = int(sum(d.model_input.length for d in data) / len(data) * args.batch * total)
    print(f"{len(data)} examples, {per_epoch} steps/epoch, {total} steps, ~{n_tok} training tokens "
          f"(~${n_tok / 1e6 * 0.396:.2f} at $0.396/M)")
    if args.dry:
        show(data[0], tok)
        return

    out = HERE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"examples": len(data), "steps": total}, indent=1))
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    tc = service.create_lora_training_client(BASE_MODEL, rank=args.rank, seed=args.seed,
                                             train_mlp=True, train_attn=True, train_unembed=False)
    ttl = args.ttl_days * 86400
    checkpoints, step = {}, 0
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = list(range(len(data)))
            random.Random(args.seed + epoch).shuffle(order)
            for s in range(per_epoch):
                batch = [data[i] for i in order[s * args.batch:(s + 1) * args.batch]]
                lr = lr_at(step, total, args.lr, args.warmup)
                t0 = time.time()
                fb = tc.forward_backward(batch, "cross_entropy")
                op = tc.optim_step(types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8,
                                                    weight_decay=0.0, grad_clip_norm=1.0))
                res = fb.result()
                op.result()
                nll = compute_mean_nll([o["logprobs"] for o in res.loss_fn_outputs],
                                       [d.loss_fn_inputs["weights"] for d in batch])
                rec = {"step": step + 1, "epoch": epoch + 1, "lr": lr, "nll": nll, "secs": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
                step += 1
            path = tc.save_weights_for_sampler(f"{args.name}-e{epoch + 1}", ttl_seconds=ttl).result().path
            checkpoints[f"sampler_e{epoch + 1}"] = path
            print("saved", path, flush=True)
            (out / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1))
    checkpoints["state_final"] = tc.save_state(f"{args.name}-final", ttl_seconds=ttl).result().path
    (out / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1))
    print(json.dumps(checkpoints, indent=1))


if __name__ == "__main__":
    main()
