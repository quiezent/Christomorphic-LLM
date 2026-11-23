"""The middot of the weights, before any training: in a base model's hidden states, do verses that
the Bible links sit nearer each other than the same verses with their links scrambled? No training, no generated text.

Each BSB verse goes in alone, as plain text. Its vector at a layer is the mean of that layer's hidden states over the
verse's tokens. Layer 0 is the word embeddings, before any context. It already carries word meaning, not only spelling.

To scramble a link, one end chosen at random stays and the other is replaced by a verse from the same book, in the same
length band, that the link type does not join to it. Each result averages 20 scrambles. The interval resamples verses
at both ends of every pair, and each draw uses the next scramble in turn.

Shared wording alone would make linked verses look near (a same-word link shares the word by construction), so every
result is also given after removing what TF-IDF word overlap and verse length explain in one linear fit ("beyond wording").
That control is approximate: other forms of it move the values by about 0.03 to 0.06.

    python measure_base.py embed --name olmo3_7b --model allenai/Olmo-3-7B-Instruct --revision <commit> --4bit   # -> runs/seven/<name>.npy
    python measure_base.py compare --name olmo3_7b                                            # -> runs/seven/<name>_compare.{json,md}
"""
import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bible import CODES, verses
from rd0 import XREF, jl, ob_ref

HERE = Path(__file__).parent
OUT = HERE / "runs" / "seven"
SEED = 20261109
BOOTSTRAP = 200
SCRAMBLES = 20
VOTES = 10
NAMES = {"7307": "Spirit (רוּחַ)", "2451": "wisdom (חָכְמָה)", "998": "understanding (בִּינָה)", "6098": "counsel (עֵצָה)",
         "1369": "might (גְּבוּרָה)", "1847": "knowledge (דַּעַת)", "3374": "fear (יִרְאָה)"}


def load_model(model_id, revision, four_bit, adapter=None, cpu_words=False):
    """A base model in eval mode, with a LoRA adapter on top if one is given. Vision-language checkpoints load through the image-text class.
    cpu_words keeps an untied word-embedding table on the CPU: a lookup is a copy, and the card gains about 1 GB (Ministral's table is
    131,072 × 4,096 in bf16). Checked on 11 Nov: loss and logits bit-identical; LoRA gradients differ by 0.42% (relative L2), inside
    the card's own run-to-run difference of 0.42 to 0.57% with the table in either place."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kw = {"revision": revision, "dtype": torch.bfloat16, "device_map": {"": 0}}
    gemma = "gemma" in model_id.lower()
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                                       llm_int8_enable_fp32_cpu_offload=gemma)
    if gemma:  # never let Gemma 4's 5.6 GB per-layer embedding table onto the card: a spill there leaves the model slow even after moving it off
        lang = "model.language_model."
        kw["device_map"] = {"lm_head": 0, **{lang + p: 0 for p in ("embed_tokens", "layers", "norm", "rotary_emb", "per_layer_model_projection", "per_layer_projection_norm")},
                            lang + "embed_tokens_per_layer": "cpu", "model.vision_tower": "cpu", "model.audio_tower": "cpu", "model.embed_vision": "cpu", "model.embed_audio": "cpu"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    inner = getattr(model, "model", None)
    for part in ("vision_tower", "multi_modal_projector", "audio_tower"):  # text only: free the card
        if getattr(inner, part, None) is not None:
            setattr(inner, part, None)
    table = getattr(getattr(inner, "language_model", None), "embed_tokens_per_layer", None)
    if table is not None:  # the table stays on the CPU; only the few rows a batch reads go to the GPU
        # accelerate offloads it to "meta" and copies all 5.6 GB onto the card at every forward, so hold the real weights here instead
        from accelerate.hooks import remove_hook_from_module
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        with safe_open(hf_hub_download(model_id, "model.safetensors", revision=revision), "pt") as f:
            weight = f.get_tensor("model.language_model.embed_tokens_per_layer.weight")
        remove_hook_from_module(table)
        table.weight = torch.nn.Parameter(weight, requires_grad=False)
        table.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
        table.register_forward_hook(lambda m, a, out: out.to(0))
    words = model.get_input_embeddings()
    if cpu_words and words.weight is not model.get_output_embeddings().weight:  # never a tied table: the output layer would move with it
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(words)
        words.to("cpu")
        words.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
        words.register_forward_hook(lambda m, a, out: out.to(0))
    torch.cuda.empty_cache()
    if adapter:
        from peft import PeftModel
        if table is not None:  # PEFT re-dispatches any model whose map names the CPU, which pushes Gemma's word embeddings off the card
            model.hf_device_map = {"": 0}
        model = PeftModel.from_pretrained(model, adapter)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if "gemma" in model_id.lower():  # its tokenizer adds no BOS to plain text, and the model expects one
        tok.add_bos_token = True
    return model.eval(), tok


def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def ordered():
    bsb = verses("BSB")
    key = lambda v: (CODES.index(v.split(".")[0]), int(v.split(".")[1]), int(v.split(".")[2]))
    ids = sorted(bsb, key=key)
    return ids, [bsb[v] for v in ids]


def embed(args):
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model, args.revision, args.four_bit, args.adapter)
    cfg = text_config(model)
    n_layers, dim = cfg.num_hidden_layers, cfg.hidden_size
    LAYERS = (0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers)
    print("loaded; layers", n_layers, "dim", dim, "sampled", LAYERS, "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)

    ids, texts = ordered()
    if args.limit:
        ids, texts = ids[:args.limit], texts[:args.limit]
    lengths = np.array([len(tok(t, add_special_tokens=False).input_ids) for t in texts])
    done = np.lib.format.open_memmap(OUT / f"{args.name}.npy", mode="w+", dtype=np.float32, shape=(len(ids), len(LAYERS), dim))
    t0, n = time.time(), 0
    for start in range(0, len(ids), 4096):  # sort within large blocks by length, so batches waste little padding
        block = sorted(range(start, min(start + 4096, len(ids))), key=lambda i: lengths[i])
        for b in range(0, len(block), args.batch):
            idx = block[b:b + args.batch]
            enc = tok([texts[i] for i in idx], add_special_tokens=True, padding=True, return_tensors="pt").to(0)  # the model's own BOS, if it has one
            with torch.inference_mode():
                hs = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            mask = enc["attention_mask"].clone()
            if tok.bos_token_id is not None:
                mask[enc["input_ids"] == tok.bos_token_id] = 0  # the BOS is not part of the verse
            mask = mask.unsqueeze(-1).float()
            for j, layer in enumerate(LAYERS):
                mean = (hs[layer].float() * mask).sum(1) / mask.sum(1)
                done[idx, j] = mean.cpu().numpy()
            n += len(idx)
        print(f"{n}/{len(ids)} verses, {n / (time.time() - t0):.0f}/s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    done.flush()
    meta = {"model": args.model, "revision": args.revision, "adapter": args.adapter, "four_bit": args.four_bit, "layers": LAYERS, "ids": ids, "tokens": lengths.tolist(),
            "torch": torch.__version__, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (OUT / f"{args.name}_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def expand(s):
    r = ob_ref(s)
    return [f"{r[0]}.{r[1]}.{v}" for v in range(r[2], r[3] + 1)] if r else []


def num(strong):
    m = re.search(r"\d+", strong or "")
    return m[0] if m else ""


def word_sets():
    members = defaultdict(set)
    for row in jl(HERE / "data" / "orig" / "WLC.jsonl"):
        ns = {num(w.get("strong")) for w in row["words"]}
        for n in NAMES:
            if n in ns:
                members[n].add(row["id"])
    return members


def renderings():
    """How the BSB renders רוּחַ in each verse, sorted into spirit, wind, breath or other."""
    out = {}
    for row in jl(HERE / "data" / "orig" / "BSB_tables.jsonl"):
        for w in row["words"]:
            if num(w.get("strong")) == "7307" and (w.get("strong") or "").startswith("H"):
                g = (w.get("gloss") or "").lower()
                out.setdefault(row["id"], "spirit" if "spirit" in g else "wind" if "wind" in g else "breath" if "breath" in g else "other")
    return out


def openbible(index):
    """All links with enough votes, and the subset stated verse to verse (a range-to-range line pairs every verse of each)."""
    links, single = set(), set()
    for line in open(XREF, encoding="utf-8").read().splitlines()[1:]:
        a, b, votes = line.split("\t")[:3]
        if int(votes) >= VOTES:
            for x in expand(a):
                for y in expand(b):
                    if x != y and x in index and y in index:
                        links.add((min(x, y), max(x, y)))
                        if "-" not in a and "-" not in b:
                            single.add((min(x, y), max(x, y)))
    return links, single


def tfidf(texts):
    docs = [Counter(re.findall(r"[a-z']+", t.lower())) for t in texts]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    vecs = []
    for d in docs:
        v = {w: (1 + math.log(c)) * math.log(n / df[w]) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def overlap(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b[w] for w, x in a.items() if w in b)


def scramble(pairs, joined, ids, index, tokens, rng):
    """Keep one end of each pair, chosen at random, and replace the other by a verse of its book and length band that this
    link type does not join to the end kept."""
    book = lambda v: v.split(".")[0]
    pool = defaultdict(list)
    bands = np.quantile(tokens, [0.2, 0.4, 0.6, 0.8])
    band = lambda v: int(np.searchsorted(bands, tokens[index[v]]))
    for v in ids:
        pool[book(v), band(v)].append(v)
    out = []
    for a, b in pairs:
        if rng.random() < 0.5:
            a, b = b, a
        cands = pool[book(b), band(b)]
        for _ in range(200):
            c = rng.choice(cands)
            if c not in (a, b) and (min(a, c), max(a, c)) not in joined:
                out.append((a, c))
                break
        else:
            out.append(None)
    return out


def auc(pos, neg, wp=None, wn=None):
    """The chance that a linked pair is nearer than a scrambled pair, ties counted half; optionally weighted."""
    wp = np.ones(len(pos)) if wp is None else wp
    wn = np.ones(len(neg)) if wn is None else wn
    u, inv = np.unique(np.concatenate([pos, neg]), return_inverse=True)
    p = np.bincount(inv[:len(pos)], wp, len(u))
    n = np.bincount(inv[len(pos):], wn, len(u))
    return float((p * (np.cumsum(n) - n / 2)).sum() / (p.sum() * n.sum()))


def residual(y, features):
    X = np.column_stack([np.ones(len(y))] + features)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def compare(args):
    meta = json.loads((OUT / f"{args.name}_meta.json").read_text(encoding="utf-8"))
    ids, tokens = meta["ids"], np.array(meta["tokens"])
    index = {v: i for i, v in enumerate(ids)}
    arr = np.load(OUT / f"{args.name}.npy", mmap_mode="r")
    bsb = verses("BSB")
    vecs = tfidf([bsb[v] for v in ids])
    rng = random.Random(SEED)
    members = word_sets()
    kinds = renderings()

    sets = {}
    for n, title in NAMES.items():
        vs = sorted(v for v in members[n] if v in index)
        sets[title] = {(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]}
    ruach = sorted(v for v in members["7307"] if v in index and v in kinds)
    sets["Spirit (רוּחַ), rendered differently in English"] = {(min(a, b), max(a, b)) for i, a in enumerate(ruach) for b in ruach[i + 1:] if kinds[a] != kinds[b]}
    union = {v for s in members.values() for v in s if v in index}
    nt = set(CODES[39:])
    ob, single = openbible(index)
    sets[f"OpenBible {VOTES}+ votes, both ends in the seven-word verses"] = {p for p in ob if p[0] in union and p[1] in union}
    sets[f"OpenBible {VOTES}+ votes, seven-word verse to New Testament"] = {p for p in ob if (p[0] in union and p[1].split(".")[0] in nt) or (p[1] in union and p[0].split(".")[0] in nt)}
    sets[f"OpenBible {VOTES}+ votes, whole Bible"] = ob
    sets[f"OpenBible {VOTES}+ votes, whole Bible, stated verse to verse"] = single
    if args.focus:  # for before-and-after: only the sets the training is about, split by whether the verses were trained
        split = json.loads((HERE / "runs" / "train" / "split.json").read_text(encoding="utf-8"))
        trained, held = set(split["train"]), set(split["held"])
        by_word = {n: sets[title] for n, title in NAMES.items()}
        same = set().union(*by_word.values())
        to_nt = next(s for title, s in sets.items() if "to New Testament" in title)
        rows = [json.loads(l)["verses"] for l in open(HERE / "runs" / "train" / "links.jsonl", encoding="utf-8")]
        words = lambda v: re.findall(r"[a-z']+", bsb[v].lower())
        seen = {tuple(w[i:i + 8]) for v in {v for r in rows for v in r} for w in [words(v)] for i in range(len(w) - 7)}
        cover = lambda w: len({j for i in range(len(w) - 7) if tuple(w[i:i + 8]) in seen for j in range(i, i + 8)}) / max(len(w), 1)
        held = {v for v in held if cover(words(v)) < 0.5}  # some held-out verses are near-copies of trained ones (Kings/Chronicles, Isaiah/2 Kings)
        together = {(min(a, b), max(a, b)) for r in rows for i, a in enumerate(r) for b in r[i + 1:]}
        linked = same | ob
        both = lambda s, g: {p for p in s if p[0] in g and p[1] in g}

        def control(group, out=linked):  # any shared drift of the trained verses shows here too: subtract it
            t = sorted(group)
            pairs = [(a, b) for i, a in enumerate(t) for b in t[i + 1:] if (a, b) not in out]
            random.Random(SEED).shuffle(pairs)
            return set(pairs[:20000])

        six = set().union(*(s for n, s in by_word.items() if n != "7307"))
        sets = {"Same word, both trained": both(same, trained),
                "Same word, both trained, never read together in the links arm": both(same, trained) - together,
                "Same word, both held out": both(same, held),
                "רוּחַ alone, both trained": both(by_word["7307"], trained),
                "רוּחַ alone, both held out": both(by_word["7307"], held),
                "The six other words, both trained": both(six, trained),
                "The six other words, both held out": both(six, held),
                "OpenBible to New Testament, seven-word verse trained": {p for p in to_nt if p[0] in trained or p[1] in trained},
                "OpenBible to New Testament, held-out Old Testament verse (the New Testament verse may be trained)": {p for p in to_nt if p[0] in held or p[1] in held},
                "Control: both trained, not linked": control(trained),
                "Control: both held out, not linked": control(held)}
        relabel = HERE / "runs" / "train" / "wrong.json"
        if relabel.exists():  # the wrong-links arm's sets come last, so every set above keeps its scrambles and its numbers
            perm = json.loads(relabel.read_text(encoding="utf-8"))["perm"]
            image = lambda s: {(min(perm.get(a, a), perm.get(b, b)), max(perm.get(a, a), perm.get(b, b))) for a, b in s}
            taught = image(linked)  # every relation the wrong-links arm read in place of the Bible's
            sets[TAUGHT_LINKS] = both(same, trained) - taught
            sets[TAUGHT_WRONG] = image(both(same, trained)) - linked
            sets[NEITHER] = control(trained, linked | taught)
            # the word split (12 Nov 2026), last again so every set above keeps its scrambles and its numbers
            sets[SIX_LINKS] = both(six, trained) - taught
            sets[SIX_WRONG] = image(both(six, trained)) - linked
            sets[RUACH_LINKS] = both(by_word["7307"], trained) - taught
            sets[RUACH_WRONG] = image(both(by_word["7307"], trained)) - linked

    results = []
    la = np.log(tokens)
    layer_vecs = {}
    for j, layer in enumerate(meta["layers"]):
        m = np.asarray(arr[:, j], dtype=np.float32)
        m = m - m.mean(0)
        layer_vecs[layer] = m / np.linalg.norm(m, axis=1, keepdims=True)
    for title, joined in sets.items():
        pairs = sorted(joined)
        a = np.array([index[p[0]] for p in pairs])
        b = np.array([index[p[1]] for p in pairs])
        tf_pos = np.array([overlap(vecs[i], vecs[j]) for i, j in zip(a, b)])
        negs, dropped = [], 0
        for _ in range(SCRAMBLES):
            found = [s for s in scramble(pairs, joined, ids, index, tokens, rng) if s]
            dropped = max(dropped, len(pairs) - len(found))
            sa = np.array([index[s[0]] for s in found])
            c = np.array([index[s[1]] for s in found])
            negs.append((sa, c, np.array([overlap(vecs[i], vecs[j]) for i, j in zip(sa, c)])))
        row = {"set": title, "pairs": len(pairs), "dropped_at_most": dropped, "layers": {},
               "tfidf_auc": float(np.mean([auc(tf_pos, n[2]) for n in negs]))}
        for layer, m in layer_vecs.items():
            pos = (m[a] * m[b]).sum(1)
            raw, beyond, parts = [], [], []
            for sa, c, tf_neg in negs:
                neg = (m[sa] * m[c]).sum(1)
                res = residual(np.concatenate([pos, neg]), [np.concatenate([tf_pos, tf_neg]), np.concatenate([la[a], la[sa]]), np.concatenate([la[b], la[c]])])
                raw.append(auc(pos, neg))
                beyond.append(auc(res[:len(pos)], res[len(pos):]))
                parts.append((res[:len(pos)], res[len(pos):], sa, c))
            boot, draws = np.random.default_rng(SEED), []
            for d in range(BOOTSTRAP):
                rp, rn, sa, c = parts[d % SCRAMBLES]
                verts = np.unique(np.concatenate([a, b, sa, c]))
                w = np.zeros(len(ids))
                np.add.at(w, boot.choice(verts, len(verts)), 1)
                wp, wn = w[a] * w[b], w[sa] * w[c]
                draws.append(auc(rp, rn, wp, wn) if wp.sum() and wn.sum() else float("nan"))
            row["layers"][layer] = {"auc": float(np.mean(raw)), "beyond_wording_auc": float(np.mean(beyond)), "scramble_sd": float(np.std(beyond)),
                                    "low": float(np.nanquantile(draws, 0.025)), "high": float(np.nanquantile(draws, 0.975)), "draws": draws}
        results.append(row)
        print(title, row["pairs"], "pairs;", json.dumps(row["layers"][meta["layers"][2]]), flush=True)

    md = [f"# Linked against scrambled: {meta['model']}{' (4-bit)' if meta['four_bit'] else ''}", "",
          f"0.50 means linked verses are no nearer than scrambled ones. Each cell averages {SCRAMBLES} scrambles. 'Beyond wording' removes what "
          "TF-IDF overlap and length explain in one linear fit, and is approximate. Its interval resamples verses at both ends of every pair.", "",
          "| Link set | Pairs | TF-IDF alone | " + " | ".join(f"Layer {l}" for l in meta["layers"]) + " |",
          "|---|---:|---:|" + "---:|" * len(meta["layers"])]
    for r in results:
        cells = [f"{r['layers'][l]['beyond_wording_auc']:.3f} ({r['layers'][l]['low']:.3f}–{r['layers'][l]['high']:.3f}) raw {r['layers'][l]['auc']:.3f}" for l in meta["layers"]]
        md.append(f"| {r['set']} | {r['pairs']:,} | {r['tfidf_auc']:.3f} | " + " | ".join(cells) + " |")
    tag = "_focus" if args.focus else ""
    (OUT / f"{args.name}{tag}_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (OUT / f"{args.name}{tag}_compare.json").write_text(json.dumps({"meta": {k: meta[k] for k in ("model", "four_bit", "layers", "peak_gb")}, "votes": VOTES, "seed": SEED, "scrambles": SCRAMBLES, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(md))


TAUGHT_LINKS = "Same word, both trained, not a wrong-links pair"
TAUGHT_WRONG = "Wrong-links pair, both trained, not a real link"
NEITHER = "Control: both trained, linked in neither arm"
SIX_LINKS = "The six other words, both trained, not a wrong-links pair"
SIX_WRONG = "Wrong-links image of the six other words, both trained, not a real link"
RUACH_LINKS = "רוּחַ alone, both trained, not a wrong-links pair"
RUACH_WRONG = "Wrong-links image of רוּחַ, both trained, not a real link"
SEED_RULE = ("Rule set before seed 2 was trained. On 'Same word, both trained': seed 2's links − scrambled "
             "keeps seed 1's sign and reaches at least 0.005. The two same-arm differences (links seed 1 − seed 2, scrambled seed 1 − seed 2) hold only "
             "training-run noise: about 0.002 or less means seed 1's gap is several times that noise; 0.005 or more means it is near noise.")


def seed_section(rows, primary, d, v):
    """Two training runs per arm: the contrasts across runs, and the same-arm differences that are run noise alone."""
    contrasts = [("links", "scrambled", "Seed 1: links − scrambled"), ("links_s2", "scrambled_s2", "Seed 2: links − scrambled"),
                 ("links", "scrambled_s2", "Links 1 − scrambled 2"), ("links_s2", "scrambled", "Links 2 − scrambled 1"),
                 ("links", "links_s2", "Links: seed 1 − seed 2 (noise only)"), ("scrambled", "scrambled_s2", "Scrambled: seed 1 − seed 2 (noise only)")]
    sets = ["Same word, both trained", "Same word, both held out", "OpenBible to New Testament, seven-word verse trained"]
    q = lambda x: f"{np.nanquantile(x, 0.025):+.3f} to {np.nanquantile(x, 0.975):+.3f}"
    md = ["### Second training run", "", SEED_RULE, "", "| Contrast | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    for a, b, label in contrasts:
        md.append(f"| {label} | " + " | ".join(f"{v(a, s, primary) - v(b, s, primary):+.3f} ({q(d(a, s, primary) - d(b, s, primary))})" for s in sets) + " |")
    s = sets[0]
    g1 = v("links", s, primary) - v("scrambled", s, primary)
    g2 = v("links_s2", s, primary) - v("scrambled_s2", s, primary)
    noise = max(abs(v("links", s, primary) - v("links_s2", s, primary)), abs(v("scrambled", s, primary) - v("scrambled_s2", s, primary)))
    repeats = g2 * np.sign(g1) >= 0.005
    size = "about 0.002 or less" if noise <= 0.0025 else ("0.005 or more, so seed 1's gap is near noise" if noise >= 0.005 else "between 0.002 and 0.005")
    md += ["", f"Against the rule: seed 2's gap is {g2:+.3f} against seed 1's {g1:+.3f}, so it {'repeats' if repeats else 'does not repeat'}. "
               f"The larger same-arm difference is {noise:.3f}: {size}.", ""]
    return md


WRONG_RULE = ("Rule set before the wrong-links arm was trained, and revised before training after an "
              "independent review. The wrong-links arm reads the links arm's walks with each trained seven-word verse replaced by another read "
              "exactly as often: an order as consistent as the Bible's, whose relations are the Bible's only by chance. 2,253 of its 9,345 steps "
              "are still real links: 482 through the 10 verses alone in their visit count, which keep their place (Isaiah 11:2 among them), and "
              "1,771 by chance (31% of trained pairs share a word). Those pairs are left out of A and B, but they pull both toward zero. "
              "A = links − wrong links on same-word pairs the wrong-links arm was not given. B = wrong links − links on the wrong-links arm's own "
              "pairs that are not real links, minus the same difference on trained pairs linked in neither arm: drift between runs moves those two "
              "sets together and does not move the same-word pairs with them, so A is not netted. Both at the primary layer, beyond wording, with "
              "links seed 1, which shares the wrong-links run's initialisation and row order; A, B and A − B are read as printed, to three decimals. "
              "A ≥ 0.005 and A − B ≥ 0.005: the Bible's links drew their verses together more than an arbitrary consistent order drew its own. "
              "A ≥ 0.005 and A − B from −0.004 to +0.004: the two orders are not told apart. B ≥ 0.005 and A − B ≤ −0.005: the model took up the "
              "arbitrary order more than the Bible's. Otherwise, no reading: the links arm did not draw the Bible's pairs together clearly more than "
              "the wrong-links arm. A first or third reading whose A − B interval includes zero is stated as not established beyond verse sampling. "
              "The reading holds only if links seed 2 gives the same one; otherwise it is not settled. There is one wrong-links run, so its own run "
              "noise is not seen; between the two links runs, A's pairs moved by 0.003. Readings other than the first are negatives: the model "
              "remains closer to the pattern of this world, never evidence against Scripture.")
READINGS = ("the Bible's links drew their verses together more than an arbitrary consistent order drew its own",
            "the two orders are not told apart",
            "the model took up the arbitrary order more than the Bible's",
            "no reading: the links arm did not draw the Bible's pairs together clearly more than the wrong-links arm")


def reading(a, b):
    """Read in thousandths, as printed, so a reading never disagrees with the numbers beside it."""
    m = lambda x: round(float(f"{x:+.3f}") * 1000)
    a, b, gap = m(a), m(b), m(a - b)
    if a >= 5 and gap >= 5:
        return READINGS[0]
    if a >= 5 and abs(gap) < 5:
        return READINGS[1]
    if b >= 5 and gap <= -5:
        return READINGS[2]
    return READINGS[3]


def wrong_section(rows, primary, d, v):
    """The wrong-links arm against both links runs and both scrambled runs, on the pairs each arm was given and the other was not."""
    sets = [TAUGHT_LINKS, TAUGHT_WRONG, "Same word, both trained", "Same word, both held out", NEITHER]
    arms = [a for a in ("base", "links", "links_s2", "wrong", "scrambled", "scrambled_s2", "tinystories") if a in rows and all(s in rows[a] for s in sets)]
    cell = lambda a, s: f"{v(a, s, primary):.3f}"
    q = lambda x: f"{np.nanquantile(x, 0.025):+.3f} to {np.nanquantile(x, 0.975):+.3f}"
    gap = lambda x, y, s: v(x, s, primary) - v(y, s, primary)
    dgap = lambda x, y, s: d(x, s, primary) - d(y, s, primary)
    stat_a = lambda x, y: gap(x, y, TAUGHT_LINKS)  # x in the links role, y in the wrong-links role
    stat_b = lambda x, y: gap(y, x, TAUGHT_WRONG) - gap(y, x, NEITHER)  # run drift moves these two sets together; it does not move A's pairs with them
    md = ["### Wrong links", "", WRONG_RULE, "", "| Pairs | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    md += [f"| {s} | " + " | ".join(cell(a, s) for a in arms) + " |" for s in sets]
    contrasts = [(x, y, label) for x, y, label in [("links", "wrong", "Links 1 − wrong links"), ("links_s2", "wrong", "Links 2 − wrong links"),
                                                   ("wrong", "scrambled", "Wrong links − scrambled 1"), ("wrong", "scrambled_s2", "Wrong links − scrambled 2")]
                 if x in arms and y in arms]
    md += ["", "| Contrast | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    for x, y, label in contrasts:
        md.append(f"| {label} | " + " | ".join(f"{gap(x, y, s):+.3f} ({q(dgap(x, y, s))})" for s in sets) + " |")
    md.append("")
    found = []
    for links, label in (("links", "links seed 1"), ("links_s2", "links seed 2")):
        a, b = stat_a(links, "wrong"), stat_b(links, "wrong")
        da = dgap(links, "wrong", TAUGHT_LINKS)
        db = dgap("wrong", links, TAUGHT_WRONG) - dgap("wrong", links, NEITHER)
        found.append(reading(a, b))
        unsure = found[-1] in (READINGS[0], READINGS[2]) and np.nanquantile(da - db, 0.025) <= 0 <= np.nanquantile(da - db, 0.975)
        md.append(f"- With {label}: A {a:+.3f} ({q(da)}), B {b:+.3f} ({q(db)}), A − B {a - b:+.3f} ({q(da - db)}, approximate: the sets resample "
                  f"different verses): {found[-1]}{', not established beyond verse sampling' if unsure else ''}. "
                  f"Shown, not read: B before netting {gap('wrong', links, TAUGHT_WRONG):+.3f}.")
    noise = [(f"{x} − {y}", stat_a(x, y), stat_b(x, y)) for x, y in (("links", "links_s2"), ("scrambled", "scrambled_s2")) if x in arms and y in arms]
    md += ["", "Run noise alone, two runs of one arm in the two roles: " + "; ".join(f"{n}: A {a:+.3f}, B {b:+.3f}" for n, a, b in noise) + ".",
           "", f"Against the rule: {found[0]}; " + ("links seed 2 gives the same reading, so it holds." if found[0] == found[1]
                                                    else f"links seed 2 gives a different one ({found[1]}), so it is not settled."), ""]
    return md


WORD_RULE = ("Rule set on 12 Nov 2026, before a second wrong-links run was trained and after the first run showed the split; revised three "
             "times the same day, before training, after independent reviews. The first run's readings are shown, not read; its A enters the "
             "run-noise test. The second run is the same arm with the same relabelling at seed 2: it shares links seed 2's initialisation and "
             "row order. Both wrong-links runs keep the word table on the CPU while training and both links runs kept it on the card (checked on "
             "one batch: forward pass bit-identical, gradients within the card's own run-to-run noise); memory settings changed between them "
             "too, without changing the arithmetic. It tests whether the split survives a second training run, for this relabelling and these "
             "verses, not for other orders or other verses. The wrong-links rule's pairs fall in two groups, each with its own A and B: the six "
             "words other than רוּחַ (A on 8,456 pairs sharing wisdom, understanding, counsel, might, knowledge or fear, B on 8,430) and רוּחַ (A "
             "on 28,024, B on 28,053); 15 A pairs and 18 B pairs are in both groups. A = links − wrong links on the group's both-trained "
             "same-word pairs the wrong-links arm was not given, minus the same difference on trained pairs linked in neither arm (the control). "
             "B = wrong links − links on the wrong-links arm's image of all the group's both-trained same-word pairs, real links left out, minus "
             "the same difference on the control. Unlike the pooled rule, A is netted too, for both groups alike: between the scrambled arm's "
             "two runs the six words' A pairs moved with the control (+0.007 and +0.007), so a drift of a whole run would otherwise decide the "
             "reading. Because A and B share the control, a move of the control alone shifts A − B by twice that move. Both at the primary "
             "layer, beyond wording, with links seed 2. A and B are rounded to three decimals, A − B is taken from the rounded values, and they "
             "are read with the wrong-links rule's four readings and a fifth: A ≤ −0.005 where none of the first three applies, the wrong-links "
             "arm, which never read these pairs as neighbouring verses, drew them together more than the links arm (A alone). A first or third "
             "reading whose A − B interval includes zero, or a fifth whose A interval does, is stated as not established beyond verse sampling; "
             "'includes zero' means the unrounded 2.5% to 97.5% range contains 0. A group's reading holds only if links seed 1 gives the same "
             "one, qualifiers aside; otherwise it is not settled. This compares the same second run with the other links run; it is not a "
             "further training run. Run noise is the larger of two changes on the group's A pairs, as printed: the links arm's seed 1 − seed 2, "
             "net of the control, and the wrong-links arm's second-run A minus its first-run A, both with links seed 2. A reading is stated as "
             "near run noise if that noise is 0.005 or more and more than half the size of the second run's A, unless the first run's A was "
             "already 0.005 or more in size and the second run's has the same sign and is no smaller. The six words are the test: their lift is "
             "borne out only by a first reading that holds and is not near run noise, stated with whether its A − B interval includes zero; any "
             "other outcome means it is not borne out. With the first run's A at +0.012, this comes to: the second run's A at least +0.008, and "
             "A − B at least +0.005. רוּחַ is read as well, not as a second test. Shown, not read: the run noise on the image pairs and on the "
             "control, the scrambled arm's two runs, the wrong-links rule's pooled pairs with the second run, and the held-out pairs, too few to "
             "tell 0 from the lift being tested. Readings other than the first are negatives: the model remains closer to the pattern of this "
             "world, never evidence against Scripture.")
WORD_READINGS = READINGS + ("the wrong-links arm, which never read these pairs as neighbouring verses, drew them together more than the links arm",)
GROUPS = (("the six other words", SIX_LINKS, SIX_WRONG), ("רוּחַ", RUACH_LINKS, RUACH_WRONG))
UNSURE = ", not established beyond verse sampling"


def word_reading(a, b):
    """The wrong-links rule's four readings, and a fifth for a group the wrong-links arm drew together more. In thousandths, as printed."""
    found = reading(a, b)
    return WORD_READINGS[4] if found == READINGS[3] and round(float(f"{a:+.3f}") * 1000) <= -5 else found


def word_section(rows, primary, d, v, status="not trained"):
    """The word split: the six other words and רוּחַ, each with its own A and B, both netted against the control. The second
    wrong-links run is read; the first is shown. status says why the second run is missing, when it is."""
    sets = [SIX_LINKS, SIX_WRONG, RUACH_LINKS, RUACH_WRONG, NEITHER, "The six other words, both held out", "רוּחַ alone, both held out"]
    arms = [a for a in ("base", "links", "links_s2", "wrong", "wrong_s2", "scrambled", "scrambled_s2", "tinystories") if a in rows and all(s in rows[a] for s in sets)]
    q = lambda x: f"{np.nanquantile(x, 0.025):+.3f} to {np.nanquantile(x, 0.975):+.3f}"
    spans_zero = lambda x: np.nanquantile(x, 0.025) <= 0 <= np.nanquantile(x, 0.975)
    gap = lambda x, y, s: v(x, s, primary) - v(y, s, primary)
    net = lambda x, y, s: gap(x, y, s) - gap(x, y, NEITHER)  # a drift of a whole run moves the control too: take it out
    dnet = lambda x, y, s: (d(x, s, primary) - d(y, s, primary)) - (d(x, NEITHER, primary) - d(y, NEITHER, primary))
    r3 = lambda x: float(f"{x:+.3f}") + 0.0  # as printed; + 0.0 turns -0.0 into 0.0
    m = lambda x: round(r3(x) * 1000)
    ruach = lambda tl: " (read as well, not the test)" if tl == RUACH_LINKS else ""
    md = ["### The word split", "", WORD_RULE, "", "| Pairs | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    md += [f"| {s} | " + " | ".join(f"{v(a, s, primary):.3f}" for a in arms) + " |" for s in sets]
    md.append("")
    found = {}  # (wrong run, group, links run) -> (reading, not established beyond verse sampling, A as printed)
    for wrong, matched, title in (("wrong", "links", "Wrong links, seed 1, the run the split was chosen from (readings shown, not read)"),
                                  ("wrong_s2", "links_s2", "Wrong links, seed 2, the run this rule reads")):
        if wrong not in arms:
            md += [f"{title}: {status if wrong not in rows else 'measured, but without every set this section reads'}.", ""]
            continue
        md += [f"{title}:", ""]
        for group, tl, tw in GROUPS:
            for links in (matched, "links_s2" if matched == "links" else "links"):
                a, b = r3(net(links, wrong, tl)), r3(net(wrong, links, tw))
                da, db = dnet(links, wrong, tl), dnet(wrong, links, tw)
                r = word_reading(a, b)
                unsure = bool((r in (READINGS[0], READINGS[2]) and spans_zero(da - db)) or (r == WORD_READINGS[4] and spans_zero(da)))
                found[(wrong, group, links)] = (r, unsure, a)
                md.append(f"- {group[0].upper() + group[1:]}{ruach(tl)}, with links seed {1 if links == 'links' else 2}: A {a:+.3f} ({q(da)}), "
                          f"B {b:+.3f} ({q(db)}), A − B {r3(a - b):+.3f} ({q(da - db)}, approximate): {r}{UNSURE if unsure else ''}.")
        md.append("")
    if "wrong_s2" in arms:
        a1 = {g: found.get(("wrong", g, "links_s2"), (None, None, 0.0))[2] for g, _, _ in GROUPS}
        a2 = {g: found[("wrong_s2", g, "links_s2")][2] for g, _, _ in GROUPS}
        shift = {g: r3(a2[g] - a1[g]) for g in a2}  # equals the wrong-links arm's seed 1 − seed 2 on the A pairs, net of the control
        moved = {g: r3(net("links", "links_s2", tl)) for g, tl, _ in GROUPS}
        md += ["Run noise on each group's A pairs, as printed: " + "; ".join(
                   f"{g}: links seed 1 − seed 2, net of the control, {moved[g]:+.3f}; wrong links, the second run's A minus the first's, {shift[g]:+.3f}"
                   for g in a2) + ".",
               "", "Shown, not read. Run noise on the image pairs (seed 1 − seed 2, net of the control): " + "; ".join(
                   f"{g}: links {r3(net('links', 'links_s2', tw)):+.3f}, wrong links {r3(net('wrong', 'wrong_s2', tw)):+.3f}" for g, _, tw in GROUPS)
               + f". The wrong-links arm's control, seed 1 − seed 2, not netted: {r3(gap('wrong', 'wrong_s2', NEITHER)):+.3f}. The scrambled arm's "
               "two runs, not netted, on the A pairs: " + "; ".join(f"{g} {r3(gap('scrambled', 'scrambled_s2', tl)):+.3f}" for g, tl, _ in GROUPS)
               + f"; on its control {r3(gap('scrambled', 'scrambled_s2', NEITHER)):+.3f}. The wrong-links rule's pooled pairs with the second run, "
               "as that rule takes them: " + "; ".join(f"with links seed {n}: A {r3(gap(x, 'wrong_s2', TAUGHT_LINKS)):+.3f}, "
                                                      f"B {r3(gap('wrong_s2', x, TAUGHT_WRONG) - gap('wrong_s2', x, NEITHER)):+.3f}"
                                                      for x, n in (("links_s2", 2), ("links", 1))) + ".", ""]
        verdict = None
        for group, tl, _ in GROUPS:
            (first, unsure, _), (other, _, _) = found[("wrong_s2", group, "links_s2")], found[("wrong_s2", group, "links")]
            holds = first == other
            noise = max(abs(moved[group]), abs(shift[group]))
            grew = abs(m(a1[group])) >= 5 and m(a1[group]) * m(a2[group]) > 0 and abs(m(a2[group])) >= abs(m(a1[group]))  # a lift no smaller than the first run's is not noise
            near = m(noise) >= 5 and 2 * m(noise) > abs(m(a2[group])) and not grew
            md.append(f"Against the rule, {group}{ruach(tl)}: {first}{UNSURE if unsure else ''}; "
                      + ("links seed 1 gives the same reading, so this reading holds" if holds
                         else f"links seed 1 gives a different one ({other}), so this reading is not settled")
                      + (f"; the run noise is {noise:.3f}, so this reading is near run noise" if near else "") + ".")
            if tl == SIX_LINKS:
                if first == READINGS[0] and holds and not near:
                    verdict = ("borne out in the second training run, for this relabelling and these verses, "
                               + ("but its A − B interval over resampled verses includes zero" if unsure else "and its A − B interval over resampled verses excludes zero"))
                elif first == READINGS[0] and holds:
                    verdict = f"not borne out: the second run's A is {a2[group]:+.3f}, against {a1[group]:+.3f} in the first run, so it is near run noise"
                else:
                    verdict = f"not borne out: the second run reads '{first}'" + ("" if holds else ", and links seed 1 reads otherwise")
        md += ["", f"The test, the six other words: the lift is {verdict}.", ""]
    return md


def report(args):
    """Links minus scrambled, paired on the same bootstrap draws, at the third-quarter layer (fixed before any result was read)."""
    keys, md = ("olmo3_7b", "ministral3_8b", "gemma4_e4b"), ["# Before and after: linked against scrambled verses", ""]
    load = lambda name: json.loads((OUT / f"{name}_focus_compare.json").read_text(encoding="utf-8")) if (OUT / f"{name}_focus_compare.json").exists() else None
    for key in keys:
        runs = {"base": load(key), **{a: load(f"{key}_{a}") for a in ("links", "scrambled", "tinystories")}}
        if not runs["links"] or not runs["scrambled"]:
            md += [f"## {key}", "", "The links and scrambled arms did not both finish.", ""]
            continue
        rows = {a: {r["set"]: r for r in d["results"]} for a, d in runs.items() if d}
        layers = [str(l) for l in runs["links"]["meta"]["layers"]]
        primary, first = layers[3], layers[0]
        d = lambda a, s, l: np.array(rows[a][s]["layers"][l]["draws"])
        v = lambda a, s, l: rows[a][s]["layers"][l]["beyond_wording_auc"]
        ctrl = lambda s: "Control: both held out, not linked" if "held" in s.split(",")[1] else "Control: both trained, not linked"
        md += [f"## {key} (layer {primary} of {layers[-1]})", "",
               "Scores are beyond wording; 0.50 means linked verses are no nearer than scrambled ones. Net subtracts the same arm's unlinked-pair control, "
               "so shared drift of trained verses cancels. The interval is on the paired difference and resamples verses only, so it says nothing about run-to-run noise"
               + ("; the second training runs below show that noise. " if (OUT / f"{key}_links_s2_focus_compare.json").exists() else ", and this model has one training run per arm. ")
               + "Scramble partners of trained pairs are mostly verses never read in training, so absolute levels mix the relations with having been read; contrasts between arms share their partners. "
               "Each arm's all-verse mean vector is subtracted before the cosine. 'Held out' drops 14 of the 146 held-out verses as near-copies of trained text (8-word cover of 0.5 or more).", "",
               "| Pairs | Base | Links | Scrambled | TinyStories | Links − scrambled | Interval | Net of control | Interval |", "|---|---:|---:|---:|---:|---:|---|---:|---|"]
        for s in rows["links"]:
            if s in (TAUGHT_LINKS, TAUGHT_WRONG, NEITHER, SIX_LINKS, SIX_WRONG, RUACH_LINKS, RUACH_WRONG):  # the wrong-links sections below show these
                continue
            if s.startswith("Control"):
                cells = [f"{v(a, s, primary):.3f}" if a in rows and s in rows[a] else "–" for a in ("base", "links", "scrambled", "tinystories")]
                md.append(f"| {s} | " + " | ".join(cells) + " | | | | |")
                continue
            diff = d("links", s, primary) - d("scrambled", s, primary)
            gap = v("links", s, primary) - v("scrambled", s, primary)
            c = ctrl(s)
            net = diff - (d("links", c, primary) - d("scrambled", c, primary))
            net_gap = gap - (v("links", c, primary) - v("scrambled", c, primary))
            q = lambda x: f"{np.nanquantile(x, 0.025):+.3f} to {np.nanquantile(x, 0.975):+.3f}"
            cells = [f"{v(a, s, primary):.3f}" if a in rows and s in rows[a] else "–" for a in ("base", "links", "scrambled", "tinystories")]
            md.append(f"| {s} | " + " | ".join(cells) + f" | {gap:+.3f} | {q(diff)} | {net_gap:+.3f} | {q(net)} |")
        zero = max(abs(v("links", s, first) - v("scrambled", s, first)) for s in rows["links"] if s in rows["scrambled"])
        md += ["", f"Sanity check: LoRA does not touch the word embeddings, so at layer {first} the arms should agree. Largest links − scrambled there: {zero:.4f}.", ""]
        for a in ("links", "scrambled", "tinystories"):
            t = HERE / "runs" / "train" / f"{key}_{a}" / "train.json"
            if t.exists():
                j = json.loads(t.read_text(encoding="utf-8"))
                tail = [e["loss"] for e in j["log"][-len(j["log"]) // 4:]]  # one logged batch is two rows: too noisy to quote alone
                md.append(f"- {a}: {j['tokens']:,} tokens, {j['minutes']} minutes, mean logged loss over the last quarter {np.mean(tail):.3f}, stop note: {j['stop_note']}")
        md.append("")
        second = {a: load(f"{key}_{a}_s2") for a in ("links", "scrambled")}
        if all(second.values()):
            rows.update({f"{a}_s2": {r["set"]: r for r in second[a]["results"]} for a in second})
            md += seed_section(rows, primary, d, v)
            for a in ("links", "scrambled"):
                j = json.loads((HERE / "runs" / "train" / f"{key}_{a}_s2" / "train.json").read_text(encoding="utf-8"))
                tail = [e["loss"] for e in j["log"][-len(j["log"]) // 4:]]
                md.append(f"- {a}, seed 2: {j['tokens']:,} tokens, {j['minutes']} minutes, mean logged loss over the last quarter {np.mean(tail):.3f}, stop note: {j['stop_note']}")
            md.append("")
        relabelled = load(f"{key}_wrong")
        if relabelled and all(a in rows and TAUGHT_LINKS in rows[a] for a in ("links", "links_s2")):
            rows["wrong"] = {r["set"]: r for r in relabelled["results"]}
            md += wrong_section(rows, primary, d, v)
            j = json.loads((HERE / "runs" / "train" / f"{key}_wrong" / "train.json").read_text(encoding="utf-8"))
            tail = [e["loss"] for e in j["log"][-len(j["log"]) // 4:]]
            md += [f"- wrong links: {j['tokens']:,} tokens, {j['minutes']} minutes, mean logged loss over the last quarter {np.mean(tail):.3f}, stop note: {j['stop_note']}", ""]
            t = HERE / "runs" / "train" / f"{key}_wrong_s2" / "train.json"
            j = json.loads(t.read_text(encoding="utf-8")) if t.exists() else None
            again = load(f"{key}_wrong_s2")
            if again and j and not j["stop_note"]:  # a void run is never read, even if someone measured it
                rows["wrong_s2"] = {r["set"]: r for r in again["results"]}
            status = (("measured, but its train.json is missing" if again else "not trained") if not j
                      else f"trained but void ({j['stop_note']})" if j["stop_note"] else "trained, not yet measured")
            if all(SIX_LINKS in rows[a] for a in ("links", "links_s2", "wrong")):
                md += word_section(rows, primary, d, v, status)
            if j:
                tail = [e["loss"] for e in j["log"][-len(j["log"]) // 4:]]
                md += [f"- wrong links, seed 2: {j['tokens']:,} tokens, {j['minutes']} minutes, mean logged loss over the last quarter {np.mean(tail):.3f}, stop note: {j['stop_note']}", ""]
    path = HERE / "runs" / "train" / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))


def keep_running():
    """While this step runs, ask Windows not to slow it (Efficiency mode) and not to sleep: on 9 Nov, Gemma's CPU-side table lookup
    fell to a third of its speed a few minutes into steps run while no one was at the machine. The screen may still turn off.
    Both requests end with the process; no setting is changed."""
    import ctypes
    import sys
    from ctypes import wintypes
    if sys.platform != "win32":
        return

    class Throttle(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    state = Throttle(1, 0x1, 0)  # execution speed is controlled here, and never throttled
    k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))  # 4: ProcessPowerThrottling
    k.SetThreadExecutionState(0x80000000 | 0x00000001)  # continuous: system required. Never the display: the Legion 5 screen is OLED


def in_venv():
    """Every arm so far ran in this folder's .venv. A bare `python` here is the system install, with other torch, transformers,
    tokenizers and numpy versions (12 Nov 2026), so a step run there would not be comparable."""
    import sys
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise SystemExit(f"run with {HERE / '.venv' / 'Scripts' / 'python.exe'}, not {sys.executable}")


def main():
    in_venv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--name", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--revision", required=True)
    e.add_argument("--adapter", default=None, help="a LoRA adapter folder to apply over the base")
    e.add_argument("--4bit", dest="four_bit", action="store_true")
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(fn=embed)
    c = sub.add_parser("compare")
    c.add_argument("--name", required=True)
    c.add_argument("--focus", action="store_true", help="only the before-and-after sets, split by trained and held-out verses")
    c.set_defaults(fn=compare)
    sub.add_parser("report").set_defaults(fn=report)
    args = p.parse_args()
    keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
