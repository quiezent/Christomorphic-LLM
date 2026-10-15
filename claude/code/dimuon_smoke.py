"""Three-step smoke test: does Dimuon train a gpt-oss-20b LoRA on Tinker? (Tinker Owner's advice, 15 Oct 2026.)

Two clients, rank 32, same seed and datum: AdamW at LR 1e-4 and Dimuon at LR 1e-3 (the docs' starting rates).
Each takes three steps on one short non-Scripture passage; the loss before and after shows whether an update happened.
No checkpoint is saved.

    python dimuon_smoke.py
"""
from tinker import types
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer

from evaluate import PROJECT_ID
from train import BASE_MODEL

TEXT = ("The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step. "
        "Measure twice and cut once. Many hands make light work.")


def main():
    tok = get_tokenizer(BASE_MODEL)
    ids = tok.encode(TEXT, add_special_tokens=False)
    datum = types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                        loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0 / (len(ids) - 1)] * (len(ids) - 1)})
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    arms = {"adamw": (types.AdamOptimizerConfig(), lambda: types.AdamParams(learning_rate=1e-4)),
            "dimuon": (types.DimuonOptimizerConfig(), lambda: types.DimuonParams(learning_rate=1e-3))}
    for name, (cfg, params) in arms.items():
        tc = service.create_lora_training_client(BASE_MODEL, rank=32, seed=20261015, train_unembed=False, optimizer=cfg)
        losses = []
        for _ in range(3):
            res = tc.forward_backward([datum], "cross_entropy").result()
            losses.append(round(res.metrics.get("loss:sum", float("nan")), 4))
            tc.optim_step(params()).result()
        res = tc.forward_backward([datum], "cross_entropy").result()
        losses.append(round(res.metrics.get("loss:sum", float("nan")), 4))
        print(f"{name:7s} mean NLL per token before/after each step: {losses}", flush=True)


if __name__ == "__main__":
    main()
