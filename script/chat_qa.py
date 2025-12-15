import sys
from pathlib import Path

import tinker
from tinker import types

# Ensure local cookbook modules are importable (renderer + tokenizer utils)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tinker_cookbook import tokenizer_utils

# 1. Point to YOUR sampler weights path (from save_weights_for_sampler or save_weights_and_get_sampling_client)
MODEL_PATH = "tinker://617176da-0732-5902-a4c3-1cb8402b7c29:train:0/sampler_weights/llama1b-christomorphic-chat"

# 2. Base model name must match what you trained
BASE_MODEL = "meta-llama/Llama-3.2-1B"


def main():
    # Connect to Tinker and create a sampling client for your fine-tuned model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)

    system_prompt = (
#        "You are a Christomorphic assistant."
        "You are a useful assistant."
#        "Answer every question by revealing Jesus Christ as the center, "
#        "honoring Scripture, the Trinity, and the glory (kavod) of God."
    )

    print("Christomorphic model loaded. Press Enter on an empty line to quit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            break
        if not user_text:
            break

        prompt_text = f"{system_prompt}\n\nQuestion: {user_text}\nAnswer:"
        model_input = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

        sampling_params = types.SamplingParams(
            max_tokens=200,
            temperature=0.2,
            top_p=0.9,
            stop=["\n"],
        )

        result = sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()

        answer = tokenizer.decode(result.sequences[0].tokens).strip()
        print(f"Model: {answer}\n")


if __name__ == "__main__":
    main()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
