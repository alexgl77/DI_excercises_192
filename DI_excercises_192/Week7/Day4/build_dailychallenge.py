"""
Generate dailychallenge.ipynb for Week 7 Day 4:
Fine-tune bigscience/bloomz-560m with LoRA + Hugging Face PEFT
on the Abirate/english_quotes dataset.
"""

import json
import os


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = [
    md(
        "# Daily Challenge — Fine-Tune an LLM with LoRA via Hugging Face PEFT",
        "",
        "**Course:** Developers Institute  **Week 7 - Day 4**  ",
        "**Author:** Alex Goldbaum",
        "",
        "End-to-end **Parameter-Efficient Fine-Tuning** (PEFT) of `bigscience/bloomz-560m`",
        "using LoRA on the **`Abirate/english_quotes`** dataset (10% sample). We will:",
        "",
        "1. Install `peft` + `datasets`.",
        "2. Load the BLOOMZ-560M base model + tokenizer.",
        "3. Load and tokenize a 10% sample of the quotes dataset.",
        "4. Configure LoRA (`LoraConfig`) and wrap the model with `get_peft_model`.",
        "5. Fine-tune with `Trainer` + `TrainingArguments` + a causal-LM data collator.",
        "6. Save the LoRA adapter.",
        "7. Reload it with `PeftModel.from_pretrained` and generate text.",
        "",
        "**⚠️ Runs in Colab on either CPU or GPU.** With a GPU the training takes a",
        "few minutes; on CPU it works but is noticeably slower — keep the dataset",
        "sample small and `num_train_epochs` ≤ 5.",
    ),

    md("## Setup"),
    code(
        "%pip install -qU peft==0.4.0 datasets transformers accelerate",
    ),
    code(
        "import os, time",
        "import torch",
        "from pathlib import Path",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print('Device:', device)",
        "",
        "# Cache directory the brief suggests",
        "Path('cache').mkdir(exist_ok=True)",
    ),

    # ============================================================
    # 1. Load model + tokenizer
    # ============================================================
    md("## 1. Load the Pre-Trained Model + Tokenizer"),
    code(
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        "model_name = 'bigscience/bloomz-560m'",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)",
        "foundation_model = AutoModelForCausalLM.from_pretrained(model_name)",
        "",
        "# BLOOM has no pad token by default — reuse the EOS token so batching works",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "print(f'Model: {model_name}')",
        "print(f'Number of parameters: {sum(p.numel() for p in foundation_model.parameters()):,}')",
    ),

    # ============================================================
    # 2. Load + preprocess dataset
    # ============================================================
    md(
        "## 2. Load and Preprocess the Quotes Dataset",
        "",
        "We take a **10% slice** of the `Abirate/english_quotes` training split, then",
        "tokenize the `quote` field in batches. The `Trainer` will turn each tokenized",
        "example into a causal-LM example via the data collator.",
    ),
    code(
        "from datasets import load_dataset",
        "",
        "# 10% sample of the training split",
        "data = load_dataset('Abirate/english_quotes', split='train[:10%]')",
        "print(f'Dataset size: {len(data)}')",
        "print('First example:')",
        "print(data[0])",
    ),
    code(
        "# Tokenize the quote text",
        "data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)",
        "",
        "# Show 5 examples so we can eyeball the tokenization output",
        "train_sample = data.select(range(5))",
        "for i, ex in enumerate(train_sample):",
        "    print(f'-- Example {i} --')",
        "    print('quote     :', ex['quote'])",
        "    print('input_ids :', ex['input_ids'][:20], '...' if len(ex['input_ids']) > 20 else '')",
        "    print('len       :', len(ex['input_ids']))",
        "    print()",
    ),

    # ============================================================
    # 3 + 4. LoRA config + apply
    # ============================================================
    md(
        "## 3. Configure LoRA and Wrap the Model",
        "",
        "LoRA hyperparameters explained:",
        "",
        "- **`r=4`**: rank of the low-rank update. Higher = more capacity, more params.",
        "  4 is a strong default for BLOOM-560M.",
        "- **`lora_alpha=32`**: scaling factor — the effective update is `(alpha / r)`",
        "  times the low-rank product. A common pattern is `alpha = 2*r` or `alpha = 8*r`.",
        "- **`target_modules=['query_key_value']`**: BLOOM packs Q, K and V into a",
        "  single linear layer named `query_key_value`. LoRA attaches there.",
        "- **`lora_dropout=0.05`**: regularization on the LoRA update.",
        "- **`bias='none'`**: do not train biases — LoRA only modifies the linear",
        "  projections.",
        "- **`task_type='CAUSAL_LM'`** tells PEFT the downstream task is causal LM.",
    ),
    code(
        "from peft import LoraConfig, get_peft_model",
        "",
        "lora_config = LoraConfig(",
        "    r=4,",
        "    lora_alpha=32,",
        "    target_modules=['query_key_value'],  # BLOOM's combined QKV projection",
        "    lora_dropout=0.05,",
        "    bias='none',",
        "    task_type='CAUSAL_LM',",
        ")",
        "",
        "peft_model = get_peft_model(foundation_model, lora_config)",
        "peft_model.print_trainable_parameters()",
    ),
    md(
        "**Reading the print.** PEFT prints something like `trainable params: 786,432 ||",
        "all params: 559M || trainable%: 0.14%`. That **0.14%** is the LoRA promise: we",
        "fine-tune a few hundred thousand parameters instead of half a billion, and the",
        "saved adapter is a few megabytes instead of gigabytes.",
    ),

    # ============================================================
    # 5. Training arguments + Trainer
    # ============================================================
    md(
        "## 4. Set Up Training and Fine-Tune",
        "",
        "We use a **causal-LM data collator** (`mlm=False`) so the labels are the",
        "input shifted by one position — the model learns to predict the next token.",
        "A high `learning_rate=3e-2` is fine for LoRA: only a few hundred thousand",
        "params move, and they start at zero.",
    ),
    code(
        "import transformers",
        "from transformers import TrainingArguments, Trainer",
        "",
        "output_directory = os.path.join('cache', 'peft_lab_outputs')",
        "",
        "training_args = TrainingArguments(",
        "    report_to='none',",
        "    output_dir=output_directory,",
        "    auto_find_batch_size=True,",
        "    learning_rate=3e-2,",
        "    num_train_epochs=3,",
        "    use_cpu=not torch.cuda.is_available(),  # automatic: use CPU if no CUDA",
        "    logging_steps=10,",
        "    save_strategy='no',  # we save manually after training",
        ")",
        "",
        "trainer = Trainer(",
        "    model=peft_model,",
        "    args=training_args,",
        "    train_dataset=data,",
        "    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),",
        ")",
        "",
        "trainer.train()",
    ),

    # ============================================================
    # 6. Save the LoRA adapter
    # ============================================================
    md(
        "## 5. Save the Fine-Tuned LoRA Adapter",
        "",
        "`save_pretrained` writes **only the LoRA weights and the adapter config**,",
        "not the 560-million-parameter base model. The result is a directory of a few",
        "megabytes that can be combined with any compatible BLOOMZ checkpoint at",
        "inference time.",
    ),
    code(
        "time_now = time.strftime('%Y%m%d_%H%M%S')",
        "peft_model_path = os.path.join(output_directory, f'peft_model_{time_now}')",
        "",
        "trainer.model.save_pretrained(peft_model_path)",
        "tokenizer.save_pretrained(peft_model_path)",
        "",
        "print(f'Saved LoRA adapter to: {peft_model_path}')",
        "for f in os.listdir(peft_model_path):",
        "    p = os.path.join(peft_model_path, f)",
        "    print(f'  - {f}  ({os.path.getsize(p) / 1024:.1f} KB)')",
    ),

    # ============================================================
    # 7. Reload + inference
    # ============================================================
    md(
        "## 6. Reload the LoRA Model and Generate Text",
        "",
        "At inference we reload the base BLOOMZ-560M, then attach our saved LoRA",
        "adapter with `PeftModel.from_pretrained(..., is_trainable=False)`. The",
        "result is a callable language model fine-tuned on quotes.",
    ),
    code(
        "from peft import PeftModel",
        "",
        "# Reload the base model fresh (so it has no LoRA attached yet)",
        "base_model_reloaded = AutoModelForCausalLM.from_pretrained(model_name)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "loaded_model = PeftModel.from_pretrained(",
        "    base_model_reloaded,",
        "    peft_model_path,",
        "    is_trainable=False,",  # frozen, inference only",
        ")",
        "loaded_model.to(device).eval()",
        "print('LoRA adapter reloaded and merged on top of the base model.')",
    ),
    code(
        "@torch.no_grad()",
        "def generate(prompt: str, model=loaded_model, max_new_tokens: int = 60,",
        "             temperature: float = 0.7, top_p: float = 0.9):",
        "    inputs = tokenizer(prompt, return_tensors='pt').to(device)",
        "    outputs = model.generate(",
        "        **inputs,",
        "        max_new_tokens=max_new_tokens,",
        "        do_sample=True,",
        "        temperature=temperature,",
        "        top_p=top_p,",
        "        repetition_penalty=1.2,",
        "        pad_token_id=tokenizer.eos_token_id,",
        "    )",
        "    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]",
        "",
        "",
        "prompts = [",
        "    'Two things are infinite: ',",
        "    'Be the change ',",
        "    'In three words I can sum up everything I have learned about life: ',",
        "    'The best way to predict the future ',",
        "]",
        "for p in prompts:",
        "    print(f'>>> {p}')",
        "    print(generate(p))",
        "    print()",
    ),

    # ============================================================
    # Optional: before vs after
    # ============================================================
    md(
        "## 7. (Bonus) Before vs After — Same Prompts on the Base Model",
        "",
        "Worth confirming the LoRA actually changed something. We generate the same",
        "prompts from the **base BLOOMZ-560M** (no LoRA) and compare.",
    ),
    code(
        "base_for_compare = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()",
        "",
        "@torch.no_grad()",
        "def base_generate(prompt: str, max_new_tokens: int = 60):",
        "    inputs = tokenizer(prompt, return_tensors='pt').to(device)",
        "    outputs = base_for_compare.generate(",
        "        **inputs,",
        "        max_new_tokens=max_new_tokens,",
        "        do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2,",
        "        pad_token_id=tokenizer.eos_token_id,",
        "    )",
        "    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]",
        "",
        "",
        "prompt = 'Two things are infinite: '",
        "print('BASE  :', base_generate(prompt))",
        "print('LoRA  :', generate(prompt))",
    ),

    md(
        "## Summary",
        "",
        "- We fine-tuned a **560 M-parameter LLM** while only training a small",
        "  fraction (often <0.2%) of its weights using **LoRA** + **PEFT**.",
        "- The saved adapter is just a few megabytes, completely separable from the",
        "  base model — the same base BLOOMZ can host many task-specific LoRA",
        "  adapters in production.",
        "- `Trainer` + `LoraConfig` make the workflow indistinguishable from full",
        "  fine-tuning, but at a tiny fraction of the cost. This is the foundation",
        "  of modern PEFT methods (LoRA, QLoRA, AdaLoRA, IA3) and is how most",
        "  open-source LLM fine-tuning is done today.",
        "",
        "**Next steps for production.**",
        "- Sweep `r`, `lora_alpha` and `learning_rate` on a held-out validation",
        "  split — these three knobs drive the quality/cost trade-off.",
        "- Add `eval_strategy='steps'` with a held-out portion of the dataset to",
        "  track convergence.",
        "- Try `target_modules=['query_key_value', 'dense', 'dense_h_to_4h',",
        "  'dense_4h_to_h']` to attach LoRA to the MLP projections too — usually",
        "  improves quality at a still-tiny parameter cost.",
        "- For 7B+ models, combine LoRA with **4-bit quantization** (QLoRA) so the",
        "  base model fits in consumer GPUs.",
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = os.path.join(os.path.dirname(__file__), "dailychallenge.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
