"""
Generate dailyxp.ipynb for Week 7 Day 4:
LoRA from scratch in PyTorch — six exercises culminating in fine-tuning
a frozen MLP on MNIST with only the LoRA layers trainable.
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
        "# Daily XP — LoRA from Scratch in PyTorch",
        "",
        "**Course:** Developers Institute  **Week 7 - Day 4**  ",
        "**Author:** Alex Goldbaum",
        "",
        "Six exercises that build **Low-Rank Adaptation (LoRA)** step by step:",
        "1. A `LoRALayer` with low-rank matrices A and B.",
        "2. A `LinearWithLoRA` wrapper that adds LoRA to any `nn.Linear`.",
        "3. Drop the wrapper into a small network and confirm outputs are unchanged.",
        "4. A merged-weights variant `LinearWithLoRAMerged` that is **mathematically",
        "   equivalent** but cheaper at inference.",
        "5. A 3-layer MLP with every `Linear` replaced by a merged LoRA layer.",
        "6. Freeze the original linear weights and **fine-tune only LoRA** on MNIST,",
        "   then compare against the full-parameter baseline.",
        "",
        "Recommended: run on a Colab **GPU** runtime — MNIST is small but the LoRA",
        "loop still benefits from CUDA.",
    ),

    md("## Setup"),
    code(
        "%pip install -qU torch torchvision",
    ),
    code(
        "import copy",
        "import time",
        "",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from torchvision import datasets, transforms",
        "from torch.utils.data import DataLoader",
        "",
        "RANDOM_SEED = 123",
        "torch.manual_seed(RANDOM_SEED)",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print('Device:', DEVICE)",
    ),

    # ============================================================
    # Exercise 1: LoRALayer
    # ============================================================
    md(
        "## Exercise 1 — `LoRALayer`",
        "",
        "LoRA learns a **low-rank update** to a pretrained linear layer: instead of",
        "fine-tuning the full `(in_dim × out_dim)` weight matrix, we add the product",
        "of two small matrices `A (in_dim × rank)` and `B (rank × out_dim)`, scaled by",
        "`alpha`. `A` is initialised with small random values and `B` with zeros so",
        "that **at initialisation the LoRA update is exactly zero** — the model",
        "starts indistinguishable from the original.",
    ),
    code(
        "class LoRALayer(nn.Module):",
        "    def __init__(self, in_dim, out_dim, rank, alpha):",
        "        super().__init__()",
        "        std_dev = 1.0 / torch.sqrt(torch.tensor(rank).float())",
        "        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)",
        "        self.B = nn.Parameter(torch.zeros(rank, out_dim))",
        "        self.alpha = alpha",
        "",
        "    def forward(self, x):",
        "        # x: (batch, in_dim) -> (batch, out_dim)",
        "        return self.alpha * (x @ self.A @ self.B)",
        "",
        "",
        "# Quick sanity check",
        "torch.manual_seed(RANDOM_SEED)",
        "lora = LoRALayer(in_dim=10, out_dim=5, rank=2, alpha=8)",
        "x = torch.randn(3, 10)",
        "y = lora(x)",
        "print(f'input shape : {x.shape}')",
        "print(f'output shape: {y.shape}')",
        "print('output should be all zeros at init (B is zeros):', torch.allclose(y, torch.zeros_like(y)))",
    ),

    # ============================================================
    # Exercise 2: LinearWithLoRA
    # ============================================================
    md(
        "## Exercise 2 — `LinearWithLoRA`",
        "",
        "Wrap an existing `nn.Linear` with a `LoRALayer`. The forward pass returns",
        "the **sum** of the original linear transformation and the LoRA adaptation —",
        "so before any training, the wrapper is functionally identical to the bare",
        "linear layer.",
    ),
    code(
        "class LinearWithLoRA(nn.Module):",
        "    def __init__(self, linear, rank, alpha):",
        "        super().__init__()",
        "        self.linear = linear",
        "        self.lora = LoRALayer(",
        "            in_dim=linear.in_features,",
        "            out_dim=linear.out_features,",
        "            rank=rank,",
        "            alpha=alpha,",
        "        )",
        "",
        "    def forward(self, x):",
        "        return self.linear(x) + self.lora(x)",
        "",
        "",
        "# Sanity test",
        "torch.manual_seed(RANDOM_SEED)",
        "base = nn.Linear(in_features=10, out_features=5)",
        "wrap = LinearWithLoRA(base, rank=2, alpha=8)",
        "x = torch.randn(3, 10)",
        "print('Original output :', base(x))",
        "print('Wrapped output  :', wrap(x))",
        "print('Match at init?   ', torch.allclose(base(x), wrap(x)))",
    ),

    # ============================================================
    # Exercise 3
    # ============================================================
    md(
        "## Exercise 3 — Apply LoRA to a Single Linear Layer",
        "",
        "Define a one-layer network, then wrap it with `LinearWithLoRA`. We expect",
        "**identical outputs** before any training because the LoRA term is zero at",
        "init.",
    ),
    code(
        "torch.manual_seed(RANDOM_SEED)",
        "layer = nn.Linear(in_features=10, out_features=2)",
        "x = torch.randn(1, 10)",
        "",
        "print('x       :', x)",
        "print('layer   :', layer)",
        "print('Original output:', layer(x))",
        "",
        "# Wrap with LoRA",
        "layer_lora_1 = LinearWithLoRA(layer, rank=2, alpha=4)",
        "print('LoRA output    :', layer_lora_1(x))",
        "print('Outputs match (expected True at init):',",
        "      torch.allclose(layer(x), layer_lora_1(x)))",
    ),

    # ============================================================
    # Exercise 4: Merged
    # ============================================================
    md(
        "## Exercise 4 — `LinearWithLoRAMerged`",
        "",
        "Equivalent to `LinearWithLoRA`, but instead of running two separate matmuls",
        "we **merge** the LoRA delta into a single combined weight matrix and call",
        "`F.linear` once. This is the **deployment form** of LoRA: after fine-tuning,",
        "merging means the adapted model has the same inference cost as the original.",
        "",
        "Math: `y = x W^T + alpha · x A B = x (W + alpha · (A B)^T)^T = F.linear(x, W + alpha · (AB)^T, b)`.",
    ),
    code(
        "class LinearWithLoRAMerged(nn.Module):",
        "    def __init__(self, linear, rank, alpha):",
        "        super().__init__()",
        "        self.linear = linear",
        "        self.lora = LoRALayer(",
        "            linear.in_features, linear.out_features, rank, alpha,",
        "        )",
        "",
        "    def forward(self, x):",
        "        lora = self.lora.A @ self.lora.B  # (in_dim, out_dim)",
        "        combined_weight = self.linear.weight + self.lora.alpha * lora.T",
        "        return F.linear(x, combined_weight, self.linear.bias)",
        "",
        "",
        "torch.manual_seed(RANDOM_SEED)",
        "base = nn.Linear(10, 2)",
        "merged = LinearWithLoRAMerged(base, rank=2, alpha=4)",
        "wrap = LinearWithLoRA(base, rank=2, alpha=4)",
        "x = torch.randn(1, 10)",
        "",
        "print('Bare linear     :', base(x))",
        "print('LinearWithLoRA  :', wrap(x))",
        "print('LinearWithMerged:', merged(x))",
        "print('Wrap == merged at init?', torch.allclose(wrap(x), merged(x)))",
    ),
    md(
        "Both forms produce the same numbers at initialisation. Once we train them,",
        "they will diverge from the base linear by the same amount — but the merged",
        "form's forward pass is **a single matmul** instead of two, so it is the",
        "preferred shape once you are done fine-tuning.",
    ),

    # ============================================================
    # Exercise 5: MLP with LoRA on every layer
    # ============================================================
    md(
        "## Exercise 5 — 3-Layer MLP with LoRA on Every Layer",
        "",
        "Define a standard 3-layer MLP for MNIST classification (784 → 128 → 64 → 10),",
        "then walk through `model.layers` replacing every `nn.Linear` with",
        "`LinearWithLoRAMerged`. The print confirms the structural change.",
    ),
    code(
        "class MultilayerPerceptron(nn.Module):",
        "    def __init__(self, num_features, num_hidden_1, num_hidden_2, num_classes):",
        "        super().__init__()",
        "        self.layers = nn.Sequential(",
        "            nn.Linear(num_features, num_hidden_1),",
        "            nn.ReLU(),",
        "            nn.Linear(num_hidden_1, num_hidden_2),",
        "            nn.ReLU(),",
        "            nn.Linear(num_hidden_2, num_classes),",
        "        )",
        "",
        "    def forward(self, x):",
        "        return self.layers(x)",
        "",
        "",
        "num_features = 784   # 28 * 28 MNIST pixels (flattened)",
        "num_hidden_1 = 128",
        "num_hidden_2 = 64",
        "num_classes  = 10",
        "",
        "learning_rate = 5e-3",
        "num_epochs = 2",
        "",
        "torch.manual_seed(RANDOM_SEED)",
        "model = MultilayerPerceptron(",
        "    num_features=num_features,",
        "    num_hidden_1=num_hidden_1,",
        "    num_hidden_2=num_hidden_2,",
        "    num_classes=num_classes,",
        ").to(DEVICE)",
        "",
        "optimizer_pretrained = torch.optim.Adam(model.parameters(), lr=learning_rate)",
        "print(model)",
    ),

    # ============================================================
    # Loading dataset
    # ============================================================
    md("## Loading the MNIST Dataset"),
    code(
        "BATCH_SIZE = 64",
        "",
        "train_dataset = datasets.MNIST(",
        "    root='data', train=True, transform=transforms.ToTensor(), download=True,",
        ")",
        "test_dataset = datasets.MNIST(",
        "    root='data', train=False, transform=transforms.ToTensor(), download=True,",
        ")",
        "",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)",
        "test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)",
        "",
        "for images, labels in train_loader:",
        "    print('Image batch dimensions:', images.shape)",
        "    print('Image label dimensions:', labels.shape)",
        "    break",
    ),

    # ============================================================
    # Eval + training helpers
    # ============================================================
    md("## Evaluation helper"),
    code(
        "def compute_accuracy(model, data_loader, device):",
        "    model.eval()",
        "    correct_pred, num_examples = 0, 0",
        "    with torch.no_grad():",
        "        for features, targets in data_loader:",
        "            features = features.view(-1, num_features).to(device)",
        "            targets = targets.to(device)",
        "            logits = model(features)",
        "            _, predicted_labels = torch.max(logits, 1)",
        "            num_examples += targets.size(0)",
        "            correct_pred += (predicted_labels == targets).sum().item()",
        "    return 100.0 * correct_pred / num_examples",
    ),
    md("## Training loop"),
    code(
        "def train(num_epochs, model, optimizer, train_loader, device):",
        "    start_time = time.time()",
        "    for epoch in range(num_epochs):",
        "        model.train()",
        "        for batch_idx, (features, targets) in enumerate(train_loader):",
        "            features = features.view(-1, num_features).to(device)",
        "            targets = targets.to(device)",
        "",
        "            logits = model(features)",
        "            loss = F.cross_entropy(logits, targets)",
        "",
        "            optimizer.zero_grad()",
        "            loss.backward()",
        "            optimizer.step()",
        "",
        "            if not batch_idx % 400:",
        "                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | '",
        "                      f'Batch {batch_idx:03d}/{len(train_loader)} | Loss: {loss.item():.4f}')",
        "",
        "        with torch.set_grad_enabled(False):",
        "            acc = compute_accuracy(model, train_loader, device)",
        "        print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}  training accuracy: {acc:.2f}%')",
        "        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')",
        "    print(f'Total training time: {(time.time() - start_time) / 60:.2f} min')",
    ),

    md("## Train the baseline MLP (full fine-tuning)"),
    code(
        "train(num_epochs, model, optimizer_pretrained, train_loader, DEVICE)",
        "print(f'\\nTest accuracy (baseline MLP): {compute_accuracy(model, test_loader, DEVICE):.2f}%')",
    ),

    # ============================================================
    # Replace linear layers with LoRA
    # ============================================================
    md(
        "## Replace Linear Layers with LoRA (rank=4, alpha=8)",
        "",
        "Deep-copy the trained baseline so we keep it for comparison, then surgically",
        "replace `layers[0]`, `layers[2]`, `layers[4]` (the three `nn.Linear`s — index",
        "1 and 3 are the `ReLU` activations).",
    ),
    code(
        "model_lora = copy.deepcopy(model)",
        "",
        "model_lora.layers[0] = LinearWithLoRAMerged(model_lora.layers[0], rank=4, alpha=8)",
        "model_lora.layers[2] = LinearWithLoRAMerged(model_lora.layers[2], rank=4, alpha=8)",
        "model_lora.layers[4] = LinearWithLoRAMerged(model_lora.layers[4], rank=4, alpha=8)",
        "",
        "model_lora.to(DEVICE)",
        "print(model_lora)",
        "",
        "print(f'\\nTest accuracy orig model: {compute_accuracy(model, test_loader, DEVICE):.2f}%')",
        "print(f'Test accuracy LoRA model: {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')",
    ),
    md(
        "**Sanity check.** The two accuracies should be **identical** because we have",
        "not trained the LoRA matrices yet — `B = 0` makes the LoRA contribution",
        "exactly zero. The LoRA model is a drop-in replacement at init.",
    ),

    # ============================================================
    # Exercise 6: Freeze + fine-tune LoRA only
    # ============================================================
    md(
        "## Exercise 6 — Freeze Linear Layers, Fine-Tune Only LoRA",
        "",
        "Walk the module tree and set `requires_grad=False` on every `nn.Linear`.",
        "Anything else (notably the `LoRALayer`'s `A` and `B`) stays trainable. After",
        "freezing we print every parameter's `requires_grad` flag to verify.",
    ),
    code(
        "def freeze_linear_layers(model):",
        "    for child in model.children():",
        "        if isinstance(child, nn.Linear):",
        "            for param in child.parameters():",
        "                param.requires_grad = False",
        "        else:",
        "            freeze_linear_layers(child)",
        "",
        "",
        "freeze_linear_layers(model_lora)",
        "",
        "print('Trainable status per parameter:')",
        "for name, param in model_lora.named_parameters():",
        "    print(f'  {name}: requires_grad={param.requires_grad}  shape={tuple(param.shape)}')",
        "",
        "n_total = sum(p.numel() for p in model_lora.parameters())",
        "n_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)",
        "print(f'\\nTotal params    : {n_total:,}')",
        "print(f'Trainable params: {n_trainable:,}  ({n_trainable / n_total * 100:.2f}%)')",
    ),
    md(
        "Only the LoRA `A` and `B` matrices show `True` — the original linear",
        "weights and biases are frozen. The trainable-parameter count drops",
        "dramatically (~3-5% of the original MLP), which is exactly the LoRA",
        "selling point: orders of magnitude less storage per adapter.",
    ),
    md("## Fine-tune only the LoRA parameters"),
    code(
        "optimizer_lora = torch.optim.Adam(",
        "    [p for p in model_lora.parameters() if p.requires_grad],",
        "    lr=learning_rate,",
        ")",
        "",
        "train(num_epochs, model_lora, optimizer_lora, train_loader, DEVICE)",
        "print(f'\\nTest accuracy LoRA finetune: {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')",
    ),
    code(
        "# Final comparison",
        "print('=== Final comparison ===')",
        "print(f'Test accuracy orig model       : {compute_accuracy(model, test_loader, DEVICE):.2f}%')",
        "print(f'Test accuracy LoRA-finetuned   : {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')",
        "print(f'Total params (orig)            : {sum(p.numel() for p in model.parameters()):,}')",
        "print(f'Trainable params during LoRA FT: {sum(p.numel() for p in model_lora.parameters() if p.requires_grad):,}')",
    ),

    md(
        "## Summary",
        "",
        "- **`LoRALayer`** holds two small matrices `A` and `B`; the update at init is",
        "  zero (`B` is zeros), so layered into a pretrained model it does not change",
        "  its behaviour until trained.",
        "- **`LinearWithLoRA`** adds the LoRA term *in parallel* with the original",
        "  linear (two matmuls). **`LinearWithLoRAMerged`** rolls the LoRA delta into",
        "  a single combined weight (one matmul) — same math, lower inference cost.",
        "- Replacing every `Linear` in an MLP with `LinearWithLoRAMerged` and",
        "  freezing the originals leaves only a *tiny* fraction of parameters",
        "  trainable, which is the whole point of LoRA: **adapt a large model at a",
        "  fraction of the cost**.",
        "- On MNIST the LoRA-only fine-tune recovers (and sometimes matches) the",
        "  fully-trainable model accuracy — at a tiny number of new parameters,",
        "  cheap to store and easy to swap. The same trick scales to LLM fine-tuning,",
        "  which is why LoRA / QLoRA dominate adapter-based training today.",
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

out_path = os.path.join(os.path.dirname(__file__), "dailyxp.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
print(f"Cells: {len(cells)}")
