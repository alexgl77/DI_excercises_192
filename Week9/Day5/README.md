# Mini Project Day - Meta-Analysis of Recent LLM Research Papers

This folder contains the mini project for Week 9, Day 5.

## Files

- `miniprojectday.ipynb` - Colab-compatible notebook with the report and a PDF generation cell.
- `meta_analysis_llms.pdf` - final 2025-2026 report to submit.
- `meta_analysis_llms.md` - editable Markdown source of the report.
- `fetch_sources.py` - Python script that fetches paper metadata from arXiv, ACL Anthology, and OpenReview.
- `papers_sources.json` - source metadata produced by `fetch_sources.py`.
- `papers_sources.csv` - table version of the fetched source metadata.
- `generate_pdf.py` - local helper script used to generate the PDF from the report content.

## Where is the code?

The project is a written meta-analysis, so the analysis itself is written in:

- `meta_analysis_llms.md`
- `miniprojectday.ipynb`

The Python code that generates the final PDF from that analysis is:

- `generate_pdf.py`

The Python code that fetches/validates the paper sources is:

- `fetch_sources.py`

In Google Colab, open `miniprojectday.ipynb` and run the last code cell. That cell installs `reportlab`, downloads the project files from GitHub if needed, runs `fetch_sources.py` to create the source metadata files, and then runs `generate_pdf.py` to create `meta_analysis_llms.pdf`.

## Google Colab link

After pushing this folder to GitHub, submit this Colab link:

https://colab.research.google.com/github/alexgl77/DI_excercises_192/blob/main/Week9/Day5/miniprojectday.ipynb

If the repository name or branch changes, use this pattern:

`https://colab.research.google.com/github/<YOUR_USERNAME>/<YOUR_REPO>/blob/<BRANCH>/Week9/Day5/miniprojectday.ipynb`

## Papers Used

1. DeepSeek-AI et al. (2025/2026), "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", Nature 2025 / arXiv v2 2026.  
   https://arxiv.org/abs/2501.12948

2. Kimi Team et al. (2025), "Kimi k1.5: Scaling Reinforcement Learning with LLMs".  
   https://arxiv.org/abs/2501.12599

3. Muennighoff et al. (2025), "s1: Simple test-time scaling", EMNLP 2025.  
   https://aclanthology.org/2025.emnlp-main.1025/

4. Ye et al. (2025), "LIMO: Less is More for Reasoning", COLM 2025.  
   https://openreview.net/forum?id=T2TZ0RY4Zk

5. Wan et al. (2026), "Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning", ICLR 2026.  
   https://openreview.net/forum?id=RduOiisl1S

## Suggested Git Commands

From the repository root:

```bash
git add "Week9/Day5"
git commit -m "Add Week9 Day5 LLM meta-analysis mini project"
git push origin main
```
