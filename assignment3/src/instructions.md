# Assignment 3 — Transformer Fine-tuning + Robustness + Limitations

**Weeks:** 6–8
**Topics:** Transfer learning, slice evaluation, responsible reporting

---

## 1. Overview

Fine-tune a pretrained Transformer for AG News classification, compare it against your best model from Assignment 2, and perform robustness/slice evaluation. Conclude with a limitations section grounded in evidence.

---

## 2. Learning Outcomes

- Fine-tune a pretrained Transformer encoder for classification.
- Evaluate beyond a single score using slice/robustness analyses.
- Write scientifically grounded limitations and failure-mode reporting.

---

## 3. Requirements

- **Fine-tune one pretrained model** (e.g., DistilBERT / BERT / RoBERTa) for 4-way classification.
- **Compare against best Assignment 2 model** (same splits, same metrics).
- **Report Accuracy + Macro-F1 + confusion matrix** for both.
- **Perform at least two robustness/slice evaluations** (see options below).
- **Include a short Limitations section** (½–1 page) based on results and examples.

---

## 4. Robustness / Slice Evaluation

Choose **at least TWO** from the following:

### Length Buckets
Evaluate performance by input length. Define bins and report per-bin metrics.

### Input Field Stress Test
Headline-only vs headline+description (if your AG News version supports both fields).

### Keyword Masking Probe
Mask a small list of class-indicative keywords and measure performance drop.

### Label-Noise Sensitivity (Simple)
Train with reduced training size (e.g., 25%, 50%, 100%) and compare trends.

### The one we chose: **Keyword Masking Probe** and **Label-Noise Sensitivity** (Simple)

- We wan't to see if the model relies heavily on a few keywords (which could indicate overfitting or lack of robustness)
- And how sensitive it is to training data size, which can reveal its learning efficiency and generalization capabilities. If we do it well, we could even calculate the "scaling law" for our model, which is a very interesting analysis to do.

---

## 5. Instructions

1. **Reuse the same dataset split** from Assignments 1–2.
2. **Fine-tune the Transformer** — document tokenizer, max length, LR, batch size, epochs, early stopping.
3. **Evaluate** your Transformer and your best neural model on the same test set.
4. **Run two slice evaluations** and report results clearly (tables/plots encouraged).
5. **Write limitations** grounded in observed failures + slice results.

---

## 6. Deliverables

### Final Report (5–6 pages, PDF)

- Best baseline summary (A1/A2) + Transformer setup
- Comparison table: baseline vs neural vs Transformer (dev + test)
- Robustness/slice evaluation results + interpretation
- Error analysis (≥10 examples) emphasizing harmful/important failures
- Limitations + suggested future improvements

### Code Repository

- Training + evaluation scripts
- Slice evaluation script/notebook
- Clear reproduction instructions

### One-Page Model Card (appendix or separate PDF)

- Intended use
- Metrics
- Known failures

---

## 7. Student Checklist

- [ ] Transformer fine-tuned and evaluated
- [ ] Comparison against best Assignment 2 model included
- [ ] ≥2 slice evaluations completed and reported
- [ ] Limitations section grounded in evidence
- [ ] Model Card included
- [ ] Repo reproduces main tables

---

## 8. Submission (Brightspace)

**Assignment 3 Submission Folder:**
- Final report PDF
- Model Card (PDF)
- Repo link + release tag/commit hash

---

## 9. Grading Rubric (100 points)

| Criterion | Points | What We Look For |
|---|---|---|
| Transformer fine-tuning + evaluation | 35 | Correct setup, appropriate dev usage, stable results, clear reporting of hyperparameters. |
| Comparison + conclusions | 20 | Fair comparison to best prior model; conclusions supported by evidence. |
| Robustness / slice evaluation | 20 | Two meaningful slices, clear methodology, correct interpretation. |
| Limitations + responsible reporting | 15 | Limitations grounded in results; discusses failure modes and potential impact. |
| Reproducibility + clarity | 10 | Clean repo, runnable instructions, readable tables/figures and writing. |

**Total:** 100 points

---

## 10. Due Date

6th of april
---

## Appendix A — Reproducibility Checklist

**Recommended for all reports:**

- [ ] Exact dataset version/source stated
- [ ] Train/dev/test sizes reported + seed reported
- [ ] All hyperparameters listed
- [ ] One command to reproduce main results
- [ ] Runtime notes (approximate) included