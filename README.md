# AuditScribe

Autoregressive smart-contract audit text generation using Simple RNN, Bidirectional RNN, LSTM and GRU.

## Problem statement and objective

Smart-contract audit reports contain specialized language describing vulnerabilities, impacts and mitigations. This project investigates how four recurrent neural architectures learn that language through next-word prediction and sequential generation. The objective is to compare their predictive performance, computational cost and generated-text quality under comparable experimental conditions, then examine how sampling temperature changes coherence, diversity, contextual consistency and repetition.

The system models audit prose. It does not inspect Solidity contracts or verify security findings.

## Dataset description and source

Dataset: a curated corpus of public smart-contract audit findings from the [Solodit Content Repository](https://github.com/solodit/solodit_content), pinned to [revision 4d5330d12978c7a8fc2aab5b171112784fb671bd](https://github.com/solodit/solodit_content/tree/4d5330d12978c7a8fc2aab5b171112784fb671bd).

The processed corpus contains **483 reports, 5,140 sections and 630,352 tokens**. It spans multiple auditors, protocols, vulnerability classes and severity levels. Its heterogeneous formatting and repeated audit terminology create both useful domain structure and preprocessing challenges. The [source manifest](results/source_manifest.json) records original URLs, SHA-256 hashes, auditor names, groups and split assignments. Dataset rights and attribution are described in [DATASET_NOTICE.md](DATASET_NOTICE.md).

| Split | Reports | Protocol groups | Tokens | Input-target windows | Unknown-token share |
|---|---:|---:|---:|---:|---:|
| Training | 387 | 294 | 501,620 | 136,015 | 3.47% |
| Validation | 48 | 37 | 76,859 | 10,000 | 5.78% |
| Test | 48 | 37 | 51,873 | 10,000 | 5.79% |

## Preprocessing and autoregressive sequences

Code blocks, URLs, images, HTML tags and tables are removed. Inline code becomes `identifier`. Text is lowercased; numeric strings are removed; English words and six punctuation marks are retained. Short lines, sections below 30 tokens and exact duplicate cleaned sections are excluded.

Reports are grouped by a coarse protocol-name heuristic before an 80/10/10 group split with seed 42. This reduces overlap from re-audits but cannot guarantee separation of all related protocol families. The training-only vocabulary contains **4,000 tokens**, including `<unk>`. Each input contains **24 preceding tokens** and its target is the next token. Windows advance by three positions and never cross section boundaries. All training windows are used; validation and test are each sampled deterministically to 10,000 windows.

This is word-level language modelling with punctuation as separate tokens. During generation, a predicted token is appended to the prompt and the process repeats using the latest 24 tokens. See [methodology](docs/methodology.md) for the exact pipeline and evaluation definitions.

## Models implemented

| Model | Mechanism | Context representation |
|---|---|---|
| Simple RNN | Tanh recurrent state | Final left-to-right hidden state |
| Bidirectional RNN | Forward and backward tanh recurrences | Concatenated final states over the known prefix |
| LSTM | Input, forget and output gates with a cell state | Final hidden state |
| GRU | Reset and update gates | Final hidden state |

Every model uses a 64-dimensional embedding, one recurrent layer with 96 hidden units per direction, dropout 0.25 and a vocabulary classifier. Bidirectional RNN reads both directions **within the preceding context only**; the target token is never part of its input. Model parameter counts are reported because equal hidden width does not imply equal capacity.

## Comparable experimental conditions

All four models share the dataset, vocabulary, context length, training seed (42), batch-order seed, **16 epochs**, batch size **512**, AdamW optimizer, initial learning rate **0.0015**, weight decay **0.0001**, and gradient clipping at norm **1.0**. The same validation-driven ReduceLROnPlateau rule halves learning rate after two non-improving epochs, down to 0.0001. The recorded learning-rate trajectories can differ because validation behaviour differs by architecture.

The checkpoint with minimum validation cross entropy is selected for each model. Test data is not used for checkpoint selection. The final experiment uses all available training windows and stronger regularization than the preliminary run, while retaining the same vocabulary and validation/test examples. A single training seed limits statistical conclusions.

## Execution steps and dependencies

Python 3.12 and Git are required. The recorded run used PyTorch 2.14.1+cpu on CPU with two threads. Required Python packages are listed in [requirements.txt](requirements.txt). No API keys, wallet or hosted database are needed.

```bash
git clone https://github.com/themisogynist/auditscribe-web3-textgen.git
cd auditscribe-web3-textgen
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell activation instead:
# .venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m auditscribe.data --download
python -m unittest discover -s tests -v
python -m auditscribe.train
python -m auditscribe.analyze
python -m auditscribe.report
python -m auditscribe.generate --model gru --temperature 0.5 --prompt "the contract does not validate"
```

Run commands from the repository root. Dependency installation and the first dataset acquisition require internet access. Training regenerates the local checkpoints and recorded outputs. [AuditScribe.ipynb](AuditScribe.ipynb) provides a notebook view of the experiment and its saved results; Jupyter is optional and can be installed with `python -m pip install notebook`.

SQLite stores document provenance, cleaned passages, run configuration, metrics and generated text at `data/processed/auditscribe.sqlite`. NumPy arrays feed the minibatches. Downloaded report text, database files, arrays and checkpoints are excluded from the public repository; the commands above recreate them.

## Results obtained

| Model | Test loss | Perplexity | Prediction accuracy | Training time (s) |
|---|---:|---:|---:|---:|
| RNN | 4.7873 | 119.97 | 19.89% | 142.7 |
| BIRNN | 4.8832 | 132.06 | 18.90% | 245.2 |
| LSTM | 4.7365 | 114.03 | 20.43% | 267.1 |
| GRU | 4.7279 | 113.06 | 20.74% | 254.4 |

The smoothed unigram baseline achieves **7.18% accuracy**, **5.8556 loss** and **349.19 perplexity**. All four recurrent models outperform it. Primary accuracy includes punctuation and unknown-token targets; [RESULTS.md](RESULTS.md) also reports known-target and lexical-word accuracy. Lower loss/perplexity and higher accuracy indicate better next-token prediction on the fixed test set.

## Model comparison and evaluation

GRU obtains the lowest validation loss (4.6892), lowest test loss (4.7279), lowest perplexity (113.06) and highest accuracy (20.74%). LSTM is close at 20.43% accuracy and 114.03 perplexity. Simple RNN is fastest, while Bidirectional RNN has the most parameters and the weakest test performance in this run. GRU uses fewer parameters and less measured training time than LSTM.

The comparison includes model size, selected epoch, validation/test loss, perplexity, prediction accuracy and training time. Timing sums all minibatch training across 16 epochs and excludes validation and sampling; fit wall time is recorded separately. See [full results](RESULTS.md), [metrics CSV](results/metrics.csv), [training history](results/history.csv) and [experiment configuration](results/config.json).

![Training and validation loss](results/learning_curves.png)
![Model performance](results/model_comparison.png)

## Sampling-temperature analysis

Every model generates from the **same three seed sequences**:

1. `the contract does not validate`
2. `an attacker can manipulate the`
3. `to mitigate this vulnerability`

Temperatures are **0.5, 1.0 and 1.5**. Each prompt/temperature/model combination uses generation seeds 101, 202 and 303 and a 60-token continuation, yielding **108 outputs**. Sampling uses softmax(logits / temperature), suppresses `<unk>` and applies no top-k or top-p filter.

| Temperature | Coherence and contextual consistency | Diversity and repetition |
|---|---|---|
| 0.5 | More recognizable audit phrases, but incomplete explanations and generic templates | Mean distinct-bigram ratios 0.678-0.812; repeated-trigram fractions 0.075-0.167 |
| 1.0 | Broader vocabulary with grammatical errors and topic drift | Distinct-bigram ratios 0.983-0.996; little repeated-trigram content |
| 1.5 | Fragmented syntax and weak topic continuity | Distinct-bigram ratios at least 0.996; no repeated trigrams in these samples |

These observations distinguish lexical novelty from semantic quality. Temperature 0.5 is the most useful tested setting for recognizable short fragments, although none produces reliable technical reports. [Qualitative analysis](docs/qualitative_analysis.md) compares coherence, diversity, contextual consistency and repetition for all four models. [Temperature metrics](results/temperature_summary.csv) provide the numerical results.

![Temperature effects](results/temperature_analysis.png)

## Generated-text samples

**Complete model outputs:** [plain text](results/generated_samples.txt), [readable Markdown](results/generated_samples.md), [CSV](results/samples.csv) and [JSON](results/samples.json). Each record identifies model, prompt, temperature and random seed. Outputs are retained without correction or cherry-picking.

For example, with prompt `the contract does not validate`, Simple RNN at temperature 0.5 and seed 101 begins: `the owner is not checked in the contract`. This locally relevant opening subsequently drifts into repeated function/identifier phrases; the complete output is available in the sample files.

## Conclusion

**GRU is the best-performing model for this experiment.** It leads the validation and test likelihood measures and prediction accuracy. At temperature 0.5, it has the lowest mean repeated-trigram fraction and ties Simple RNN for the highest distinct-bigram ratio. LSTM is a close quantitative alternative but is more repetitive at low temperature in the sampled outputs.

This conclusion applies to the selected corpus, configuration and single training seed. It does not establish universal architectural superiority. Generated text remains grammatically imperfect and may be technically false; the project evaluates autoregressive language modelling, not security-audit correctness.

## Repository organization

| Location | Contents |
|---|---|
| `auditscribe/` | Dataset preparation, four model architectures, training, generation, analysis and report generation |
| `results/` | Metrics, all generated samples, plots, configuration, dataset statistics and source manifest |
| `docs/` | Methodology and qualitative comparison |
| `tests/` | Preprocessing, split integrity, model shape/gradient and output-consistency checks |
| `AuditScribe.ipynb` | Notebook presentation of the experiment and recorded results |
| `requirements.txt` | Core dependencies |
| `DATASET_NOTICE.md` | Dataset attribution and redistribution terms |
