# Efficient-HA

**Experimental codebase used in Ke Xu's research participation in SAKED: vision-language hallucination evaluation and decoding interventions.**

[Paper: SAKED](https://arxiv.org/abs/2602.09825) · [Decoding implementation](utils/deco_greedy.py) · [Experiment configurations](configs/deco) · [Recorded experiments](opera_log)

SAKED studies how instability across attention heads, model layers, and decoding tokens relates to hallucinations in large vision-language models, and uses stability-aware knowledge-enhanced decoding to improve generation. This repository documents my experimental work within that research: baseline reproduction, decoding modifications, and model evaluation. It includes research adaptations and development artifacts; it is not a claim that every implementation here is the SAKED algorithm or that every paper result can be reproduced from this branch.

## Research contributions

- **Evaluation workflow:** experiments with Gemma-3, Qwen-VL, and LLaVA, connecting response generation, POPE/CHAIR scoring, and structured prediction/metric logs.
- **Adaptive decoding:** improvements to an existing decoding method, including a self-implemented entropy-based alpha schedule that changes the contribution of early-layer logits during generation.
- **Candidate selection:** modifications to candidate-token filtering rules and thresholds, with confidence checks and intermediate diagnostics.
- **Baseline reproduction:** OPERA/DeCo experiments on LLaVA and MiniGPT-4 as part of the research work. The current top-level runners expose the methods listed below; an OPERA or MiniGPT-4 command should not be inferred from the historical scope alone.

The implementation builds on existing methods and third-party software. Personal contribution here means the experimental work and adaptations above, not sole authorship of the entire codebase or authorship of the SAKED paper.

## What is in this branch?

The default working branch is **`my-fix`**.

| Component | Location | Purpose |
|---|---|---|
| Caption generation | `generate_response_chair.py` | Generate COCO captions; periodically save results and skip completed image IDs |
| Caption evaluation | `chair.py` | Compute CHAIRs, CHAIRi, object recall, and length statistics |
| Object-presence evaluation | `generate_response_pope.py` | Run random/popular/adversarial POPE subsets and save predictions and classification metrics |
| Decoding interventions | `utils/deco_greedy.py` | Candidate selection, early-layer selection, alpha scheduling, and logit fusion |
| Model configurations | `configs/deco/` | Shared defaults and model-specific overrides |
| Logging | `utils/logger_utils.py` | Run summaries and diagnostic output |
| Experiment scripts | `scripts/chair_bash/`, `scripts/pope_bash/` | Historical batch-run examples; review paths before use |
| Results | `opera_log/` | Saved responses and metrics from experiments |
| Local Transformers checkout | `transformers/` | Editable library used by the experimental code |
| Additional tasks | `generate_response_AMBER.py`, `generate_response_mme.py` | Additional research entry points, outside the walkthrough below |

### Model coverage

Saved artifacts include Qwen2.5-VL-3B-Instruct, Gemma-3-4B-IT, Gemma-3n variants, and LLaVA-1.5-7B. These records provide experiment provenance, not a guarantee that every model/method combination works unchanged in a fresh environment.

The current runners use Hugging Face auto classes, chat templates, and `qwen_vl_utils` preprocessing. Check the processor interface and local Transformers implementation when changing model families. PaliGemma and Flickr8k were mentioned in an earlier README; this walkthrough does not present them as validated evaluation paths.

## Setup

The experiments target **Linux with an NVIDIA GPU**. The environment snapshot specifies Python 3.10 and a PyTorch 2.7 / CUDA 12.6 stack. GPU memory requirements depend on model, image resolution, generation length, and batch size.

```bash
git clone --branch my-fix https://github.com/mayflower6157/Efficient-HA.git
cd Efficient-HA

mamba env create -f environment.yml
mamba activate EHA-mayflower
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip install qwen-vl-utils nltk
python -m pip install -e ./transformers
```

Install the local Transformers checkout **after** the requirements so the experimental code uses it rather than an unrelated package release. The requirements include environment-specific CUDA pins and are not a portable lockfile; resolve any dependency conflicts for the target machine before running experiments.

The caption loader explicitly requests `flash_attention_2`. Install a compatible FlashAttention build for the chosen Python, PyTorch, CUDA, and ABI combination. The previous README's CUDA 12.4 wheel should not be copied blindly into a CUDA 12.6 environment. See the [FlashAttention project](https://github.com/Dao-AILab/flash-attention) for installation guidance.

For CHAIR text processing:

```bash
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4
```

Check the selected runtime:

```bash
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda); print(transformers.__file__); print(torch.cuda.is_available())"
```

Model checkpoints may require access approval or authentication through the model provider. Obtain the corresponding access before launching a run.

## Data preparation

The walkthrough uses COCO 2014:

```text
/path/to/coco2014/
├── val2014/
│   └── COCO_val2014_000000xxxxxx.jpg
└── annotations/
    ├── captions_train2014.json
    ├── captions_val2014.json
    ├── instances_train2014.json
    └── instances_val2014.json
```

- Download the images and annotations from [COCO](https://cocodataset.org/#download).
- POPE question files are under `pope_coco/`: random, popular, and adversarial.
- CHAIR scoring uses both training and validation caption/instance annotations to build its object reference.
- Run commands from the repository root because configurations and POPE paths are relative to it.

**CHAIR sample-list prerequisite:** `process_json()` in `generate_response_chair.py` currently hard-codes an absolute path ending in `opera_log/llava-1.5/greedy.jsonl`. Change that `input_jsonl` assignment to your local sample list before running. It must be JSONL with an integer `image_id` in each record, for example:

```json
{"image_id": 397133}
```

The generator uses those IDs to locate COCO validation images; it does not scan every image in `--datapath`. Keep the same sample list across methods.

## Run POPE

Start with a small batch size and a fresh output directory:

```bash
python generate_response_pope.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --datapath /path/to/coco2014/val2014 \
  --pope_type random \
  --method greedy \
  --batch_size 1 \
  --max_tokens 8 \
  --device cuda:0 \
  --output outputs/pope/qwen25vl3b/random/greedy/
```

For the adapted DeCo path, use `--method deco --early_exit_layers 10` and a different output directory. Repeat with `--pope_type popular` and `--pope_type adversarial` for the other subsets.

Output files include:

- `POPE_type_<subset>_<method>_detailed.jsonl`: image, question, response, predicted label, reference label, and correctness.
- `POPE_type_<subset>_<method>_metric.json`: accuracy, confusion matrix, per-class precision/recall/F1, and sample count.

The current answer parser labels responses containing negative expressions as “no” and others as “yes”; inspect ambiguous responses when interpreting scores. `--resume_if_exists` resumes by existing record count, so reuse it only with the same ordered dataset, batch size, and configuration.

## Generate captions and compute CHAIR

After fixing the sample-list path described above:

```bash
python generate_response_chair.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --datapath /path/to/coco2014/val2014 \
  --method greedy \
  --device cuda:0 \
  --max_tokens 64 \
  --output outputs/chair/qwen25vl3b/greedy/responses.json
```

The caption runner has branches for `greedy`, `beam`, `dola`, `deco`, and `vcd`. Availability in the dispatcher does not establish equal compatibility across model families. Use a separate process and output path for each experiment: decoding helpers patch generation methods at runtime.

Score the generated captions:

```bash
python chair.py \
  --cap_file outputs/chair/qwen25vl3b/greedy/responses.json \
  --image_id_key image_id \
  --caption_key response \
  --coco_path /path/to/coco2014/annotations \
  --cache outputs/chair/coco2014-cache.pkl \
  --save_path outputs/chair/qwen25vl3b/greedy/metrics.json
```

The cache is created if it does not exist. An existing cache takes precedence over `--coco_path`; use a fresh cache when changing annotation sources.

- **CHAIRs:** fraction of captions containing at least one hallucinated object.
- **CHAIRi:** fraction of mentioned objects that are hallucinated.
- **Recall:** coverage of reference objects.
- Lower CHAIRs/CHAIRi is better; interpret these alongside recall and generated length.

The caption loop skips completed image IDs when reusing an output file. Failed/OOM samples can be skipped, so check completed sample counts and compare methods on the same evaluated IDs.

## Adaptive alpha scheduling

The configuration loader reads `configs/deco/base.json`, then applies a **shallow** override from `configs/deco/<model-name>.json`. The model name is the last component of the Hugging Face ID.

For Qwen2.5-VL-3B-Instruct, the committed entropy schedule is:

```json
{
  "alpha_schedule": {
    "type": "entropy",
    "schedule": {
      "low": 0.25,
      "medium": 0.45,
      "high": 0.6,
      "very_high": 0.75
    },
    "entropy_ranges": {
      "low": [0.0, 0.7],
      "medium": [0.7, 1.2],
      "high": [1.2, 1.8],
      "very_high": [1.8, 2.5]
    },
    "mode": "adaptive"
  }
}
```

The implementation computes entropy from the final-layer distribution for the current token, maps it to a configured interval, and uses the resulting alpha in early-layer logit fusion. Intervals are lower-inclusive and upper-exclusive. Entropy is averaged across the batch before scheduling, so this is a per-generation-step batch-level coefficient, not a separate alpha for each example in a multi-example batch.

The helper also supports time-based and layer-based schedules. Other configuration fields control top-p/top-k candidate filtering, entropy/margin/KL confidence checks, and persistence. The committed values are experimental settings, not generally optimal hyperparameters. Values outside configured entropy ranges retain the alpha passed to the helper.

## Results and reproducibility

Explore `opera_log/chair_eval_results/` and `opera_log/pope_eval_results/` for saved model/method outputs. Historical directory labels are not substitutes for the exact configuration used in a run.

For a comparison, record:

1. Repository commit and local Transformers version.
2. Model ID/revision, environment, GPU, precision, and attention backend.
3. Dataset split, image IDs, prompt, and generation length.
4. Decoding method, early-exit layers, and a copy of the JSON configuration.
5. Completed sample count, failures, predictions, and evaluation output.

No aggregate performance improvement is claimed in this README. A fair result requires matched settings and sample coverage. The documented commands have been checked against the source interfaces; they have not been re-run end-to-end on a GPU as part of this documentation update.

## Acknowledgments

Related paper:

**SAKED: Mitigating Hallucination in Large Vision-Language Models via Stability-Aware Knowledge Enhanced Decoding** — arXiv preprint. [arXiv:2602.09825](https://arxiv.org/abs/2602.09825).

This repository uses or adapts existing decoding methods and tools, including OPERA, DeCo, DoLa, VCD, Hugging Face Transformers, and FlashAttention. The header of `chair.py` credits the [original CHAIR implementation](https://github.com/LisaAnne/Hallucination) and its subsequent adaptation. Please retain source attribution and follow the applicable upstream licenses and dataset terms.
