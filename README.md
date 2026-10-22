# SpliNet Embedder API

A lightweight serving API for the **SpliNet Sentence Embedder**.

The GitHub repository contains the API layer only. The trained runtime assets are distributed independently through the repository's **v1.0.0 GitHub Release** and are downloaded automatically on first launch.

## Model

- Encoder parameters: **82,270,464**
- Embedding dimension: **768**
- Maximum sequence length: **512 tokens**
- Pooling: masked mean
- Output: L2-normalized embeddings
- Fine-tuning: **134,378 positive QQP duplicate pairs**
- Fine-tuning epochs: **1**
- Optimization steps: **8,958**

## Evaluation

| Benchmark | Metric | Score |
|---|---|---:|
| STSBenchmark | Pearson | **0.5257** |
| STSBenchmark | Spearman | **0.5174** |
| SciFact | nDCG@10 | **0.1928** |
| SciFact | MAP@10 | **0.1686** |
| SciFact | MRR@10 | **0.1766** |
| SciFact | Recall@10 | **0.2614** |
| SciFact | Recall@100 | **0.4413** |
| Twenty Newsgroups | Mean V-measure | **0.1532** |
| Twenty Newsgroups | Mean homogeneity | **0.1510** |
| Twenty Newsgroups | Mean completeness | **0.1555** |

Twenty Newsgroups evaluation used **59,545 texts**. The full SciFact evaluation used **5,183 documents** and **300 queries**. STSBenchmark used the complete **1,379-pair test set**.

## Repository structure

```text
SpliNet-Embedder-API/
    LICENSE
    README.md
    api.py
    requirements.txt
```

No trained weights are committed to the repository.

## Release assets

Release **v1.0.0** contains:

```text
splinet_embedder.ts
spiece.model
model_info.json
```

On first startup, `api.py` downloads these files into a local visible `assets/` directory and verifies their SHA-256 hashes before loading the model.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python api.py
```

or:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The first launch downloads the model assets from GitHub Release `v1.0.0`. Subsequent launches reuse the verified local copies.

Interactive OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

## REST endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Embeddings

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["A cat sits on a mat.","A dog runs outside."]}'
```

### Similarity

```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"text_a":"A cat sits on a mat.","text_b":"There is a cat on the mat."}'
```

## Python use

```python
from api import encode, similarity

embeddings = encode([
    "A cat sits on a mat.",
    "There is a cat on the mat."
])

print(embeddings.shape)
print(
    similarity(
        "A cat sits on a mat.",
        "There is a cat on the mat."
    )
)
```

## Inference

CUDA is selected automatically when available. On CUDA, FP16 autocast is used for inference.

If an inference batch exceeds available CUDA memory, the API automatically halves the batch size and retries.

## Provenance

Base SpliNet model:

https://github.com/AngshulMajumdar/SpliNet

Hugging Face model release:

https://huggingface.co/Angshul/SpliNet-Embedder

The Hugging Face model release and this GitHub API are independent distributions.

## License

GNU Affero General Public License v3.0.
