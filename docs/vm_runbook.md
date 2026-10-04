# VM runbook: evals on a Linux GPU VM

For running the SWE-bench evals on a Linux x86_64 VM with an NVIDIA GPU. The GPU only speeds up embedding (indexing
repos for retrieval); the agent's sandboxes and the SWE-bench grader do not use it. On x86_64 the SWE-bench images run
natively, without the emulation they need on Apple Silicon, so test runs are faster too.

## 1. System packages

- Docker, with your user in the `docker` group (`docker run hello-world` works without sudo).
- PostgreSQL 17 with pgvector. On Ubuntu, from the PostgreSQL apt repository: `postgresql-17` and
  `postgresql-17-pgvector`.
- Python 3.13, git, and the NVIDIA driver (`nvidia-smi` shows the GPU).
- Disk: SWE-bench images are 1-3 GB each; the 15-task smoke suite needs about 40 GB of Docker storage.

## 2. Databases

```
sudo -u postgres createuser --pwprompt $USER
sudo -u postgres createdb -O $USER patchpilot
sudo -u postgres createdb -O $USER patchpilot_test
for db in patchpilot patchpilot_test; do sudo -u postgres psql -d $db -c "CREATE EXTENSION IF NOT EXISTS vector"; done
```

Ubuntu's Postgres requires a password over TCP, unlike Homebrew's, so put it in the URLs (step 4).

## 3. Python environment with GPU embeddings

```
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Swap the CPU build of fastembed for the GPU one. Use the same version as fastembed in requirements.txt.
pip uninstall -y fastembed onnxruntime
pip install fastembed-gpu==0.8.1
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

The last line must list `CUDAExecutionProvider`. If it does not, onnxruntime-gpu cannot find CUDA: it needs the CUDA
and cuDNN versions listed for its release in onnxruntime's CUDA execution provider docs (recent releases: CUDA 12,
cuDNN 9). `fastembed-gpu==0.8.1` was not verified to exist when this was written; if pip cannot find it, install the
closest published version and run the tests in step 5.

## 4. Settings

```
cp .env.example .env
```

Then set in `.env`:

```
EMBEDDING_DEVICE=cuda
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/patchpilot
TEST_DATABASE_URL=postgresql://<user>:<password>@localhost:5432/patchpilot_test
ANTHROPIC_API_KEY=...          # only needed for real agent runs, not dry runs
```

Langfuse keys are optional; tracing stays off without them.

## 5. Build and check

```
docker build -t patchpilot-sandbox:latest sandbox/
python -m pytest -q            # offline; needs Docker and Postgres
```

## 6. Confirm the GPU is used

Every eval run prints the embedding device first:

```
Embeddings: jinaai/jina-embeddings-v2-base-code on cuda
```

If it says `on cpu, fell back from cuda: <reason>`, fix what the reason names: "no CUDA support" means the CPU build
of fastembed/onnxruntime is still installed (step 3); "failed to load" or a library error means a CUDA or cuDNN
mismatch. Runs still work on the CPU, just slower.

## 7. Index, then run

```
# Pulls each task's image, checks its test runner, and indexes the repo (warming the embedding cache). No LLM calls.
python -m evals.run --suite smoke --dry-run

# Real runs ask before spending; --max-cost stops new tasks past the cap. Reruns resume.
python -m evals.run --suite smoke --config full --limit 3 --max-cost 2
python -m evals.run --suite smoke --experiment ablation --max-cost 20
python -m evals.run --suite smoke --experiment context --max-cost 10

python -m evals.grade evals/results/smoke
python -m evals.report evals/results/smoke
```

Embeddings are cached in Postgres by content, so each repo is embedded once per distinct file content, across tasks
and configs. See `evals/README.md` for what each step does and what the metrics mean.

## 8. Bring results back

Everything a report needs is under `evals/results/<name>/`: per-task JSON, `predictions.jsonl`, the grader's `logs/`,
and `report.md`. Copy that folder; nothing else is needed to regenerate the report.
