"""One-command setup: dependencies, models, index, smoke test.

    python scripts/setup.py

Safe to re-run. Every step checks whether it is already done, so an
interrupted download or a half-built index resumes rather than starting
over.

Why this script exists rather than a line in the README: the models are
several gigabytes and are fetched from Hugging Face at first use, so a
new machine that just runs the server sits in silence for ten minutes
with no indication of whether it is working. This makes that explicit,
and it downloads ONLY the weight files actually used — left to itself
the library pulls safetensors, PyTorch .bin and ONNX copies of the same
model, roughly tripling the footprint.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "weights"

EMBEDDER = "BAAI/bge-m3"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

# One format only. Without this the hub fetches .safetensors AND
# pytorch_model.bin AND the ONNX exports of the same weights.
ALLOW = ["*.json", "*.txt", "*.md", "*.safetensors", "*.model", "tokenizer*", "*.py"]
IGNORE = ["*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.bin", "*.h5", "*.ckpt"]


NEW_SETTINGS_HEADER = (
    chr(10) + chr(10) + "# --- added by setup.py from .env.example ---" + chr(10)
)

def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}", flush=True)


def ok(text: str) -> None:
    print(f"      OK  {text}", flush=True)


def fail(text: str) -> None:
    print(f"\nFAILED: {text}", file=sys.stderr)
    sys.exit(1)


def check_python() -> None:
    if sys.version_info < (3, 11):
        fail(f"Python 3.11+ required, found {sys.version.split()[0]}")
    ok(f"Python {sys.version.split()[0]}")


def install_deps() -> None:
    # Probe the packages the later steps actually import, not just one.
    # A partial environment used to pass this check and then fail at
    # step 5 with a bare ModuleNotFoundError, which looks like a broken
    # repository rather than an unfinished install.
    try:
        import fastapi  # noqa: F401
        import numpy  # noqa: F401
        import pdfplumber  # noqa: F401
        import pypdf  # noqa: F401
        import sentence_transformers  # noqa: F401
        import yaml  # noqa: F401

        ok("dependencies already installed")
        return
    except ImportError as exc:
        print(f"      missing {exc.name} — installing", flush=True)
    print("      installing (this pulls PyTorch — several minutes)…", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cwd=ROOT
    )
    if result.returncode:
        fail("pip install failed — see the output above")
    ok("dependencies installed")


def fetch(repo: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"      {repo} …", flush=True)
    snapshot_download(
        repo_id=repo,
        cache_dir=str(WEIGHTS),
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
    )
    ok(repo)


def write_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        # Merge in keys added to .env.example since this .env was made.
        # Skipping an existing file entirely left early adopters running
        # on a stale config - an old reranker and a score floor from a
        # different scale - which looks like the assistant behaving
        # badly rather than like missing settings. Existing values are
        # never touched: the user's own keys and tuning survive.
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        current = env.read_text(encoding="utf-8")
        have = {
            line.split("=", 1)[0].strip()
            for line in current.splitlines()
            if "=" in line and not line.strip().startswith("#")
        }
        missing = [
            line
            for line in example.splitlines()
            if "=" in line
            and not line.strip().startswith("#")
            and line.split("=", 1)[0].strip() not in have
        ]
        if missing:
            with env.open("a", encoding="utf-8") as fh:
                fh.write(NEW_SETTINGS_HEADER)
                fh.write(chr(10).join(missing) + chr(10))
            ok(f".env updated — added {len(missing)} new setting(s)")
            for line in missing:
                print(f"         {line.split('=', 1)[0].strip()}", flush=True)
        else:
            ok(".env already present and up to date")
    else:
        env.write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
        ok("created .env from .env.example")

    # Warn when a value differs from the template's default on a setting
    # that changes behaviour. Merging only ADDS keys, so a .env created
    # before a default changed keeps the old value silently - which is
    # how someone ends up running a 42-second reranker and a score floor
    # from a different scale, and concluding the assistant is bad.
    CRITICAL = ("RERANKER_MODEL", "RETRIEVAL_SCORE_FLOOR", "GENERATOR_MODEL")
    defaults = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    mine = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in env.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    stale = [
        k for k in CRITICAL
        if k in mine and k in defaults and mine[k] and mine[k] != defaults[k]
    ]
    if stale:
        print("")
        print("      YOUR .env DIFFERS FROM THE RECOMMENDED DEFAULTS:", flush=True)
        for k in stale:
            print(f"         {k}={mine[k]}", flush=True)
            print(f"            recommended: {defaults[k]}", flush=True)
        print(
            "      If you did not change these deliberately, delete .env "
            "and re-run to pick up the current defaults.",
            flush=True,
        )

    text = env.read_text(encoding="utf-8")
    has_key = any(
        line.startswith(("GROQ_API_KEY=", "OPENROUTER_API_KEY="))
        and line.split("=", 1)[1].strip()
        for line in text.splitlines()
    )
    if not has_key:
        print(
            "\n      NOTE: no generator API key set in .env.\n"
            "      Retrieval works without one — the assistant returns the\n"
            "      right sections but writes no answer. For written answers,\n"
            "      add a free key from console.groq.com as GROQ_API_KEY.",
            flush=True,
        )


def build_index() -> None:
    sys.path.insert(0, str(ROOT))
    from app.eval.run import CACHE, build_corpus, read_corpus, save_corpus

    if read_corpus() is not None:
        ok("corpus index already built")
        return
    if not any((ROOT / "Documents").glob("*.pdf")):
        print("      no PDFs in Documents/ — skipping. Upload them in the UI.", flush=True)
        return
    print(
        "      parsing and embedding the manuals. This is the slowest step:\n"
        "      5 to 10 minutes on a laptop CPU. Progress below.",
        flush=True,
    )
    corpus = build_corpus(progress=True)
    save_corpus(corpus)
    ok(f"{len(corpus.chunks)} passages indexed -> {CACHE.parent}")


def smoke() -> None:
    sys.path.insert(0, str(ROOT))
    from app.eval.run import read_corpus
    from app.models.embedder.local import LocalEmbedder
    from app.query.retrieve import InMemoryIndex

    corpus = read_corpus()
    if corpus is None:
        print("      no index to test — skipped", flush=True)
        return
    index = InMemoryIndex()
    index.add(corpus.chunks, corpus.vectors)
    embedder = LocalEmbedder(cache_dir=str(WEIGHTS))
    hits = index.search(
        "maximum screw head height",
        embedder.embed_query_sync("maximum screw head height"),
        limit=3,
    )
    if not hits:
        fail("retrieval returned nothing — the index may be corrupt")
    ok(f"retrieval works — top hit {hits[0].chunk.document} p{hits[0].chunk.page}")


def main() -> None:
    print("Installation & Commissioning Assistant — setup")
    total = 6
    step(1, total, "Checking Python")
    check_python()
    step(2, total, "Installing dependencies")
    install_deps()
    step(3, total, "Downloading models (~2.5 GB, one time)")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for repo in (EMBEDDER, RERANKER):
        fetch(repo)
    step(4, total, "Configuring .env")
    write_env()
    step(5, total, "Building the corpus index")
    build_index()
    step(6, total, "Smoke test")
    smoke()

    print(
        "\nDone. Start the assistant with:\n"
        "    python -m uvicorn app.main:app --port 8000\n"
        "then open http://127.0.0.1:8000\n"
        "\nFirst start loads the models into memory and takes about a minute.",
        flush=True,
    )


if __name__ == "__main__":
    main()
