# SOPM interface — Self-Organizing Prompt Maps for Lightweight Prompt Adaptation
#
# Build:  docker compose build
# Run:    docker compose up
#
# The image ships the embedding model (all-MiniLM-L6-v2) and the pre-computed
# embedding cache, so the first run needs no download beyond the LLM itself.

FROM python:3.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	# sentence-transformers/transformers cache, baked at build time
	HF_HOME=/opt/hf-cache \
	# Same workaround sopm_core.py applies: avoids protobuf/streamlit clashes
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

# The pip bundled in the base image (23.0.1) mis-normalizes distribution names
# (`typing_extensions` vs `typing-extensions`), discards the matching wheel and
# falls back to building the sdist — which fails under a custom --index-url.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# CPU-only torch: the default PyPI wheel drags in the whole CUDA stack (~2GB),
# which is dead weight for this workload (150–3000 short prompts).
RUN pip install --no-cache-dir \
	--index-url https://download.pytorch.org/whl/cpu \
	torch==2.8.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the default embedding model into the image so the container starts
# without reaching out to Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
	&& chown -R 1000:1000 /opt/hf-cache

# --- Application code and data ---
COPY sopm_core.py embeddings_db.py sopmInterface.py ./
COPY ag_news_prompts.csv ./
COPY bench_data/ ./bench_data/

# Seed copy of the embedding cache. The entrypoint installs it into the
# artifacts volume on first boot, so re-embedding the corpus is never needed.
COPY artifacts/embeddings.sqlite /opt/sopm-seed/embeddings.sqlite

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run unprivileged; /app/artifacts is the only path written at runtime and is
# chowned here so the named volume inherits the right ownership.
RUN useradd --create-home --uid 1000 sopm \
	&& mkdir -p /app/artifacts \
	&& chown -R sopm:sopm /app
USER sopm

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
	CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["streamlit", "run", "sopmInterface.py", \
	"--server.port=8501", \
	"--server.address=0.0.0.0", \
	"--server.headless=true", \
	"--browser.gatherUsageStats=false"]
