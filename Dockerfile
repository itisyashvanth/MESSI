# ==============================================================================
# MESSI — Dockerfile
# Offline-deployable, no cloud inference dependencies.
# ==============================================================================
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (offline from cache after first build)
RUN python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_sm

# Copy project source
COPY . .

# Create required directories
RUN mkdir -p data/{raw,annotated,synthetic} models

# Generate minimal emoji vocab (needed for untrained demo runs)
RUN python -c "
from preprocessing import build_vocab_from_texts, save_vocab
from config import EMOJI_VOCAB_PATH
vocab = build_vocab_from_texts(['😠','😤','😡','💀','🔥','😊','👍','🙏','🤬','😢','😭','❤️','⚠️','🚨'])
save_vocab(vocab, EMOJI_VOCAB_PATH)
print('Emoji vocab built.')
"

# Expose web server port
EXPOSE 5000

# Default: run web server (accessible at http://localhost:5000)
# Override for CLI:  docker run messi python main.py --text "your message"
ENTRYPOINT ["python"]
CMD ["server.py", "--host", "0.0.0.0", "--port", "5000"]
