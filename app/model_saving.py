# ==========================================================
#  DOWNLOAD + SAVE MODELS LOCALLY (for later use)
# ==========================================================

import os
from huggingface_hub import snapshot_download
from logger import get_logger

logger = get_logger(__name__)

os.makedirs("models", exist_ok=True)

# # ----------------------------------------------------------
# # Download SentenceTransformer embedding model
# # ----------------------------------------------------------
logger.info(" Downloading embedding model...")

embed_repo = "sentence-transformers/all-MiniLM-L6-v2"
embed_local = "models/all-MiniLM-L6-v2_local"

snapshot_download(
    repo_id=embed_repo,
    local_dir=embed_local,
    local_dir_use_symlinks=False,
    resume_download=True
)

logger.info(f" Embedding model saved at: {embed_local}")


# # ----------------------------------------------------------
# #  Download Text Generation model (tinyllama)
# # ----------------------------------------------------------
logger.info("\n Downloading text generation model (TinyLlama)...")


llm_repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
llm_local = "models/TinyLlama-1.1B-Chat-v1.0_local"

snapshot_download(
    repo_id=llm_repo,
    local_dir=llm_local,
    local_dir_use_symlinks=False,
    resume_download=True
)

logger.info(f"Text generation model saved at: {llm_local}")


logger.info("\n Downloading image generation model (sd turbo)...")


sd_repo = "stabilityai/sd-turbo"
sd_local = "models/sd-turbo_local"

snapshot_download(
    repo_id=sd_repo,
    local_dir=sd_local,
    local_dir_use_symlinks=False,
    resume_download=True
)

logger.info(f"Image generation model saved at: {sd_local}")