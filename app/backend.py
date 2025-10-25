import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from app.logger import get_logger

logger= get_logger(__name__)

# Load saved local models
EMBED_MODEL_PATH = "models/all-MiniLM-L6-v2_local"
LLM_MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0_local"

logger.info("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

logger.info("Loading LLM model (local)... this may take a minute")
text_gen = pipeline(
    "text-generation",
    model=LLM_MODEL_PATH,
    tokenizer=LLM_MODEL_PATH,
    device_map="cpu",
    max_new_tokens=200
)

# Load FAISS index
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"

logger.info(" Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
logger.info(f"Loaded {len(meta)} chunks of story data")

def retrieve(query, top_k=4):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [{"score": float(D[0][i]), **meta[I[0][i]]} for i in range(top_k)]
    return results

def build_prompt(query, retrieved):
    context = "\n\n---\n\n".join([r["text"] for r in retrieved])
    return f"""
    You are StoryBot, a witty storyteller. Use the CONTEXT below to answer the USER QUESTION in a creative way, lighthearted, funny tone — short and lively (max 3–4 sentences).
    After your answer, add a line starting with 'IMAGE_PROMPT:' describing a cartoon-style funny scene related to your reply.

    CONTEXT:
    {context}

    USER QUESTION: {query}
    REPLY:
    """

def generate_funny_reply(query):
    retrieved = retrieve(query)
    if not retrieved or retrieved[0]["score"] < 0.45:
        return (
            "I don't know that, my curious traveler! My brain is on vacation, and I'm not sure when it'll be back.",
            "cartoon of a confused storyteller holding a map and scratching his head"
        )

    prompt = build_prompt(query, retrieved)
    output = text_gen(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )[0]["generated_text"]
# high temperature=0.8 and top_p=0.9 settings encourage creativity and diversity,
#  which makes the model more likely to produce interesting, witty responses (as requested)

    # Remove the original prompt
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    # Find and clean the image prompt
    if "IMAGE_PROMPT:" in output:
        reply_part, image_part = output.split("IMAGE_PROMPT:", 1)
        reply = reply_part.strip()
        image_prompt = image_part.strip()

        # Clean image prompt: remove story noise
        image_prompt = image_prompt.split("\n")[0]  # only first line
        image_prompt = image_prompt[:180]  # keep it short
        if not any(word in image_prompt.lower() for word in ["cartoon", "illustration", "drawing", "funny"]):
            image_prompt = "cartoon illustration of " + image_prompt

        return reply, image_prompt
    else:
        # Fallback
        return output.strip(), "cartoon illustration of a whimsical story scene"
