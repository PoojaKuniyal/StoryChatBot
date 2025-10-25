import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils import extract_text_from_pdf, chunk_text
from app.logger import get_logger
from app.custom_exception import CustomException

logger = get_logger(__name__)

# custom start pages as PDFs have prefaces/contents
START_PAGES = {
    "alice_in_wonderland.pdf": 6,
    "arabian_nights.pdf": 32,
    "gullivers_travel.pdf": 11,
}

def build_index(
    pdf_folder="data/stories_pdf",
    index_path="data/index.faiss",
    meta_path="data/meta.pkl"
):
    try:

        logger.info("Building FAISS index from stories...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        chunks, meta = [], []

        for file in os.listdir(pdf_folder):
            if not file.endswith(".pdf"):
                continue
            path = os.path.join(pdf_folder, file)

            # Use custom start page 
            start_page = START_PAGES.get(file, 0)
            logger.info(f"Extracting from {file} (starting page {start_page})...")

            text = extract_text_from_pdf(path, start_page=start_page)

            story_chunks = chunk_text(text)
            for i, chunk in enumerate(story_chunks):
                meta.append({
                    "id": len(chunks),
                    "text": chunk,
                    "source": file,
                    "page_start": start_page
                })
                chunks.append(chunk)

        logger.info(f" Total text chunks: {len(chunks)}")

        logger.info(" Encoding embeddings...")
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)

        logger.info("Building FAISS index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Index built and saved at: {index_path}")
        logger.info(f"Metadata saved at: {meta_path}")

    except Exception as e:
        logger.error(f"Failed to build the index {e}")
        raise CustomException("Failed to build the index")

if __name__ == "__main__":
    build_index()
