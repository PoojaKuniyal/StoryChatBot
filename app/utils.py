import pdfplumber,re
from typing import List
from app.custom_exception import CustomException
from app.logger import get_logger

logger = get_logger(__name__)

def extract_text_from_pdf(path: str, start_page: int = 0) -> str: # takes a filepath (pdf) and returns the extracted text as a single string
    try:
        logger.info("Text extraction from pdf initailized...")
        text = [] # hold page text

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[start_page:]: # loop through each page
                page_text = page.extract_text() or "" # extract text or return None
                text.append(page_text)

        joined  = "\n".join(text)  # Join text

        # apply regex cleaning
        joined  = re.sub(r'(?<=\w)\d+', '', joined )                     # Remove numbers attached to words
        joined  = re.sub(r'\n?\s*\d+\s*\n?', '', joined )                # Remove standalone footnote numbers
        joined  = re.sub(r'\s+\n', '\n', joined )                        # Remove trailing spaces before newlines
        joined  = re.sub(r'\n{2,}', '\n\n', joined )                     # Collapse multiple newlines
        
        joined  = re.sub(r'(?<!\n)\n(?!\n)', ' ', joined )               # Replace single newlines with space
        joined  = joined .replace('“', '"').replace('”', '"').replace("’", "'")
        joined  = re.sub(r'[ \t]+', ' ', joined )                        # Normalize spaces
        logger.info("Successfully preprocessed the text")
        return joined 
        

    except Exception as e:
        logger.error(f"Failed to preprocess and chunk {e}")
        raise CustomException("Error while preprocessing", e)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    try:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        logger.info("successfully chunked the text")
        return chunks
    except Exception as e:
        logger.error(f"failed to chunk the text {e}")
        raise CustomException("Failed to chunk the text",e)

if __name__=='__main__':
    et = extract_text_from_pdf('data\stories_pdf\Alice_In_Wonderland.pdf',start_page=6)
    chunk_text(et)