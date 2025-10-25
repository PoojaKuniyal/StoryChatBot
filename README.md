# 🤖 Funny Tales Chatbot

**AI-powered funny storytelling chatbot** built as part of the *Es Magico AI Studio* assignment.
It answers user questions about classic stories in a humorous tone, and even generates related images — all **offline-capable** and **lightweight**.

---
[![Watch the demo](https://img.shields.io/badge/🌌-Watch%20Demo-red)](https://vimeo.com/1130448114?fl=ip&fe=ec)

## 🧠 Overview

**Project Goal:**
Create a **Funny Story Chatbot** trained on three public-domain classics:

* *Alice in Wonderland*
* *Arabian Nights*
* *Gulliver’s Travels*

The chatbot can:

* Answer questions about these stories with a **funny, story-like tone**
* **Generate related images**
* Handle unrelated queries gracefully
* Run entirely **offline on CPU**

---

## ⚙️ Architecture

```
User Query
   ↓
Semantic Search (FAISS)
   ↓
Context Retrieved
   ↓
TinyLlama (Text Generation)
   ↓
Funny Story Reply
   ↓
Stable Diffusion Turbo (Image Generation)
   ↓
Flask Web Interface Output
```

---

## 🧩 Tech Stack

| Component           | Technology                             |                                                
| ------------------- | -------------------------------------- | 
| **Backend**         | Flask                                  |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 | 
| **Text Model**      | TinyLlama-1.1B-Chat-v1.0               | 
| **Image Model**     | stabilityai/sd-turbo                   | 
| **Vector Store**    | FAISS                                  | 
| **PDF Extraction**  | pdfplumber + regex                     | 
| **Frontend**        | Flask + Jinja2 + CSS                   | 

---


## 🧠 RAG for Retrieval & Generation

**Retrieval:**

* User query → embedding → top 4 chunks via FAISS
* If similarity < 0.45 → returns funny fallback:

> "I don't know that, my curious traveler! My brain is on vacation…"

**Generation:**

* Retrieved context + prompt template → TinyLlama response
* Output includes:

  * `reply_text` (funny answer)
  * `image_prompt` (for image generation)

---

## 🎨 Image Generation

* Uses **Stable Diffusion Turbo (sd-turbo)** via `diffusers`
* Saves generated images under `static/generated_*.png`

---

## 🧰 Project Structure

```
project_root/
│
├── app/
│   ├── __init__.py
│   ├── backend.py
│   ├── utils.py
│   ├── logger.py
│   ├── model_saving.py
    ├── test_index.py
├── data/
│   ├── stories_pdf/
│   ├── index.faiss
│   ├── meta.pkl
│
├── models/
│   ├── all-MiniLM-L6-v2_local/
│   ├── TinyLlama-1.1B-Chat-v1.0_local/
│   └── sd-turbo_local/
│
├── static/
│   ├── style.css
│   └── generated_*.png
│
├── templates/
│   └── index.html
│
└── application.py
```

---

## 🧮 Local Model Setup

Models are stored locally for **offline inference**:

```
models/
├── all-MiniLM-L6-v2_local/
├── TinyLlama-1.1B-Chat-v1.0_local/
└── sd-turbo_local/
```

Downloaded using:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir="models/TinyLlama-1.1B-Chat-v1.0_local")
```

✅ Benefits:

* No runtime downloads
* Offline & CPU-friendly
* Faster startup


---

## 🚀 Future Enhancements

* 🎤 Add **speech-to-text (Whisper)** & **text-to-speech (gTTS)**
* 🌍 **Multilingual** support (Hindi, Spanish)
* 💬 **Session memory** for multi-turn chats
* ☁️ **Docker + Render/HF Spaces** deployment

---

## 🏁 Conclusion

**Funny Tales Chatbot** demonstrates an end-to-end **Retrieval-Augmented Generation** system that runs entirely on open-source, offline tools.
It blends **semantic search, creative text generation, and AI-driven image synthesis** — built for fun, education, and storytelling.
