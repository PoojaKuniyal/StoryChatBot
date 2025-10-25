# app.py
import os
import time
from flask import Flask, render_template, request, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
from app.backend import generate_funny_reply 

# Create Flask app
app = Flask(__name__)

# Load Stable Diffusion pipeline once at startup
 
SD_LOCAL_PATH = "models/sd-turbo_local"  #  image model folder name differs

def load_sd_pipeline(local_path=SD_LOCAL_PATH):
    try:
        print(f"Loading Stable Diffusion pipeline from: {local_path}")
        pipe = StableDiffusionPipeline.from_pretrained(local_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (device == "cuda") else torch.float32
        pipe = pipe.to(device, dtype=dtype) if device == "cuda" else pipe.to(device)
        print(" Stable Diffusion pipeline loaded.")
        return pipe
    except Exception as e:
        print("Failed to load SD pipeline from local path.", e)
        # try to load from HF model id as fallback (may attempt network)
        try:
            print("Trying fallback model id 'runwayml/stable-diffusion-v1-5'...")
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            print("Fallback SD pipeline loaded.")
            return pipe
        except Exception as e2:
            print(" Could not load any Stable Diffusion pipeline:", e2)
            return None

sd_pipe = load_sd_pipeline()

STATIC_DIR = 'static'
def generate_and_save_image(prompt_text: str, out_dir: str = STATIC_DIR) -> str:
    """
    Generate an image using the loaded sd_pipe and save it to static/.
    Returns the relative static path (to be used in the template) or None on failure.
    """
    if sd_pipe is None:
        print("Image pipeline not available. Skipping image generation.")
        return None

    # Build a concise prompt (the backend returns an 'image prompt' string already)
    prompt = prompt_text.strip()
    if not prompt:
        prompt = "A whimsical cartoonish scene from a children's fairy tale, playful, colorful"

# Add style prefix
    prompt = "whimsical storybook art style, colorful, vintage children’s illustration, " + prompt
    # Create unique filename
    fname = f"generated_{int(time.time())}.png"
    out_path = os.path.join(out_dir, fname)

    try:
        # Generate image (this runs on CPU if no GPU; slow but works)
        print("Generating image (this may take time on CPU)...")
        result = sd_pipe(prompt, num_inference_steps=25)
        image = result.images[0]
        image.save(out_path)
        print(f"Saved image to {out_path}")
        # Return path relative to static folder for Jinja: url_for('static', filename=...). But we use static folder as app.static_folder.
        # Our static folder is 'app/static', Flask static route will serve it under '/static/<filename>'
        return f"/static/{fname}"
    except Exception as e:
        print("Image generation failed:", e)
        return None


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    user_query = ""
    reply_text = None
    image_path = None
    image_prompt = None

    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            # Use existing backend to get (reply_text, image_prompt)
            try:
                reply_text, image_prompt = generate_funny_reply(user_query)
                print(f"\n[DEBUG] Generated reply:\n{reply_text}\n")
            except Exception as e:
                reply_text = f"Error while generating reply: {e}"
                image_prompt = None

            # Generate image using the image_prompt returned by backend
            try:
                img_rel_path = None
                if image_prompt:
                    print(f"[DEBUG] Image prompt sent to SD: {image_prompt}")
                    img_rel_path = generate_and_save_image(image_prompt)
                else:
                    img_rel_path = generate_and_save_image("")  # fallback prompt
                image_path = img_rel_path
            except Exception as e:
                print("Error during image generation:", e)
                image_path = None

    return render_template(
        "index.html",
        user_query=user_query,
        reply_text=reply_text,
        image_path=image_path,
        image_prompt=image_prompt
    )


# route to serve generated images from static (Flask serves static automatically).

if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:5000")
    
    app.run(host="127.0.0.1", port=5000, debug=True)
