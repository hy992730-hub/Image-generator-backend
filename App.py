import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

# Get the API key from environment on Render
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    print("WARNING: REPLICATE_API_TOKEN is not set. Set it in Render → Environment.")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN or ""

# Default model: fast & versatile text-to-image
# You can swap this to another model slug later if you like.
MODEL = "black-forest-labs/flux-schnell"

@app.route("/health")
def health():
    return {"ok": True}

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "").strip()
        negative = data.get("negative_prompt", "").strip() or None
        width = int(data.get("width") or 1024)
        height = int(data.get("height") or 1024)
        seed = data.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except:
                seed = None

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Call Replicate
        # FLUX-Schnell supports width/height/seed; negative prompt is appended.
        effective_prompt = prompt
        if negative:
            effective_prompt += f". Avoid: {negative}"

        output = replicate.run(
            f"{MODEL}:latest",
            input={
                "prompt": effective_prompt,
                "width": width,
                "height": height,
                **({"seed": seed} if seed is not None else {})
            },
        )

        # Replicate returns a list of image URLs (usually length 1)
        if isinstance(output, list) and output:
            return jsonify({"image": output[0]})
        # Some models return dict with 'images'
        if isinstance(output, dict) and "images" in output and output["images"]:
            return jsonify({"image": output["images"][0]})

        return jsonify({"error": "No image returned from model"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
