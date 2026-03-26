from flask import Flask, render_template, request, jsonify
import spacy
import os
import time
import joblib

# Import custom models
from models.spacy.spacy_ner import SpacyNER
from models.crf.crf_model import CRFModel
from models.crf.features import extract_features

app = Flask(__name__)

# --- Global Model Variables (Initialized Lazily) ---
_spacy_model = None
_crf_model = None
_crf_loaded = False


def get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        try:
            print("Loading SpaCy model lazily...")
            _spacy_model = SpacyNER()
            print("SpaCy model loaded successfully.")
        except Exception as e:
            print(f"Error loading SpaCy model: {e}")
    return _spacy_model


def get_crf_model():
    global _crf_model, _crf_loaded
    if _crf_model is None:
        try:
            print("Loading CRF model lazily...")
            _crf_model = CRFModel()
            crf_model_path = os.path.join("models", "crf", "crf_model.joblib")
            _crf_model.load(crf_model_path)
            _crf_loaded = True
            print("CRF model loaded successfully.")
        except Exception as e:
            print(f"Error loading CRF model: {e}")
    return _crf_model, _crf_loaded


# --- Frontend Page Routes ---
@app.route("/health")
def health():
    return "OK", 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/tool")
def tool():
    return render_template("tool.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/timeline")
def timeline():
    return render_template("timeline.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/privacy-policy")
def privacy():
    return render_template("privacy-policy.html")


@app.route("/terms-of-use")
def terms():
    return render_template("terms-of-use.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.route("/docs/user")
def docs_user():
    return render_template("docs_user.html")


@app.route("/docs/dev")
def docs_dev():
    return render_template("docs_dev.html")


# --- SEO Routes ---


@app.route("/robots.txt")
def robots():
    content = "User-agent: *\nDisallow: /api/\n"
    return content, 200, {"Content-Type": "text/plain"}


@app.route("/sitemap.xml")
def sitemap():
    # A basic static sitemap
    pages = [
        "/",
        "/tool",
        "/about",
        "/timeline",
        "/contact",
        "/docs/user",
        "/docs/dev",
        "/privacy-policy",
        "/terms-of-use",
    ]
    base_url = "https://ner.orbin.dev"
    urlset = []
    for page in pages:
        urlset.append(f"  <url>\n    <loc>{base_url}{page}</loc>\n  </url>")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{chr(10).join(urlset)}
</urlset>"""
    return xml, 200, {"Content-Type": "application/xml"}


# --- API Endpoints ---


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    results = {
        "text": text,
        "spacy": {"entities": [], "processing_time": 0},
        "crf": {"entities": [], "processing_time": 0, "error": None},
    }

    # Get models (Lazy Load if not already in memory)
    spacy_model = get_spacy_model()
    crf_model, crf_loaded = get_crf_model()

    # Process with SpaCy
    if spacy_model:
        try:
            spacy_results = spacy_model.process(text)
            results["spacy"]["entities"] = spacy_results.get("entities", [])
            results["spacy"]["processing_time"] = spacy_results.get(
                "processing_time", 0
            )
        except Exception as e:
            results["spacy"]["error"] = str(e)
    else:
        results["spacy"]["error"] = "SpaCy model is not initialized."

    # Process with CRF
    if crf_loaded:
        try:
            start_t = time.time()
            sentence = text.split()
            # Extract features for prediction
            features = [extract_features(sentence, i) for i in range(len(sentence))]
            # Predict
            predictions = crf_model.predict([features])[0]
            end_t = time.time()

            # Map predictions to entities layout
            entities = []
            current_entity = None

            # Reconstruct character offsets for CRF based purely on word splits (approximate)
            current_char_idx = 0

            for i, (word, pred) in enumerate(zip(sentence, predictions)):
                # Find start char index of the word in original text
                start_char = text.find(word, current_char_idx)
                end_char = start_char + len(word)
                current_char_idx = end_char

                if pred != "O":
                    # Remove B- or I- prefixes if present
                    label = pred.split("-")[-1] if "-" in pred else pred

                    if (
                        pred.startswith("B-")
                        or current_entity is None
                        or current_entity["label"] != label
                    ):
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            "text": word,
                            "label": label,
                            "start": start_char,
                            "end": end_char,
                        }
                    else:
                        # Append to current entity
                        current_entity["text"] += " " + word
                        current_entity["end"] = end_char
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            results["crf"]["entities"] = entities
            results["crf"]["processing_time"] = round(end_t - start_t, 4)

        except Exception as e:
            results["crf"]["error"] = str(e)
    else:
        results["crf"]["error"] = "CRF model is not trained yet. Run trainer.py first."

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
