from flask import Flask, render_template, request, jsonify
import spacy
import os
import time
import joblib
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import custom models
from models.spacy.spacy_ner import SpacyNER
from models.crf.crf_model import CRFModel
from models.crf.features import extract_features

app = Flask(__name__)

@app.context_processor
def inject_config():
    return {
        'APP_NAME': os.getenv('APP_NAME', 'Entify'),
        'APP_MODE': os.getenv('APP_MODE', 'PROD').upper(),
        'APP_GITHUB_REPO_URL': os.getenv('APP_GITHUB_REPO_URL', 'https://github.com/softdevhassan/entify'),
        'CONTACT_EMAIL': os.getenv('CONTACT_EMAIL', 'softdevhassan.biz@gmail.com'),
        'UOS_URL': os.getenv('UOS_URL', 'https://su.edu.pk'),
        'ILM_URL': os.getenv('ILM_URL', 'https://ilm.edu.pk/campuses/index.php?campus=ILM-College-Sargodha'),
        'PROJECT_VERSION': os.getenv('PROJECT_VERSION', 'v2.8.0'),
        'TEAM_MEMBERS': os.getenv('TEAM_MEMBERS', '').split(','),
        'TEAM_HANDLES': os.getenv('TEAM_HANDLES', '').split(','),
        'DEPLOY_DOMAIN': os.getenv('DEPLOY_DOMAIN', 'entify.orbin.dev'),
    }

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


@app.route("/docs/api")
def docs_api():
    return render_template("docs_api.html")


@app.route("/docs/tests")
def docs_tests():
    return render_template("docs_tests.html")


@app.route("/sitemap")
def sitemap_page():
    return render_template("sitemap.html")


# --- SEO Routes ---


@app.route("/robots.txt")
def robots():
    base_url = f"https://{os.getenv('DEPLOY_DOMAIN', 'entify.orbin.dev')}"
    content = f"User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: {base_url}/sitemap.xml\n"
    return content, 200, {"Content-Type": "text/plain"}


@app.route("/sitemap.xml")
def sitemap():
    pages = [
        ("/", "1.0", "daily"),
        ("/tool", "0.9", "weekly"),
        ("/about", "0.8", "monthly"),
        ("/timeline", "0.8", "monthly"),
        ("/contact", "0.7", "monthly"),
        ("/docs/user", "0.7", "monthly"),
        ("/docs/dev", "0.7", "monthly"),
        ("/docs/api", "0.7", "monthly"),
        ("/docs/tests", "0.7", "monthly"),
        ("/privacy-policy", "0.5", "monthly"),
        ("/terms-of-use", "0.5", "monthly"),
        ("/sitemap", "0.6", "monthly"),
    ]
    base_url = f"https://{os.getenv('DEPLOY_DOMAIN', 'entify.orbin.dev')}"
    today = time.strftime("%Y-%m-%d")
    
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for page, priority, freq in pages:
        xml += f"  <url>\n"
        xml += f"    <loc>{base_url}{page}</loc>\n"
        xml += f"    <lastmod>{today}</lastmod>\n"
        xml += f"    <changefreq>{freq}</changefreq>\n"
        xml += f"    <priority>{priority}</priority>\n"
        xml += f"  </url>\n"
    xml += "</urlset>"
    return xml, 200, {"Content-Type": "application/xml"}


# --- API Endpoints ---


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json()
    text = data.get("text", "")
    mode = data.get("mode", "compare").lower()  # modes: 'compare', 'crf', 'spacy'

    # Input Validation & Sanitization
    if not text or not text.strip():
        return jsonify({"error": "No text provided"}), 400

    if len(text) > 5000:
        return jsonify({"error": "Input exceeds 5000 characters"}), 413

    # Simple HTML stripping
    import re
    text = re.sub(r'<[^>]*>', '', text)

    results = {
        "text": text,
        "spacy": {"entities": [], "processing_time": 0},
        "crf": {"entities": [], "processing_time": 0, "error": None},
    }

    # Get models (Lazy Load if not already in memory)
    spacy_model = get_spacy_model()
    crf_model, crf_loaded = get_crf_model()

    # Process with SpaCy
    if mode in ["spacy", "compare"]:
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
    else:
        results["spacy"] = None  # Remove key if not requested

    # Process with CRF
    if mode in ["crf", "compare"]:
        if crf_loaded:
            try:
                start_t = time.time()
                sentence = text.split()
                # Extract features for the sentence
                features = [extract_features(sentence, i) for i in range(len(sentence))]
                # Predict labels
                predictions = crf_model.predict([features])[0]
                # Predict marginal probabilities for confidence scores
                marginals = crf_model.predict_marginals([features])[0]
                end_t = time.time()

                # Map predictions to entities layout
                entities = []
                current_entity = None

                # Reconstruct character offsets for CRF based purely on word splits (approximate)
                current_char_idx = 0

                for i, (word, pred, prob_dist) in enumerate(zip(sentence, predictions, marginals)):
                    # Find start char index of the word in original text
                    start_char = text.find(word, current_char_idx)
                    if start_char == -1: start_char = current_char_idx
                    end_char = start_char + len(word)
                    current_char_idx = end_char

                    # Get confidence for the predicted label
                    confidence = round(prob_dist.get(pred, 0) * 100, 1)

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
                                "confidence": confidence
                            }
                        else:
                            # Append to current entity
                            current_entity["text"] += " " + word
                            current_entity["end"] = end_char
                            # Track minimum confidence in multi-word entity
                            current_entity["confidence"] = min(current_entity.get("confidence", 100), confidence)
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
    else:
        results["crf"] = None  # Remove key if not requested

    # Clean results: Remove None keys
    final_results = {k: v for k, v in results.items() if v is not None}
    return jsonify(final_results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
