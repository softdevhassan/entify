import os
import sys

# Add project root to path so we can import from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.data_loader import DataLoader
from models.crf.features import extract_features
from models.crf.crf_model import CRFModel
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def prepare_sentence_features(sentence_data):
    """
    Converts a sentence (list of tuples) into features and labels.
    sentence_data: [(word, pos, chunk, label), ...]
    """
    words = [token[0] for token in sentence_data]
    labels = [token[3] for token in sentence_data]

    features = [extract_features(words, i) for i in range(len(words))]
    return features, labels


def main():
    # 1. Load Data
    loader = DataLoader()
    print("Loading datasets...")

    train_file = os.path.join("data", "raw", "conll2003", "eng.train")
    test_file = os.path.join(
        "data", "raw", "conll2003", "eng.testa"
    )  # Using testa as validation

    try:
        train_data = loader.load_data(train_file)
        test_data = loader.load_data(test_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Train sentences: {len(train_data)}")
    print(f"Validation sentences: {len(test_data)}")

    # 2. Extract Features
    print("Extracting features (this may take a minute)...")
    X_train = []
    y_train = []
    for sent in train_data:
        x, y = prepare_sentence_features(sent)
        X_train.append(x)
        y_train.append(y)

    X_test = []
    y_test = []
    for sent in test_data:
        x, y = prepare_sentence_features(sent)
        X_test.append(x)
        y_test.append(y)

    # 3. Train Model
    crf_model = CRFModel(c1=0.1, c2=0.1, max_iterations=100)
    crf_model.train(X_train, y_train)

    print("Evaluating model...")
    y_pred = crf_model.predict(X_test)

    # Filter labels to only include important entity categories (ignoring O) BIO perinciple
    labels = list(crf_model.model.classes_)
    labels.remove("O")

    f1 = metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
    print(f"Weighted F1 Score (excluding 'O'): {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))

    save_path = os.path.join("models", "crf", "crf_model.joblib")
    crf_model.save(save_path)


if __name__ == "__main__":
    main()
