import sklearn_crfsuite
import joblib
import os


class CRFModel:
    def __init__(
        self, c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True
    ):
        """
        Initializes the CRF model with hyperparameters.
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
        )

    def train(self, X_train, y_train):
        """
        Trains the CRF model.
        X_train: List of lists of feature dictionaries.
        y_train: List of lists of labels.
        """
        print(f"Training CRF model with {len(X_train)} sentences...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test):
        """
        Predicts labels for a list of sentences (features).
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X_test)

    def predict_marginals(self, X_test):
        """
        Predicts marginal probabilities for a list of sentences (features).
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict_marginals(X_test)

    def save(self, file_path):
        """
        Saves the trained model to a file.
        """
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a trained model from a file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at: {file_path}")
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
