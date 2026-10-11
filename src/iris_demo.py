from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"


def train_model():
    """Train an Iris Decision Tree model and return the model and test data."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Return the model's accuracy on the supplied test data."""
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def save_model(model, model_path=MODEL_PATH):
    """Save a trained model to disk and return its path."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def main():
    model, X_test, y_test = train_model()
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    model_path = save_model(model)
    print(f"Model saved successfully at: {model_path}")

    loaded_model = joblib.load(model_path)
    loaded_accuracy = evaluate_model(loaded_model, X_test, y_test)
    print(f"Loaded Model Accuracy: {loaded_accuracy:.2f}")
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()
