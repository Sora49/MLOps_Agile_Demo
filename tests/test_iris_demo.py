import numpy as np

from src.iris_demo import evaluate_model, save_model, train_model


def test_model_trains_and_predicts():
    model, X_test, y_test = train_model()
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset({0, 1, 2})


def test_model_accuracy_is_reasonable():
    model, X_test, y_test = train_model()
    accuracy = evaluate_model(model, X_test, y_test)

    assert 0.80 <= accuracy <= 1.0


def test_model_can_be_saved(tmp_path):
    model, _, _ = train_model()
    model_path = save_model(model, tmp_path / "iris_model.pkl")

    assert model_path.exists()
    assert model_path.stat().st_size > 0
