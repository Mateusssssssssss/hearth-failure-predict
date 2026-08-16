"""Treino, balanceamento e avaliação sem vazamento de dados."""
import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

from src.config import (BEST_MODEL_DIR, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR,
                        TARGET, TEST_SIZE, ensure_directories)

logger = logging.getLogger(__name__)


def _threshold_search(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Testa todos os limiares de 0,05 a 0,95 nas previsões out-of-fold do treino.

    Não usamos as linhas de teste nesta busca, portanto não há vazamento. F1 é o
    índice otimizado pois combina precision e recall da classe de óbito.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for threshold in thresholds:  # Avalia explicitamente cada probabilidade de corte.
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predictions)),
        })
    table = pd.DataFrame(rows)
    maximum_f1 = table["f1"].max()
    tied = table.loc[table["f1"] == maximum_f1].copy()
    # Em empate, 0,5 é a decisão probabilística mais neutra.
    threshold = float(tied.iloc[(tied["threshold"] - 0.5).abs().argmin()]["threshold"])
    return threshold, table


def _samplers() -> list:
    """Opções de balanceamento aplicadas apenas em cada fold de treino."""
    return ["passthrough", SMOTE(random_state=RANDOM_STATE), RandomUnderSampler(random_state=RANDOM_STATE)]


def _build_candidates() -> dict:
    """Cria os dois algoritmos solicitados e as opções de reamostragem."""
    return {
        "random_forest": (
            Pipeline([("sampler", "passthrough"), ("classifier", RandomForestClassifier(
                n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1))]),
            {"sampler": _samplers()},
        ),
        "xgboost": (
            Pipeline([("sampler", "passthrough"), ("classifier", XGBClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1))]),
            {"sampler": _samplers()},
        ),
    }


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    """Calcula as métricas de classificação para um limiar já definido."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Treina RF/XGBoost, registra ambos e seleciona o melhor por CV ROC-AUC."""
    ensure_directories()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train.assign(**{TARGET: y_train}).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DIR / "test.csv", index=False)
    logger.info("Divisão criada: treino=%d | teste=%d", len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparisons, fitted = {}, {}
    for name, (pipeline, params) in _build_candidates().items():
        logger.info("Treinando %s e comparando: sem reamostragem, SMOTE e undersampling", name)
        search = GridSearchCV(pipeline, params, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        # Probabilidades OOF são previsões de linhas que não participaram daquele fit.
        oof_probabilities = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold, threshold_table = _threshold_search(y_train, oof_probabilities)
        test_probabilities = model.predict_proba(X_test)[:, 1]
        comparisons[name] = {
            "cv_roc_auc": float(search.best_score_),
            "best_params": {key: repr(value) for key, value in search.best_params_.items()},
            "test_metrics": _metrics(y_test, test_probabilities, threshold),
        }
        fitted[name] = {"model": model, "threshold": threshold, "threshold_table": threshold_table,
                        "test_probabilities": test_probabilities}
        logger.info("%s | CV ROC-AUC=%.3f | limiar F1=%.3f | teste ROC-AUC=%.3f", name,
                    search.best_score_, threshold, comparisons[name]["test_metrics"]["roc_auc"])

    # Apenas CV escolhe o vencedor; as métricas de teste são relatadas, não usadas na escolha.
    winner = max(comparisons, key=lambda name: comparisons[name]["cv_roc_auc"])
    winner_data = fitted[winner]
    metrics = {"model": winner, **comparisons[winner]["test_metrics"], "models": comparisons}
    logger.info("Vencedor por validação cruzada: %s", winner)

    # Artefato explicitamente solicitado para produção: sempre contém o Random Forest.
    rf_data = fitted["random_forest"]
    rf_artifact = {
        "model": rf_data["model"], "threshold": rf_data["threshold"],
        "features": X.columns.tolist(), "model_name": "random_forest", "model_version": "1.0.0",
    }
    joblib.dump(rf_artifact, BEST_MODEL_DIR / "random_forest_production.joblib")
    production_metrics = {
        "model_name": "random_forest",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "features": X.columns.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "selection_criterion": "highest_cv_roc_auc",
        "random_forest_metrics": comparisons["random_forest"],
        "selection_winner": winner,
        "selection_winner_metrics": metrics,
        "all_models_comparison": comparisons,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (MODELS_DIR / "production_metrics.json").write_text(json.dumps(production_metrics, indent=2), encoding="utf-8")
    winner_data["threshold_table"].to_csv(REPORTS_DIR / "threshold_search.csv", index=False)
    winner_predictions = (winner_data["test_probabilities"] >= winner_data["threshold"]).astype(int)
    pd.DataFrame({"y_true": y_test, "probability_death_event": winner_data["test_probabilities"], "prediction": winner_predictions}).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    _plot_confusion_matrix(y_test, winner_predictions, winner, winner_data["threshold"])
    return metrics


def _plot_confusion_matrix(y_true, y_pred, model_name, threshold) -> None:
    """Salva a matriz de confusão do modelo escolhido."""
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Sobreviveu", "Óbito"], yticklabels=["Sobreviveu", "Óbito"])
    ax.set(xlabel="Predição", ylabel="Real", title=f"Matriz de confusão — {model_name} (limiar={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
