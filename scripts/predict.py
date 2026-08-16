"""Gera probabilidades e classes para novos pacientes."""
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BEST_MODEL_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de pacientes sem DEATH_EVENT")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()
    # Recupera o pipeline e o limiar que foram definidos durante o treino.
    artifact = joblib.load(BEST_MODEL_DIR / "random_forest_production.joblib")
    df = pd.read_csv(args.input)
    missing = set(artifact["features"]).difference(df.columns)
    if missing: raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    # Converte a probabilidade em classe usando o limiar otimizado, não 0,5 fixo.
    probability = artifact["model"].predict_proba(df[artifact["features"]])[:, 1]
    result = df.copy(); result["probability_death_event"] = probability; result["prediction"] = (probability >= artifact["threshold"]).astype(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Previsões salvas em {args.output}")

if __name__ == "__main__":
    main()
