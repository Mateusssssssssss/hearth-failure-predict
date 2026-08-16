"""Executa o pipeline completo a partir da raiz do projeto."""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LOGS_DIR, PROCESSED_DIR, ensure_directories
from src.data import clean_data, load_raw_data
from src.eda import create_eda
from src.modeling import train_and_evaluate

def main():
    ensure_directories()
    # Logs simultâneos no terminal e em logs/pipeline.log para acompanhamento posterior.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Iniciando pipeline")
    # Carrega a fonte e aplica regras de qualidade antes de qualquer análise.
    raw = load_raw_data()
    clean = clean_data(raw)
    logger.info("Dados carregados: %d linhas e %d colunas", *clean.shape)
    clean.to_csv(PROCESSED_DIR / "heart_failure_clean.csv", index=False)
    # A EDA produz gráficos para inspeção; o treino divide os dados logo depois.
    create_eda(clean)
    metrics = train_and_evaluate(clean)
    logger.info("Pipeline concluído. Modelo vencedor: %s", metrics["model"])
    print(f"Modelo selecionado: {metrics['model']}")
    print(f"ROC-AUC no teste: {metrics['roc_auc']:.3f} | F1 no teste: {metrics['f1']:.3f}")
    print(f"Limiar de decisão: {metrics['threshold']:.3f}")
    print("Comparação completa: reports/model_comparison.json")
    print("Curva de limiares: reports/threshold_search.csv")

if __name__ == "__main__":
    main()
