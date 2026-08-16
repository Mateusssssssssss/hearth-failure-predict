from pathlib import Path

# Raiz do projeto, calculada sem depender do diretório de onde o script foi executado.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"

SOURCE_DATA_PATH = ROOT_DIR / "heart_failure_clinical_records_dataset.csv"
RAW_DATA_PATH = RAW_DIR / "heart_failure_clinical_records_dataset.csv"
# Coluna a ser prevista: 1 = óbito e 0 = não óbito.
TARGET = "DEATH_EVENT"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def ensure_directories() -> None:
    """Cria as pastas de saída antes que os scripts gravem artefatos nelas."""
    for directory in (RAW_DIR, PROCESSED_DIR, BEST_MODEL_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
