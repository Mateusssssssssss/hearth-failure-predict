"""Ingestão, validação e limpeza dos dados."""
import shutil
import pandas as pd
from src.config import RAW_DATA_PATH, SOURCE_DATA_PATH, TARGET, ensure_directories

REQUIRED_COLUMNS = {
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex",
    "smoking", "time", TARGET,
}

def load_raw_data() -> pd.DataFrame:
    """Copia a fonte para a camada raw (sem transformá-la) e a carrega."""
    ensure_directories()
    # Mantém uma cópia imutável da fonte na camada raw.
    if not RAW_DATA_PATH.exists():
        if not SOURCE_DATA_PATH.exists():
            raise FileNotFoundError(f"CSV não encontrado: {SOURCE_DATA_PATH}")
        shutil.copy2(SOURCE_DATA_PATH, RAW_DATA_PATH)
    return pd.read_csv(RAW_DATA_PATH)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valida esquema, remove duplicatas e trata ausências com mediana."""
    # Falha cedo se o arquivo não tiver o esquema esperado.
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    clean = df.copy()
    clean.columns = clean.columns.str.strip()
    # Registros duplicados distorcem frequências e podem contaminar treino/teste.
    clean = clean.drop_duplicates().reset_index(drop=True)
    if not set(clean[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} deve ser binária (0/1).")
    feature_columns = [c for c in clean.columns if c != TARGET]
    # Valores inválidos viram NaN; em seguida, são imputados pela mediana.
    clean[feature_columns] = clean[feature_columns].apply(pd.to_numeric, errors="coerce")
    clean[feature_columns] = clean[feature_columns].fillna(clean[feature_columns].median())
    clean = clean.dropna(subset=[TARGET])
    clean[TARGET] = clean[TARGET].astype(int)
    return clean
