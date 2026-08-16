"""Análise exploratória e gráficos."""
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.config import FIGURES_DIR, REPORTS_DIR, TARGET, ensure_directories

def create_eda(df: pd.DataFrame) -> None:
    """Gera gráficos exploratórios e um resumo da qualidade do conjunto."""
    ensure_directories()
    sns.set_theme(style="whitegrid", context="notebook")
    numeric = df.select_dtypes(include="number").columns.tolist()
    features = [c for c in numeric if c != TARGET]

    # Mostra o desbalanceamento da variável que será prevista.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, legend=False, ax=ax, palette="Set2")
    ax.set_title("Distribuição da variável alvo")
    ax.set_xlabel("Óbito (DEATH_EVENT)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=160); plt.close(fig)

    # Formato longo permite desenhar um boxplot para cada variável.
    melted = df[features].melt(var_name="variável", value_name="valor")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=melted, x="variável", y="valor", ax=ax, color="#76b7b2", fliersize=2)
    ax.set_title("Boxplots das variáveis numéricas (escala original)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "boxplots_features.png", dpi=160); plt.close(fig)

    # Correlações ajudam a investigar redundância entre variáveis.
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df[numeric].corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlação entre variáveis numéricas")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160); plt.close(fig)

    summary = {"rows": int(len(df)), "columns": int(df.shape[1]), "missing_values": df.isna().sum().to_dict(), "target_distribution": df[TARGET].value_counts().sort_index().to_dict()}
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
