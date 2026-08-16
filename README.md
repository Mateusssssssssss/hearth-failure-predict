# Previsão de Eventos por Insuficiência Cardíaca

Projeto reproduzível de ciência de dados para prever `DEATH_EVENT` usando os modelos Random Forest e XGBoost.

## Estrutura

```text
data/raw/             dados de origem
data/processed/       conjuntos tratados (gerados)
models/               pipelines e metadados treinados (gerados)
reports/figures/      EDA, matriz de confusão e comparativos (gerados)
src/                  código reutilizável
scripts/              pontos de entrada
```

## Como executar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_pipeline.py
```

O pipeline copia o CSV original para `data/raw/`, valida os dados, gera a EDA, faz a divisão estratificada treino/teste (80/20), ajusta os modelos com validação cruzada estratificada, seleciona o melhor modelo por ROC-AUC e calibra o limiar de probabilidade para maximizar F1 no treino. As métricas finais são calculadas apenas no conjunto de teste.

## Previsões

Após o treino, crie um CSV com as mesmas colunas preditoras (sem `DEATH_EVENT`) e execute:

```powershell
python scripts/predict.py --input data/novos_pacientes.csv --output reports/predictions.csv
```

> Observação clínica: este projeto é educacional e não substitui avaliação profissional. A variável `time` deve ser excluída se não estiver disponível no instante real em que se pretende fazer a predição.

## Resultados e logs

Após a execução, consulte `reports/model_comparison.json` para as métricas de Random Forest e XGBoost, `reports/threshold_search.csv` para todos os limiares de probabilidade avaliados, e `logs/pipeline.log` para acompanhar a execução.

Para produção, use `models/best_model/random_forest_production.joblib`; as métricas, parâmetros, features e data de treino ficam registrados em `models/production_metrics.json`.

## Docker Compose

Com Docker Desktop iniciado, execute:

```powershell
docker compose up --build
```
