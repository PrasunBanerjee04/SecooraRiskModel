# SecooraRiskModel
This model is intended to serve as a flood-related financial risk forecasting system. We train and optimize a TimesFM machine learning model on time-series water level data streamed from Tybee Island observatories. Combining accurate ML-predictions for water levels combined with urban surface permeability metrics and economic flood data will allow us to generate accurate financial loss predictions, aiding Tybee officials with risk management. Our tech stack includes Apache airflow for ingestion, transformation and pipeline development, TensorFlow for model training, and Postgres for data storage.

## Project Structure

```
SecooraRiskModel/
│
├── airflow/
│   ├── dags/
│   │   └── pipeline.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── model/
│   ├── train_model.py
│   └── model.pkl
│
├── data/                   
│   ├── predictions.csv
│   └── historical.csv
│   └── plot.png
│
├── data_utils/                   
│   └── DataLoader.py
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
│
└── docker-compose.yml

```

