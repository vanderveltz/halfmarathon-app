# 🏃 Half Marathon Finish Time Predictor

Aplikacja Streamlit szacująca czas ukończenia półmaratonu na podstawie płci, wieku i tempa na 5 km.  
Wytrenowana na danych **Półmaratonu Wrocławskiego 2023–2024** (~22 000 wyników).

---

## Struktura projektu

```
halfmarathon_app/
├── training_pipeline.ipynb   # Notebook ML — czyszczenie danych + trening
├── app.py                    # Aplikacja Streamlit
├── requirements.txt
├── .env.example              # Szablon zmiennych środowiskowych
└── README.md
```

---

## Konfiguracja

### 1. Zmienne środowiskowe

Skopiuj `.env.example` → `.env` i uzupełnij wartości:

```bash
cp .env.example .env
```

| Zmienna | Opis |
|---|---|
| `OPENAI_API_KEY` | Klucz API OpenAI |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Klucze Langfuse (cloud.langfuse.com) |
| `DO_SPACES_KEY` / `DO_SPACES_SECRET` | Klucze Digital Ocean Spaces |
| `DO_SPACES_REGION` | Region (np. `fra1`) |
| `DO_SPACES_BUCKET` | Nazwa bucketu |

### 2. Digital Ocean Spaces — przygotowanie

Utwórz bucket i wgraj dane:

```bash
# Struktura bucketu:
# halfmarathon-ml/
# ├── data/
# │   ├── halfmarathon_wroclaw_2023__final.csv
# │   └── halfmarathon_wroclaw_2024__final.csv
# └── models/
#     ├── model_latest.joblib
#     └── model_latest_meta.json
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

---

## Uruchomienie

### Trenowanie modelu

Otwórz i uruchom notebook `training_pipeline.ipynb` komórka po komórce.  
Model zostanie automatycznie zapisany lokalnie i wysłany do DO Spaces.

### Aplikacja lokalnie

```bash
streamlit run app.py
```

---

## Deploy na Digital Ocean App Platform

1. Wgraj kod na GitHub (lub GitLab)
2. W DO App Platform → **Create App** → wskaż repozytorium
3. Ustaw **Run Command**: `streamlit run app.py --server.port 8080 --server.address 0.0.0.0`
4. Dodaj zmienne środowiskowe w sekcji **Environment Variables** (z `.env`)
5. Deploy 🚀

---

## Jak działa aplikacja

```
Użytkownik wpisuje tekst
        │
        ▼
GPT-4o-mini (OpenAI)
  wyłuskuje: płeć, wiek, tempo_5km
        │
        ├── Langfuse loguje zapytanie + score (ile pól wyłuskano)
        │
        ▼
Walidacja danych
  └── brak danych → info co uzupełnić
        │
        ▼
GradientBoostingRegressor
  (model_latest.joblib z DO Spaces)
        │
        ▼
Wyświetlenie czasu HH:MM:SS
```

---

## Model ML

- **Algorytm**: Gradient Boosting Regressor (scikit-learn)
- **Features**: płeć (0/1), wiek, tempo 5 km (min/km)
- **Target**: czas ukończenia w sekundach
- **Dane**: 2023 + 2024, ~22 000 rekordów po czyszczeniu
- **MAE**: ~2–3 minuty (zależy od treningu)
- **R²**: ~0.97

Najsilniejszy predyktor to **tempo na 5 km** (importance ~0.90).
