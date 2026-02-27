"""
🏃 Half Marathon Finish Time Predictor
Streamlit app z integracją OpenAI + Langfuse + DO Spaces
"""

import os
import io
import json
import joblib
import boto3
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langfuse.openai import OpenAI

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Konfiguracja klientów
# ────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DO_SPACES_KEY      = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET   = os.getenv("DO_SPACES_SECRET")
DO_SPACES_REGION   = os.getenv("DO_SPACES_REGION", "fra1")
DO_SPACES_BUCKET   = os.getenv("DO_SPACES_BUCKET", "halfmarathon-ml")
DO_SPACES_ENDPOINT = f"https://{DO_SPACES_REGION}.digitaloceanspaces.com"

s3 = boto3.client(
    "s3",
    region_name=DO_SPACES_REGION,
    endpoint_url=DO_SPACES_ENDPOINT,
    aws_access_key_id=DO_SPACES_KEY,
    aws_secret_access_key=DO_SPACES_SECRET,
)

# ─────────────────────────────────────────────────────────────────
# Ładowanie modelu z DO Spaces (cache)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Ładowanie modelu z Digital Ocean Spaces…")
def load_model():
    response = s3.get_object(Bucket=DO_SPACES_BUCKET, Key="models/model_latest.joblib")
    model_bytes = response["Body"].read()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    model = joblib.load(tmp_path)

    # Meta
    try:
        meta_resp = s3.get_object(Bucket=DO_SPACES_BUCKET, Key="models/model_latest_meta.json")
        meta = json.loads(meta_resp["Body"].read())
    except Exception:
        meta = {}

    return model, meta


# ─────────────────────────────────────────────────────────────────
# Ekstrakcja danych przez LLM (OpenAI) — monitorowana przez Langfuse
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Jesteś pomocnikiem, który wyłuskuje dane o biegaczu z naturalnego tekstu.

Zwróć TYLKO poprawny JSON w formacie:
{
  "plec": "M" lub "K" lub null,
  "wiek": <liczba całkowita> lub null,
  "tempo_5km": <liczba dziesiętna, tempo w min/km na dystansie 5 km> lub null
}

Zasady:
- "plec": "M" dla mężczyzny, "K" dla kobiety
- "wiek": wiek w latach (jeśli podano rok urodzenia, oblicz wiek)
- "tempo_5km": tempo w min/km; jeśli podano np. "5:30 min/km" → 5.5, "6 minut na km" → 6.0
- Jeśli jakaś wartość jest nieznana, ustaw null
- Nie dodawaj żadnych komentarzy — tylko JSON"""


def extract_runner_data(user_text: str) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        name="extract_runner_data",  # nazwa trafia do Langfuse
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    return data

    raw = response.choices[0].message.content.strip()

    # Próbujemy sparsować JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Czasem model dodaje ```json … ``` — strip to
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    langfuse_context.update_current_observation(
        output=data,
        level="DEFAULT",
    )

    # Logujemy score: czy udało się wyłuskać wszystkie 3 pola
    filled = sum(1 for v in [data.get("plec"), data.get("wiek"), data.get("tempo_5km")] if v is not None)
    langfuse_context.score_current_observation(
        name="fields_extracted",
        value=filled / 3,
        comment=f"Wyłuskano {filled}/3 pól",
    )

    return data


# ─────────────────────────────────────────────────────────────────
# Predykcja
# ─────────────────────────────────────────────────────────────────
def predict_time(model, plec: str, wiek: int, tempo_5km: float) -> int:
    plec_encoded = 0 if plec == "M" else 1
    X = [[plec_encoded, wiek, tempo_5km]]
    seconds = int(model.predict(X)[0])
    return seconds


def seconds_to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def validate_data(data: dict) -> list[str]:
    missing = []
    if data.get("plec") not in ("M", "K"):
        missing.append("płeć (M/K)")
    if data.get("wiek") is None:
        missing.append("wiek (lub rok urodzenia)")
    elif not (18 <= int(data["wiek"]) <= 100):
        missing.append("wiek w przedziale 18-100 lat")
    if data.get("tempo_5km") is None:
        missing.append("tempo na 5 km (np. 6:00 min/km)")
    return missing


# ─────────────────────────────────────────────────────────────────
# UI Streamlit
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏃 Kalkulator półmaratonu",
    page_icon="🏃",
    layout="centered",
)

st.title("🏃 Kalkulator czasu półmaratonu")
st.subheader("Wrocław Half Marathon — predykcja na podstawie AI")

st.markdown("""
Model wytrenowany na danych z **Półmaratonu Wrocławskiego 2023 i 2024** (~22 000 wyników).

Powiedz mi coś o sobie — podaj swoje **imię, płeć, wiek i tempo na 5 km**,
a obliczę Twój szacowany czas ukończenia!
""")

# Przykłady
with st.expander("💡 Przykłady jak się przedstawić"):
    st.markdown("""
- *„Cześć, jestem Marek, mam 35 lat, jestem mężczyzną i biegam 5 km w tempie 5:30 min/km"*
- *„Nazywam się Ania, kobieta, ur. 1990, moje tempo to 6 minut na km"*
- *„Adam, 28 lat, M, pace 4:45/km"*
""")

user_input = st.text_area(
    "Przedstaw się 👇",
    placeholder="Np. Jestem Tomek, mężczyzna, mam 42 lata i biegam 5 km w tempie 5:50 min/km",
    height=120,
)

predict_btn = st.button("🔮 Oblicz mój czas!", type="primary", use_container_width=True)

if predict_btn and user_input.strip():
    with st.spinner("Analizuję Twoje dane…"):
        try:
            model, meta = load_model()
        except Exception as e:
            st.error(f"❌ Nie udało się załadować modelu z DO Spaces: {e}")
            st.stop()

        with st.spinner("Pytam AI o Twoje dane…"):
            extracted = extract_runner_data(user_input)

    missing = validate_data(extracted)

    # Wyświetl wyłuskaane dane
    if extracted:
        st.markdown("### 📋 Rozpoznane dane:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Płeć",      extracted.get("plec") or "—")
        col2.metric("Wiek",      f"{extracted.get('wiek')} lat" if extracted.get("wiek") else "—")
        col3.metric("Tempo 5 km", f"{extracted.get('tempo_5km')} min/km" if extracted.get("tempo_5km") else "—")

    if missing:
        st.warning(
            f"⚠️ Brakuje mi następujących danych, żeby obliczyć czas:\n\n"
            + "\n".join(f"- **{m}**" for m in missing)
            + "\n\nPodaj brakujące informacje i spróbuj ponownie 🙂"
        )
    else:
        predicted_s = predict_time(
            model,
            plec=extracted["plec"],
            wiek=int(extracted["wiek"]),
            tempo_5km=float(extracted["tempo_5km"]),
        )
        hms = seconds_to_hms(predicted_s)
        minutes = predicted_s / 60

        st.success("✅ Gotowe!")
        st.markdown(f"""
<div style="text-align:center; padding: 2rem; background: linear-gradient(135deg, #1a1a2e, #16213e);
border-radius: 16px; margin: 1rem 0;">
  <div style="font-size: 1rem; color: #aaa; margin-bottom: 0.5rem;">Szacowany czas ukończenia</div>
  <div style="font-size: 4rem; font-weight: bold; color: #e94560; letter-spacing: 4px;">{hms}</div>
  <div style="font-size: 1rem; color: #aaa; margin-top: 0.5rem;">({minutes:.1f} minut)</div>
</div>
""", unsafe_allow_html=True)

        # Kontekst
        if minutes < 90:
            note = "🔥 Świetny wynik! Plasuje Cię w czołówce biegaczy."
        elif minutes < 105:
            note = "💪 Dobry wynik! Powyżej średniej biegacza rekreacyjnego."
        elif minutes < 120:
            note = "🏃 Solidny wynik dla ambitnego biegacza rekreacyjnego."
        else:
            note = "🎉 Ukończenie półmaratonu to już sukces — powodzenia!"
        st.info(note)

        if meta:
            with st.expander("🔬 Informacje o modelu"):
                st.json(meta)

elif predict_btn and not user_input.strip():
    st.warning("Napisz coś o sobie, zanim klikniesz przycisk 😊")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center; color: gray; font-size: 0.8rem;'>"
    "Model wytrenowany na danych Półmaratonu Wrocławskiego 2023–2024 · "
    "Powered by Gradient Boosting + OpenAI + Langfuse"
    "</div>",
    unsafe_allow_html=True,
)
