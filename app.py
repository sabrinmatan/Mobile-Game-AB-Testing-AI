import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Konfigurera sidan
st.set_page_config(page_title="King Game Analytics", page_icon="🎮", layout="wide")

st.title("🎮 Mobile Game A/B Testing & Analysis")
st.markdown("Analys av spelardata från **Cookie Cats** för att optimera retention och förutspå churn.")


# 2. Ladda data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cookie_cats.csv")
        return df
    except FileNotFoundError:
        return None


df = load_data()

if df is not None:
    # --- FLIKAR FÖR OLIKA DELAR ---
    tab1, tab2 = st.tabs(["📊 A/B-Test Analys", "🔮 AI: Churn Prediction"])

    with tab1:
        st.header("A/B-Test: Gate 30 vs Gate 40")

        # Val av metrik
        day_option = st.selectbox("Välj mätvärde", ["retention_1 (Dag 1)", "retention_7 (Dag 7)"])
        col_name = "retention_1" if "Dag 1" in day_option else "retention_7"

        # Graf
        retention_stats = df.groupby('version')[col_name].mean() * 100
        fig = px.bar(
            x=retention_stats.index,
            y=retention_stats.values,
            color=retention_stats.index,
            title=f"Genomsnittlig {col_name} (%)",
            color_discrete_map={'gate_30': '#00CC96', 'gate_40': '#EF553B'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistiskt test
        contingency_table = pd.crosstab(df['version'], df[col_name])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        if p < 0.05:
            st.success(f"Resultatet är **Signifikant** (P-värde: {p:.5f}). Skillnaden är statistiskt säkerställd.")
        else:
            st.warning(f"Resultatet är **Ej Signifikant** (P-värde: {p:.5f}).")

    with tab2:
        st.header("🔮 AI Churn Prediction")
        st.markdown("""
        Här använder vi **Machine Learning (Random Forest)** för att förutspå om en spelare kommer sluta spela (Churn) 
        baserat på deras aktivitet de första dagarna.
        """)

        # Förbered data för ML
        # Vi vill gissa 'retention_7' (Sanningen: 1 = Stannar, 0 = Slutar)
        # Features: sum_gamerounds, retention_1, version (omgjord till siffror)

        df_ml = df.copy()
        df_ml['version_code'] = df_ml['version'].apply(lambda x: 0 if x == 'gate_30' else 1)

        X = df_ml[['sum_gamerounds', 'retention_1', 'version_code']]  # Indata
        y = df_ml['retention_7']  # Mål (Target)

        # Träna modellen (körs bara när man klickar på knappen för att spara tid)
        if st.button("Träna AI-modellen nu"):
            with st.spinner("Tränar AI-modellen på 90,000 spelare..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)

                acc = accuracy_score(y_test, model.predict(X_test))
                st.balloons()
                st.success(f"Modell tränad! Noggrannhet: **{acc * 100:.1f}%**")

                # Spara modellen i session state så vi kan använda den nedan
                st.session_state['model'] = model

        # Test-sektion: Låt användaren testa
        if 'model' in st.session_state:
            st.divider()
            st.subheader("Testa modellen på en ny spelare")

            c1, c2 = st.columns(2)
            rounds = c1.number_input("Antal spelade rundor (Total)", min_value=0, value=15)
            came_back_day1 = c2.selectbox("Kom spelaren tillbaka Dag 1?", [True, False])

            # Förbered input
            input_data = pd.DataFrame({
                'sum_gamerounds': [rounds],
                'retention_1': [1 if came_back_day1 else 0],
                'version_code': [0]  # Vi antar standardversionen
            })

            prediction = st.session_state['model'].predict(input_data)[0]
            probability = st.session_state['model'].predict_proba(input_data)[0][1]

            st.markdown("### AI:ns gissning:")
            if prediction == 1:
                st.success(f"✅ Spelaren kommer stanna! (Sannolikhet: {probability * 100:.0f}%)")
            else:
                st.error(
                    f"❌ Varning: Risk för Churn! Spelaren slutar troligen. (Sannolikhet att stanna: {probability * 100:.0f}%)")

else:
    st.error("Filen 'cookie_cats.csv' saknas.")