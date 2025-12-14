# 🎮 Mobile Game Optimization: A/B Testing & AI

Ett dataanalysprojekt som undersöker spelarbeteende i mobilspelet **Cookie Cats**. Projektet kombinerar statistisk A/B-testning med Machine Learning för att optimera retention.

## 📌 Projektöversikt
Spelutvecklarna ville veta: *"Ska vi flytta den första 'gaten' (hindret) från bana 30 till bana 40?"*
Jag analyserade data från 90 000 spelare för att svara på frågan.

**Mina resultat:**
1.  **A/B-Test:** Statistisk analys (Chi-Square) visade att retention (7-dagars) var **bättre** när gaten var kvar på bana 30. Att flytta den till 40 sänkte spelarengagemanget signifikant.
2.  **AI Prediction:** Jag tränade en **Random Forest Classifier** för att förutspå 'Churn' (om en spelare slutar) baserat på deras aktivitet de första dagarna.

## 🛠 Tekniker
- **Python & Pandas** (Data Cleaning)
- **Scipy** (Statistisk signifikansanalys / Chi-Square)
- **Scikit-Learn** (Machine Learning / Random Forest)
- **Streamlit** (Interaktiv Dashboard)
- **Plotly** (Visualisering)

## 📊 Insikt till Affärsverksamheten
Min rekommendation är att **inte** flytta gaten till bana 40. Datan visar tydligt att spelare tenderar att sluta spela tidigare om de möter hindret senare, vilket kan bero på minskad "urgency" eller uttråkning.