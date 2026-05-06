# ================================
# MediPredict — Application Streamlit
# ================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ================================
# Configuration de la page
# ================================

st.set_page_config(
    page_title="MediPredict",
    page_icon="🩺",
    layout="centered"
)

# ================================
# Sidebar
# ================================

with st.sidebar:
    st.title("🩺 MediPredict")
    st.markdown("### Application de sensibilisation")
    
    st.markdown("""
    Cette application estime un niveau de risque de diabète de type 2 à partir de données médicales simples.
    """)

    st.markdown("---")

    st.markdown("### 🤖 Modèle utilisé")
    st.write("Random Forest")

    st.markdown("### 📊 Performances")
    st.write("Accuracy : 86 %")
    st.write("AUC-ROC : 0.94")

    st.markdown("---")

    st.markdown("### 🔐 Confidentialité")
    st.write("Les données saisies ne sont pas stockées.")
    st.write("Traitement temporaire pendant la session.")

    st.markdown("---")

    st.caption("Projet pédagogique — Machine Learning & RGPD")



# ================================
# Titre et introduction
# ================================

st.title("🩺 MediPredict")
st.subheader("Prédiction du risque de diabète de type 2")

st.markdown("""
Bienvenue sur **MediPredict**, une application de sensibilisation utilisant le Machine Learning 
pour estimer le risque de diabète de type 2 à partir de données médicales simples.

⚠️ **Important :**  
Cette application ne constitue pas un outil de diagnostic médical.  
Les résultats fournis sont uniquement informatifs et ne remplacent pas l’avis d’un professionnel de santé.
""")

# ================================
# Chargement du modèle et du scaler
# ================================

model = joblib.load("model/medipredict_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.success("✅ Modèle chargé avec succès")

# ================================
# Consentement utilisateur
# ================================

st.markdown("---")
st.header("🔐 Consentement et confidentialité")

st.markdown("""
Avant d’utiliser l’application, veuillez confirmer que vous avez compris les conditions suivantes :

- Les données saisies sont utilisées uniquement pour estimer un risque de diabète.
- Les données ne sont pas stockées dans une base de données.
- Les données sont traitées temporairement pendant la session.
- L’application est un outil de sensibilisation et ne remplace pas un avis médical.
- Vous pouvez arrêter l’utilisation de l’application à tout moment.
""")

consent = st.checkbox(
    "J’accepte les conditions d’utilisation et le traitement temporaire de mes données."
)

if not consent:
    st.warning("Veuillez accepter les conditions pour accéder au formulaire.")
    st.stop()

# ================================
# Formulaire utilisateur
# ================================

# ================================
# Layout en colonnes
# ================================

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("---")
    st.header("📝 Mon profil de santé")

    st.markdown("""
    Veuillez renseigner les informations ci-dessous.  
    Les valeurs saisies seront utilisées uniquement pour estimer le niveau de risque.
    """)

    with st.form("prediction_form"):

        pregnancies = st.number_input(
           "Nombre de grossesses",
           min_value=0,
           max_value=20,
           value=0,
           help="Indiquez 0 si cette variable ne vous concerne pas."
       )

        glucose = st.number_input(
          "Taux de glucose",
          min_value=40,
          max_value=250,
          value=120,
          help="Valeur de glucose sanguin."
        )

        blood_pressure = st.number_input(
           "Pression artérielle",
           min_value=20,
           max_value=140,
           value=70,
           help="Pression artérielle diastolique."
       )

        skin_thickness = st.number_input(
           "Épaisseur du pli cutané",
           min_value=0,
           max_value=100,
           value=20,
           help="Mesure du pli cutané en mm."
        )

        insulin = st.number_input(
           "Taux d’insuline",
           min_value=0,
           max_value=900,
           value=80,
           help="Taux d’insuline mesuré."
        )

        bmi = st.number_input(
           "Indice de masse corporelle (BMI)",
           min_value=10.0,
           max_value=70.0,
           value=25.0,
           step=0.1,
           help="Indice de masse corporelle."
        )

        diabetes_pedigree = st.number_input(
           "Fonction de pedigree du diabète",
           min_value=0.0,
           max_value=3.0,
           value=0.5,
           step=0.01,
           help="Indicateur lié aux antécédents familiaux."
        )

        age = st.number_input(
           "Âge",
           min_value=18,
           max_value=100,
           value=35,
           help="Âge de l’utilisateur."
        )

        submitted = st.form_submit_button("Analyser mon profil")

with col2:

    st.markdown("---")
    st.header("ℹ️ Informations")

    st.info("""
    Cette application utilise un modèle de Machine Learning entraîné sur le dataset Pima Indians Diabetes.
    """)

    st.warning("""
    Les résultats sont informatifs et ne remplacent pas un diagnostic médical.
    """)

    st.success("""
    ✔ Données non stockées  
    ✔ Respect des principes RGPD
    """)


# ================================
# Prédiction
# ================================

if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.header("📊 Résultat de l’analyse")

    if probability < 0.35:
       risk_level = "Faible"
       risk_icon = "🟢"
       risk_message = "Votre profil présente un risque faible selon le modèle."

    elif probability < 0.65:
        risk_level = "Modéré"
        risk_icon = "🟠"
        risk_message = "Votre profil présente un risque modéré selon le modèle."

    else:
        risk_level = "Élevé"
        risk_icon = "🔴"
        risk_message = "Votre profil présente un risque élevé selon le modèle."

    st.markdown(f"""
    <div style="
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border: 1px solid #d9e2ec;
        margin-bottom: 20px;
    ">
    <h3>{risk_icon} Risque estimé : {risk_level}</h3>
    <p style="font-size:16px;">{risk_message}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(f"Score estimé par le modèle : **{probability:.2f}**")


    # Barre de progression du risque
    st.progress(float(probability))

    st.caption(
    "Plus la barre est remplie, plus le risque estimé par le modèle est élevé."
    )

    st.info(
        "Ce résultat est une estimation statistique à but informatif. "
        "Il ne constitue pas un diagnostic médical."
    )

    # ================================
    # Interprétation du résultat
    # ================================

    st.subheader(" Interprétation")

    if risk_level == "Faible":
        st.write("""
        Votre profil présente actuellement un faible niveau de risque selon le modèle.

        Cela ne signifie pas une absence totale de risque, mais les indicateurs analysés restent globalement rassurants.
        """)

    elif risk_level == "Modéré":
        st.write("""
        Le modèle détecte certains facteurs pouvant être associés à un risque modéré de diabète.

        Une attention particulière à l’hygiène de vie et un suivi médical peuvent être recommandés.
        """)

    else:
        st.write("""
        Le modèle détecte plusieurs indicateurs associés à un risque élevé de diabète.

        Il est conseillé de consulter un professionnel de santé afin d’obtenir une évaluation médicale adaptée.
        """)

    # ================================
    # Recommandations générales
    # ================================

    st.subheader(" Recommandations générales")

    st.markdown("""
    - Maintenir une alimentation équilibrée
    - Pratiquer une activité physique régulière
    - Réduire la consommation excessive de sucre
    - Effectuer un suivi médical régulier
    - Surveiller le taux de glucose
    """)

    # ================================
    # Variables importantes
    # ================================

    st.subheader(" Facteurs importants dans le modèle")

    st.write("""
    Les variables les plus influentes dans les prédictions du modèle sont :

    - Insulin
    - Glucose
    - BMI
    - Age

    Ces facteurs jouent un rôle important dans l’estimation du risque.
    """)

    st.warning(
        "Ces informations sont générales et ne doivent pas être interprétées comme un conseil médical personnalisé."
    )

