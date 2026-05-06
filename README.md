# MediPredict — Prédiction du diabète

## Description

MediPredict est une application de Machine Learning permettant d’estimer le risque de diabète de type 2 à partir de données médicales simples.

Ce projet s’inscrit dans une démarche de sensibilisation et ne constitue pas un outil de diagnostic médical.

## Objectifs

- Analyser un dataset médical
- Construire un modèle de classification
- Évaluer les performances du modèle
- Expliquer les prédictions
- Déployer une application web avec Streamlit

## Dataset

Le projet utilise le **Pima Indians Diabetes Dataset**, contenant 768 observations et 8 variables médicales.

## Modèles utilisés

- Logistic Regression
- Random Forest (modèle final retenu)

## Résultats

- Accuracy : 86 %
- AUC-ROC : 0.94

Le modèle Random Forest offre les meilleures performances, notamment en réduisant le nombre de faux négatifs.

## Explicabilité

L’importance des variables montre que les facteurs les plus influents sont :
- Insulin
- Glucose
- BMI
- Age

## Technologies utilisées

- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Joblib
- Streamlit

## Structure du projet

medipredict/
│
├── data/
├── notebooks/
├── model/
├── app/
├── README.md



## Avertissement

Ce projet est un outil de sensibilisation. Il ne remplace pas un avis médical.



Lien vers le projet:

Github: https://github.com/LevanaDems/medipredict-diabetes-ml.git

Streamlit Cloud:  https://medipredict-app.streamlit.app/