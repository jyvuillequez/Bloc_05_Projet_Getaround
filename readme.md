# Bloc 05 – Projet Getaround

Analyse de données de location de véhicules pour réduire les frictions liées aux retards entre deux locations et industrialiser une prédiction de prix via une API.

Projet réalisé dans le cadre de la certification **RNCP Niveau 6 – Concepteur Développeur en Science des données (Jedha Bootcamp)**.

## 1. Contexte & enjeux

- **Problématique métier (retards) :** définir un **seuil minimal** entre deux locations pour réduire les incidents, tout en limitant l’impact sur le volume de locations.
- **Problématique métier (pricing) :** prédire le **prix de location par jour** à partir des caractéristiques d’un véhicule (modèle, kilométrage, puissance, options…).

## 2. Objectifs du projet

- Quantifier l’impact des retards sur la location suivante (global + par flow `connect` vs `mobile`)
- Simuler des seuils entre locations :
   - % de cas problématiques résolus
   - % de locations affectées
- Construire un modèle de prédiction du prix** et le déployer via une API FastAPI
- Restituer via un dashboard Streamlit (EDA + simulateur seuils + appel API)

## 3. Compétences mobilisées

- Analyse exploratoire (EDA)
- Data cleaning / feature engineering
- Simulation et analyse de scénarios métier
- Modélisation ML (régression) pour le pricing
- Déploiement : FastAPI et Swagger `/docs`
- Dashboard : Streamlit et visulisations Plotly

## 4. Données

- Dataset délais : **Variables clés**
  - `delay_at_checkout_in_minutes` : retard (min) au checkout
  - `time_delta_with_previous_rental_in_minutes` : temps tampon entre 2 locations (gap)
  - `checkin_type` : `connect` vs `mobile`
  - `state` : filtrage sur `ended`
- Dataset pricing : *Variables clés :**
  - Numériques : `mileage`, `engine_power`
  - Catégorielles : `model_key`, `fuel`, `paint_color`, `car_type`
  - Booléennes : `private_parking_available`, `has_gps`, `has_air_conditioning`,
    `automatic_car`, `has_getaround_connect`, `has_speed_regulator`, `winter_tires`

## 5. Méthodologie

1. **Préparation des données**
   - Nettoyage, gestion des valeurs manquantes, création de variables.
2. **Analyse exploratoire**
   - Statistiques descriptives, visualisations, premières hypothèses.
3. **API Pricing**
   - Modèle ML (régression) + prétraitements (numérique + OneHot catégories)
   - Exposition du modèle via FastAPI
4. **Restitution**
   - Dashboard Streamlit (EDA + simulateur seuils + page API prediction)
   - Recommandation de seuil (global / connect) basée sur les résultats

## 6. Résultats & recommandations

### Retards / seuil
- `mobile` présente généralement un **taux d’impact** plus élevé que `connect` (global et parmi les retards).
- L’augmentation du seuil résout plus de cas problématiques**, mais augmente aussi la part de locations affectées**.
- **Recommandation :** sélectionner un seuil **raisonnable (ex : 45–60 min)** pour capter une part importante des cas résolus sans trop pénaliser le volume.

### Pricing
- Le prix journalier est généralement concentré autour d’une zone centrale (100 - 150 dollars) avec quelques outliers.
- `mileage` et `engine_power` sont des composantes structurantes.
- Le modèle est exposé via l’API et accesible depuis le dashboard.

**Limites & améliorations :**
- Prendre en compte des coûts "business réels" (annulations, coût du support utilisateur, ...) pour optimiser le seuil
- Ajouter des features (localisation, saisonnalité, ...)

## 7. Ressources projet
- Github repository : https://github.com/jyvuillequez/Bloc_05_Projet_Getaround
- Dashboard : https://jyvuillequez-projet-getaround-dashboard.hf.space
- API : https://jyvuillequez-projet-getaround-api.hf.space/docs


## 8. Installations des librairies Python
```text
python -m pip install -r requirements.txt
```

## 9. Lancer en local
- Dashboard (Streamlit) : Depuis le dossier dashboard/
```text
streamlit run get_around_app.py
```

- API (FastAPI) : Depuis le dossier api/
```uvicorn app:app --reload --host 127.0.0.1 --port 8000
streamlit run get_around_app.py
```

- API (FastAPI) : via swagger avec exemple de requête /predict
```text
{
  "input": [
    ["Peugeot", 50000, 110, "diesel", "grey", "sedan", true, true, true, true, true, true, true]
  ]
}
```

## 10. Organisation du dépôt
```text
.
├─ api/          # API FastAPI
├─ dashboard     # Dashboard Streamlit
├─ data/         # Datasets
├─ mlflow/       # Fichiers de config MLflow
├─ notebooks/    # Notebooks d'analyse
├─ présentation/ # Slides / captures
├─ src/          # Entraînement pricing (Ridge)