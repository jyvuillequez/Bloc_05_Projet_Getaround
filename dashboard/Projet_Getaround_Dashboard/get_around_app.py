import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Getaround Dashboard", layout="wide")
st.title("Getaround Dashboard")

# CONFIG
DELAY_PATH = "data/get_around_delay_analysis.csv"
PRICING_PATH = "data/get_around_pricing_project.csv"
THRESHOLDS = [0, 15, 30, 45, 60, 75, 90, 105, 120]


# LOAD DATA
df_delay = pd.read_csv(DELAY_PATH)
df_delay = df_delay[df_delay["state"] == "ended"].copy()

df_delay["checkin_type"] = df_delay["checkin_type"].astype(str).str.lower().str.strip()
df_delay["delay"] = pd.to_numeric(df_delay["delay_at_checkout_in_minutes"], errors="coerce")
df_delay["gap"] = pd.to_numeric(df_delay["time_delta_with_previous_rental_in_minutes"], errors="coerce")

# chain = lignes utiles pour "impact location suivante"
chain = df_delay.dropna(subset=["gap", "delay"]).copy()
chain["delay_pos"] = chain["delay"].clip(lower=0)
chain["impact_next"] = chain["delay_pos"] > chain["gap"]
chain["problematic"] = chain["impact_next"]
chain["is_late"] = chain["delay"] > 0
chain["friction_min"] = (chain["delay_pos"] - chain["gap"]).clip(lower=0)

# pricing
df_price = pd.read_csv(PRICING_PATH)
order = df_price["model_key"].value_counts().index.tolist()

# NAVIGATION
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Menu",
    ["Analyse des délais", "Analyse du prix", "Seuil entre locations", "Prédiction du prix"],
    index=0
)

# PAGE 1 — EDA DELAY
if page == "Analyse des délais":
    st.subheader("Analyse des délais")

    col1, col2 = st.columns(2)

    with col1:
        cap = st.slider("Zoom retard (min)", 200, 1200, 600, step=100)
        df_plot = df_delay[df_delay["delay"].between(-cap, cap)].copy()

        st.plotly_chart(
            px.histogram(df_plot, x="delay", nbins=80,
                         title=f"Distribution des retards (filtrée à ±{cap} min)",
                         labels={"delay": "Retard checkout (min)"}),
            use_container_width=True
        )

        st.plotly_chart(
            px.box(df_plot.dropna(subset=["checkin_type"]),
                   x="checkin_type", y="delay",
                   title="Retards au checkout par flow",
                   labels={"checkin_type": "Flow", "delay": "Retard (min)"}),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.histogram(chain, x="gap", nbins=80,
                         title="Distribution du gap (zone tampon) entre locations",
                         labels={"gap": "Gap (min)"}),
            use_container_width=True
        )

        impact_rate_all = chain["impact_next"].mean() * 100
        impact_rate_late = chain.loc[chain["is_late"], "impact_next"].mean() * 100

        k1, k2 = st.columns(2)
        k1.metric("% de locations impactées (global)", f"{impact_rate_all:.1f}%")
        k2.metric("% de locations impactées (si retard)", f"{impact_rate_late:.1f}%")

        tmp = []
        for flow, g in chain.groupby("checkin_type"):
            tmp.append({
                "flow": flow,
                "impact_all": g["impact_next"].mean() * 100,
                "impact_among_late": g.loc[g["is_late"], "impact_next"].mean() * 100,
                "n": len(g),
            })
        impact_df = pd.DataFrame(tmp)
        impact_long = impact_df.melt(
            id_vars=["flow", "n"],
            value_vars=["impact_all", "impact_among_late"],
            var_name="metric", value_name="rate"
        )

        metric_labels = {
        "impact_all": "Global (toutes locations)",
        "impact_among_late": "Parmi les retards",
        }
        impact_long["metric"] = impact_long["metric"].map(metric_labels)

        st.plotly_chart(
            px.bar(impact_long, x="flow", y="rate", color="metric", barmode="group",
                   title="Impact sur la location suivante (%) par canal",
                   labels={"flow": "Canal", "rate": "Taux (%)", "metric": ""}),
            use_container_width=True
        )

# PAGE 2 — EDA PRICING
elif page == "Analyse du prix":
    st.subheader("Analyse du prix")

    st.plotly_chart(
        px.histogram(df_price, x="rental_price_per_day", nbins=60,
                     title="Distribution du prix journalier",
                     labels={"rental_price_per_day": "€ / jour"}),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(df_price, x="model_key",
                     title="Distribution des marques",
                     labels={"model_key": "Marque"},
                     category_orders={"model_key": order}),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.box(df_price, x="fuel", y="rental_price_per_day",
                   title="Prix par carburant",
                   labels={"fuel": "Carburant", "rental_price_per_day": "€ / jour"}),
            use_container_width=True
        )

        st.plotly_chart(
            px.box(df_price, x="car_type", y="rental_price_per_day",
                   title="Prix par type de voiture",
                   labels={"car_type": "Type", "rental_price_per_day": "€ / jour"}),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.box(df_price, x="has_getaround_connect", y="rental_price_per_day",
                   title="Prix selon Getaround Connect",
                   labels={"has_getaround_connect": "Connect", "rental_price_per_day": "€ / jour"}),
            use_container_width=True
        )

        st.plotly_chart(
            px.scatter(df_price, x="engine_power", y="rental_price_per_day",
                       title="Puissance moteur vs prix",
                       labels={"engine_power": "Puissance", "rental_price_per_day": "€ / jour"}),
            use_container_width=True
        )

# PAGE 3 — TRADE-OFF
elif page == "Seuil entre locations":
    st.subheader("Seuil entre locations")

    scope = st.sidebar.radio("Scope", ["all", "connect"], index=0)
    T = st.sidebar.slider("Seuil T (min)", 0, 120, 45, step=15)

    d = chain.copy()
    if scope == "connect":
        d = d[d["checkin_type"] == "connect"].copy()

    n_total = len(d)
    n_problematic = int(d["problematic"].sum())

    rows = []
    for t in THRESHOLDS:
        affected = d["gap"] < t
        solved = d["problematic"] & affected

        rows.append({
            "T": t,
            "n_affected": int(affected.sum()),
            "pct_affected": float(affected.mean() * 100),
            "n_solved": int(solved.sum()),
            "pct_solved_of_problematic": float((solved.sum() / n_problematic * 100) if n_problematic else 0.0),
        })
    sim = pd.DataFrame(rows)

    r = sim.set_index("T").loc[T]
    pct_aff = float(r["pct_affected"])
    n_aff = int(r["n_affected"])
    pct_solved = float(r["pct_solved_of_problematic"])
    n_solved = int(r["n_solved"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("% locations affectées", f"{pct_aff:.1f}%")
    c2.metric("locations affectées", f"{n_aff:,}".replace(",", " "))
    c3.metric("% cas problématiques résolus", f"{pct_solved:.1f}%")
    c4.metric("cas résolus", f"{n_solved:,}".replace(",", " "))

    lab = "All" if scope == "all" else "Connect"

    st.success(
        f"Avec un seuil de **{T} minutes** ({lab}), on **évite {n_solved:,} incidents** "
        f"( **{pct_solved:.1f}%** des cas problématiques) où un retard **impacte la location suivante**."
        .replace(",", " ")
    )

    st.info(
        f"En contrepartie, **{n_aff:,} locations** (**{pct_aff:.1f}%**) ont un **gap < {T} min** : "
        "elles sont **potentiellement affectées** par le seuil."
        .replace(",", " ")
    )

    st.caption(
        "NB : l'analyse 'impact location suivante' utilise uniquement les lignes avec gap renseigné. "
        "Le % locations affectées est un proxy d'impact revenu."
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sim["T"], y=sim["pct_solved_of_problematic"], mode="lines+markers", name="% cas résolus"))
    fig1.add_trace(go.Scatter(x=sim["T"], y=sim["pct_affected"], mode="lines+markers", name="% locations affectées", yaxis="y2"))
    fig1.update_layout(
        title="Trade-off selon T (scope sélectionné)",
        xaxis_title="T (min)",
        yaxis=dict(title="% cas résolus"),
        yaxis2=dict(title="% locations affectées", overlaying="y", side="right")
    )

    tmp = []
    for t in THRESHOLDS:
        for flow, g in chain.groupby("checkin_type"):
            affected = g["gap"] < t
            solved = (g["problematic"] & affected).sum()
            tmp.append({"T": t, "flow": flow, "n_solved": int(solved)})
    solved_flow = pd.DataFrame(tmp)

    cost_all = []
    for t in THRESHOLDS:
        aff = chain["gap"] < t
        cost_all.append({"T": t, "pct_affected_all": float(aff.mean() * 100)})
    cost_all = pd.DataFrame(cost_all)

    fig2 = px.bar(
        solved_flow, x="T", y="n_solved", color="flow",
        title="Cas problématiques résolus (volume) par flow, selon T",
        labels={"T": "T (min)", "n_solved": "Nb de cas résolus", "flow": "Flow"}
    )
    fig2 = go.Figure(fig2)
    fig2.add_trace(go.Scatter(
        x=cost_all["T"], y=cost_all["pct_affected_all"], mode="lines+markers",
        name="% locations affectées (All)", yaxis="y2"
    ))
    fig2.update_layout(yaxis2=dict(overlaying="y", side="right", title="% locations affectées"))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig1, use_container_width=True)
    with right:
        st.plotly_chart(fig2, use_container_width=True)

# PAGE 4 — API PREDICTION
elif page == "Prédiction du prix":
    st.subheader("Prédiction du prix de location")

    FEATURE_ORDER = [
        "model_key",
        "mileage",
        "engine_power",
        "fuel",
        "paint_color",
        "car_type",
        "private_parking_available",
        "has_gps",
        "has_air_conditioning",
        "automatic_car",
        "has_getaround_connect",
        "has_speed_regulator",
        "winter_tires",
    ]

    API_URL = "https://jyvuillequez-projet-getaround-api.hf.space/predict"

    st.caption(f"API utilisée : {API_URL}")

    with st.form("prediction_form"):
        model_key = st.selectbox("Modèle de voiture", ["Peugeot", "Audi", "BMW"])
        mileage = st.number_input("Kilométrage", value=50000, step=1000)
        engine_power = st.number_input("Puissance moteur", value=100, step=5)
        fuel = st.selectbox("Type de carburant", ["diesel", "petrol", "electric", "hybrid"])
        paint_color = st.selectbox("Couleur", ["black", "white", "grey", "blue", "red"])
        car_type = st.selectbox("Type de voiture", ["sedan", "convertible", "suv", "coupe"])

        private_parking_available = st.checkbox("Parking privé disponible", value=True)
        has_gps = st.checkbox("GPS", value=True)
        has_air_conditioning = st.checkbox("Climatisation", value=True)
        automatic_car = st.checkbox("Boîte automatique", value=True)
        has_getaround_connect = st.checkbox("Getaround Connect", value=True)
        has_speed_regulator = st.checkbox("Régulateur de vitesse", value=True)
        winter_tires = st.checkbox("Pneus hiver", value=True)

        submitted = st.form_submit_button("Prédire le prix")

    if submitted:
        row_dict = {
            "model_key": model_key,
            "mileage": int(mileage),
            "engine_power": int(engine_power),
            "fuel": fuel,
            "paint_color": paint_color,
            "car_type": car_type,
            "private_parking_available": bool(private_parking_available),
            "has_gps": bool(has_gps),
            "has_air_conditioning": bool(has_air_conditioning),
            "automatic_car": bool(automatic_car),
            "has_getaround_connect": bool(has_getaround_connect),
            "has_speed_regulator": bool(has_speed_regulator),
            "winter_tires": bool(winter_tires),
        }

        row_list = [row_dict[col] for col in FEATURE_ORDER]
        payload = {"input": [row_list]}

        try:
            r = requests.post(API_URL, json=payload, timeout=20)
            if r.status_code == 200:
                pred = r.json()["prediction"][0]
                st.success(f"Prix estimé : **{pred:.2f} € / jour**")
            else:
                st.error(f"Erreur {r.status_code}")
                st.code(r.text)
                with st.expander("Payload envoyé"):
                    st.json(payload)
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête : {e}")
            with st.expander("Payload envoyé"):
                st.json(payload)