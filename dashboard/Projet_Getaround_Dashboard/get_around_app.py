import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "data/get_around_delay_analysis.csv"

THRESHOLDS = [0, 15, 30, 45, 60, 90, 120]


# Preprocessing data
@st.cache_data
def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df = df[df["state"] == "ended"].copy()

    df["checkin_type"] = df["checkin_type"].astype(str).str.strip().str.lower()
    df["delay"] = pd.to_numeric(df["delay_at_checkout_in_minutes"], errors="coerce")
    df["gap"] = pd.to_numeric(df["time_delta_with_previous_rental_in_minutes"], errors="coerce")

    # subset where gap is known (needed to compute impact on next rental)
    chain = df.dropna(subset=["gap", "delay"]).copy()
    chain["delay_pos"] = chain["delay"].clip(lower=0)
    chain["impact_next"] = chain["delay_pos"] > chain["gap"]
    chain["problematic"] = chain["impact_next"]
    chain["is_late"] = chain["delay"] > 0

    return df, chain


def simulate(chain_df: pd.DataFrame, thresholds, scope: str) -> pd.DataFrame:
    d = chain_df.copy()
    if scope == "connect":
        d = d[d["checkin_type"] == "connect"].copy()

    n_problematic = int(d["problematic"].sum())

    rows = []
    for T in thresholds:
        affected = d["gap"] < T
        solved = d["problematic"] & affected

        rows.append({
            "scope": scope,
            "T": T,
            "n_total": int(len(d)),
            "n_problematic": n_problematic,
            "n_affected": int(affected.sum()),
            "pct_affected": float(affected.mean() * 100),
            "n_solved": int(solved.sum()),
            "pct_solved_of_problematic": float((solved.sum() / n_problematic * 100) if n_problematic else 0.0)
        })

    return pd.DataFrame(rows)


def solved_by_flow(chain_df: pd.DataFrame, thresholds) -> pd.DataFrame:
    rows = []
    for T in thresholds:
        for flow, g in chain_df.groupby("checkin_type"):
            affected = g["gap"] < T
            solved = (g["problematic"] & affected).sum()
            rows.append({"T": T, "flow": flow, "n_solved": int(solved)})
    return pd.DataFrame(rows)


# Application
st.set_page_config(page_title="Getaround - Delay Analysis", layout="wide")
st.title("Getaround - Delay Analysis")

df, chain = load_and_prepare(DATA_PATH)

# Simulations
sim_all = simulate(chain, THRESHOLDS, scope="all")
sim_connect = simulate(chain, THRESHOLDS, scope="connect")

# Pour les stacked bars (global, all flows)
solved_flow = solved_by_flow(chain, THRESHOLDS)
cost_all = sim_all[["T", "pct_affected", "n_affected"]].copy()

# Sidebar
st.sidebar.header("Paramètres")
scope = st.sidebar.radio("Scope", ["all", "connect"], index=0, format_func=lambda x: "All" if x == "all" else "Connect")
T = st.sidebar.slider("Seuil T (min)", min_value=0, max_value=120, step=15, value=45)

# Selection
sim_current = (sim_all if scope == "all" else sim_connect).set_index("T").loc[T]

tab1, tab2 = st.tabs(["Business impact", "Retards & friction"])


# Onglet 1 : Impact business 
with tab1:
    st.subheader("1) Business impact (décision)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("% locations affectées", f"{sim_current['pct_affected']:.1f}%")
    c2.metric("# locations affectées", f"{int(sim_current['n_affected']):,}".replace(",", " "))
    c3.metric("% cas problématiques résolus", f"{sim_current['pct_solved_of_problematic']:.1f}%")
    c4.metric("# cas résolus", f"{int(sim_current['n_solved']):,}".replace(",", " "))

    st.caption(
        "NB : l'analyse 'impact location suivante' repose sur les locations dont le gap est renseigné. "
        "Le % locations affectées sert ici de proxy pour l'impact revenu."
    )

    left, right = st.columns([1, 1])

    # Coubres vs T pour le scope sélectionné
    with left:
        sim_scope = sim_all if scope == "all" else sim_connect

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_scope["T"], y=sim_scope["pct_solved_of_problematic"],
            mode="lines+markers", name="% cas résolus"
        ))
        fig.add_trace(go.Scatter(
            x=sim_scope["T"], y=sim_scope["pct_affected"],
            mode="lines+markers", name="% locations affectées", yaxis="y2"
        ))
        fig.update_layout(
            title="Trade-off selon T (scope sélectionné)",
            xaxis_title="T (min)",
            yaxis=dict(title="% cas résolus"),
            yaxis2=dict(title="% locations affectées", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Stacked bars (global) + ligne de coût (all)
    with right:
        # Barres empilées sur les problèmes résolus (volume) par flux
        fig_bar = px.bar(
            solved_flow,
            x="T", y="n_solved", color="flow",
            title="Cas problématiques résolus (volume) par flow, selon le seuil T",
            labels={"T": "T (min)", "n_solved": "Nb de cas résolus", "flow": "Flow"}
        )

        fig2 = go.Figure(fig_bar)
        fig2.add_trace(go.Scatter(
            x=cost_all["T"], y=cost_all["pct_affected"],
            mode="lines+markers", name="% locations affectées", yaxis="y2"
        ))
        fig2.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="% locations affectées"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig2, use_container_width=True)


# Onglets 2: Retards & friction
with tab2:
    st.subheader("2) Retards & friction (comprendre le problème)")

    colA, colB = st.columns(2)

    with colA:
        cap = st.slider("Zoom retard (min)", 200, 1200, 600, step=100)
        df_plot = df[df["delay"].between(-cap, cap)].copy()
        fig = px.histogram(
            df_plot, x="delay", nbins=80,
            title=f"Distribution des retards (filtrée à ±{cap} min)",
            labels={"delay": "Retard checkout (min)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            df_plot.dropna(subset=["checkin_type"]),
            x="checkin_type", y="delay", points="outliers",
            title="Retards au checkout par flow",
            labels={"checkin_type": "Flow", "delay": "Retard checkout (min)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        # Distribution du gap
        fig = px.histogram(
            chain, x="gap", nbins=80,
            title="Distribution du temps tampon (gap) entre deux locations",
            labels={"gap": "Gap (min)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Taux d'impact par flux
        impact = (
            chain.groupby("checkin_type")
                 .agg(
                     n=("impact_next", "size"),
                     impact_rate_all=("impact_next", "mean"),
                     impact_rate_among_late=("impact_next", lambda s: 0.0)
                 )
                 .reset_index()
        )
        # among late (simple)
        rows = []
        for flow, g in chain.groupby("checkin_type"):
            rows.append({
                "checkin_type": flow,
                "impact_rate_among_late": g.loc[g["is_late"], "impact_next"].mean()
            })
        tmp = pd.DataFrame(rows)

        impact = impact.drop(columns=["impact_rate_among_late"]).merge(tmp, on="checkin_type")
        impact["impact_rate_all"] *= 100
        impact["impact_rate_among_late"] *= 100

        # KPIs
        impact_rate_all = chain["impact_next"].mean() * 100
        impact_rate_among_late = chain.loc[chain["is_late"], "impact_next"].mean() * 100

        k1, k2 = st.columns(2)
        k1.metric("Impact rate (global)", f"{impact_rate_all:.1f}%")
        k2.metric("Impact rate (parmi les retards)", f"{impact_rate_among_late:.1f}%")

        impact_long = impact.melt(
            id_vars=["checkin_type", "n"],
            value_vars=["impact_rate_all", "impact_rate_among_late"],
            var_name="metric", value_name="rate"
        )

        fig = px.bar(
            impact_long, x="checkin_type", y="rate", color="metric", barmode="group",
            title="Impact sur la location suivante (%) par flow",
            labels={"checkin_type": "Flow", "rate": "Taux (%)", "metric": ""},
            hover_data={"n": True}
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption("V1 : focus décision. V2 possibles : minutes de friction (delay_pos-gap), filtres par période, etc.")