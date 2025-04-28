import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import io
import base64

# Configuration initiale de la page Streamlit
st.set_page_config(
    page_title="ECF BEQUET - Analyse de Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PARTIE 1: GÉNÉRATION DES DONNÉES DE SIMULATION
# ============================================================

def generate_training_data(n_formations=10, n_sessions=50):
    """
    Génère des données de sessions de formation fictives pour simuler 
    l'activité d'un centre de formation professionnelle.
    """
    np.random.seed(42)
    
    # Liste des formations proposées dans le secteur transport/logistique/BTP
    formations = [
        "Transport routier de marchandises", 
        "CACES R489 - Cariste", 
        "Formation permis C", 
        "Formation permis CE",
        "Formation logistique entrepôt",
        "Formation BTP - Engins de chantier",
        "Formation FCO Transport",
        "Sécurité routière professionnelle",
        "Éco-conduite",
        "Transport de matières dangereuses"
    ]
    
    # Catégories de formations
    formation_to_category = {
        "Transport routier de marchandises": "Transport",
        "CACES R489 - Cariste": "Logistique",
        "Formation permis C": "Transport",
        "Formation permis CE": "Transport",
        "Formation logistique entrepôt": "Logistique",
        "Formation BTP - Engins de chantier": "BTP",
        "Formation FCO Transport": "Transport",
        "Sécurité routière professionnelle": "Sécurité",
        "Éco-conduite": "Transport",
        "Transport de matières dangereuses": "Transport"
    }
    
    # Coûts et prix des formations
    formation_costs = {
        "Transport routier de marchandises": 1200, 
        "CACES R489 - Cariste": 900, 
        "Formation permis C": 1500, 
        "Formation permis CE": 1800,
        "Formation logistique entrepôt": 850,
        "Formation BTP - Engins de chantier": 1700,
        "Formation FCO Transport": 950,
        "Sécurité routière professionnelle": 600,
        "Éco-conduite": 500,
        "Transport de matières dangereuses": 1100
    }
    
    formation_prices = {
        "Transport routier de marchandises": 1800, 
        "CACES R489 - Cariste": 1400, 
        "Formation permis C": 2300, 
        "Formation permis CE": 2800,
        "Formation logistique entrepôt": 1300,
        "Formation BTP - Engins de chantier": 2600,
        "Formation FCO Transport": 1500,
        "Sécurité routière professionnelle": 900,
        "Éco-conduite": 700,
        "Transport de matières dangereuses": 1700
    }
    
    # Dates sur les 12 derniers mois
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
    
    # Sites de formation d'ECF BEQUET
    sites = ["Auneau", "Gellainville"]
    
    # Génération des données
    data = []
    for i in range(n_sessions):
        formation = np.random.choice(formations)
        session_date = np.random.choice(dates)
        site = np.random.choice(sites)
        capacity = np.random.randint(8, 16)
        registrations = np.random.randint(4, capacity + 1)
        attendance = np.random.randint(registrations - 2 if registrations > 2 else registrations, registrations + 1)
        cost = formation_costs[formation] + np.random.randint(-100, 101)  # Variation du coût
        price = formation_prices[formation] + np.random.randint(-200, 201)  # Variation du prix
        revenue = attendance * price
        profit = revenue - cost
        satisfaction = round(np.random.normal(8, 1), 1)  # Note de satisfaction sur 10
        
        data.append({
            "Formation": formation,
            "Catégorie": formation_to_category[formation],
            "Date": session_date,
            "Site": site,
            "Capacité": capacity,
            "Inscrits": registrations,
            "Présents": attendance,
            "TauxRemplissage": round(registrations / capacity * 100, 1),
            "TauxPrésence": round(attendance / registrations * 100, 1) if registrations > 0 else 0,
            "Coût": cost,
            "PrixVente": price,
            "Revenu": revenue,
            "Bénéfice": profit,
            "MargeNette": round(profit / revenue * 100, 1) if revenue > 0 else 0,
            "Satisfaction": satisfaction
        })
    
    return pd.DataFrame(data)

def generate_satisfaction_data(df):
    """
    Génère des données détaillées de satisfaction client
    basées sur les sessions de formation.
    """
    satisfaction_data = []
    
    for _, session in df.iterrows():
        num_responses = np.random.randint(max(1, session["Présents"] - 2), session["Présents"] + 1)
        
        for _ in range(num_responses):
            satisfaction_data.append({
                "Formation": session["Formation"],
                "Date": session["Date"],
                "Site": session["Site"],
                "ContenuFormation": round(np.random.normal(8, 1.5), 1),
                "QualitéFormateur": round(np.random.normal(8, 1.5), 1),
                "SupportsPédagogiques": round(np.random.normal(7.5, 1.5), 1),
                "EnvironnementFormation": round(np.random.normal(7, 1.5), 1),
                "PertinencePratique": round(np.random.normal(7.5, 1.5), 1),
                "SatisfactionGlobale": round(np.random.normal(session["Satisfaction"], 0.5), 1)
            })
    
    return pd.DataFrame(satisfaction_data)

# ============================================================
# PARTIE 2: FONCTIONS D'ANALYSE ET DE REPORTING
# ============================================================

def analyze_profitability_by_training(df):
    """Analyse la rentabilité par formation."""
    profitability = df.groupby("Formation").agg({
        "Revenu": "sum",
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "Inscrits": "sum",
        "Présents": "sum",
        "TauxRemplissage": "mean",
        "TauxPrésence": "mean",
        "Satisfaction": "mean"
    }).sort_values("Bénéfice", ascending=False)
    
    return profitability

def analyze_monthly_performance(df):
    """Analyse la performance mensuelle."""
    df_copy = df.copy()
    df_copy["Mois"] = df_copy["Date"].dt.strftime('%Y-%m')
    
    monthly_perf = df_copy.groupby("Mois").agg({
        "Revenu": "sum",
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "Inscrits": "sum",
        "TauxRemplissage": "mean",
        "Satisfaction": "mean"
    })
    
    return monthly_perf

def analyze_by_site(df):
    """Analyse les performances par site."""
    site_analysis = df.groupby("Site").agg({
        "Revenu": "sum",
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "Inscrits": "sum",
        "TauxRemplissage": "mean",
        "TauxPrésence": "mean",
        "Satisfaction": "mean"
    })
    
    return site_analysis

def analyze_by_category(df):
    """Analyse les performances par catégorie de formation."""
    category_analysis = df.groupby("Catégorie").agg({
        "Revenu": "sum",
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "Inscrits": "sum",
        "TauxRemplissage": "mean",
        "Satisfaction": "mean"
    }).sort_values("Bénéfice", ascending=False)
    
    return category_analysis

def analyze_satisfaction(df_satisfaction):
    """Analyse détaillée de la satisfaction client."""
    satisfaction_analysis = df_satisfaction.groupby("Formation").agg({
        "ContenuFormation": "mean",
        "QualitéFormateur": "mean",
        "SupportsPédagogiques": "mean",
        "EnvironnementFormation": "mean",
        "PertinencePratique": "mean",
        "SatisfactionGlobale": "mean"
    }).sort_values("SatisfactionGlobale", ascending=False)
    
    return satisfaction_analysis

def calculate_kpis(df):
    """Calcule les KPIs principaux du centre de formation."""
    kpis = {
        "Chiffre d'affaires total": df["Revenu"].sum(),
        "Bénéfice total": df["Bénéfice"].sum(),
        "Marge nette moyenne": df["MargeNette"].mean(),
        "Nombre total d'inscrits": df["Inscrits"].sum(),
        "Taux de remplissage moyen": df["TauxRemplissage"].mean(),
        "Taux de présence moyen": df["TauxPrésence"].mean(),
        "Satisfaction client moyenne": df["Satisfaction"].mean()
    }
    return kpis

def identify_opportunities_and_risks(df):
    """
    Identifie les formations à risque (faible performance) 
    et les opportunités (forte performance).
    """
    formation_analysis = df.groupby("Formation").agg({
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "TauxRemplissage": "mean",
        "Satisfaction": "mean"
    })
    
    # Formations à risque
    risks = formation_analysis[
        (formation_analysis["MargeNette"] < 20) |  # Marge faible
        (formation_analysis["TauxRemplissage"] < 60) |  # Faible remplissage
        (formation_analysis["Satisfaction"] < 7)  # Satisfaction basse
    ]
    
    # Opportunités
    opportunities = formation_analysis[
        (formation_analysis["MargeNette"] > 40) &  # Marge élevée
        (formation_analysis["TauxRemplissage"] > 80) &  # Bon remplissage
        (formation_analysis["Satisfaction"] > 8)  # Haute satisfaction
    ]
    
    return {
        "risks": risks.sort_values("Bénéfice"),
        "opportunities": opportunities.sort_values("Bénéfice", ascending=False)
    }

def analyze_commercial_performance(df):
    """Analyse les performances commerciales par site."""
    df_copy = df.copy()
    
    commercial_metrics = df_copy.groupby("Site").agg({
        "Capacité": "sum",
        "Inscrits": "sum",
        "Présents": "sum",
        "Revenu": "sum"
    })
    
    commercial_metrics["TauxConversion"] = (commercial_metrics["Inscrits"] / commercial_metrics["Capacité"] * 100).round(1)
    commercial_metrics["TauxPrésence"] = (commercial_metrics["Présents"] / commercial_metrics["Inscrits"] * 100).round(1)
    commercial_metrics["RevenuParInscrit"] = (commercial_metrics["Revenu"] / commercial_metrics["Inscrits"]).round(2)
    
    return commercial_metrics

def analyze_objectives_vs_actuals(df):
    """
    Simule une analyse des objectifs vs réalisations pour
    chaque site de formation.
    """
    sites = df["Site"].unique()
    objectives = {}
    
    for site in sites:
        site_data = df[df["Site"] == site]
        revenue = site_data["Revenu"].sum()
        registrations = site_data["Inscrits"].sum()
        
        # Simuler des objectifs (120% des réalisations)
        revenue_objective = revenue * 1.2
        registrations_objective = registrations * 1.2
        
        objectives[site] = {
            "RevenusObjectif": revenue_objective,
            "RevenusRéalisation": revenue,
            "RevenusAtteinte": (revenue / revenue_objective * 100).round(1),
            "InscriptionsObjectif": registrations_objective,
            "InscriptionsRéalisation": registrations,
            "InscriptionsAtteinte": (registrations / registrations_objective * 100).round(1)
        }
    
    return pd.DataFrame(objectives).T

def analyze_top_trainings_by_site(df):
    """Identifie les formations les plus rentables par site."""
    site_formation_analysis = df.groupby(["Site", "Formation"]).agg({
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "TauxRemplissage": "mean"
    }).reset_index()
    
    top_by_site = {}
    for site in df["Site"].unique():
        site_data = site_formation_analysis[site_formation_analysis["Site"] == site]
        top_trainings = site_data.sort_values("Bénéfice", ascending=False).head(3)
        top_by_site[site] = top_trainings
    
    return top_by_site

# ============================================================
# PARTIE 3: CRÉATION DE VISUALISATIONS POUR STREAMLIT
# ============================================================

def plot_profitability_chart(df):
    """Crée un graphique de rentabilité des formations pour Streamlit."""
    rentabilite = analyze_profitability_by_training(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    rentabilite_plot = rentabilite.sort_values("Bénéfice", ascending=True).tail(5)
    rentabilite_plot["Bénéfice"].plot(kind="barh", color="green", ax=ax)
    plt.title("Top 5 des formations les plus rentables (en €)")
    plt.xlabel("Bénéfice (€)")
    plt.tight_layout()
    return fig

def plot_monthly_trend_chart(df):
    """Crée un graphique d'évolution mensuelle des revenus et bénéfices pour Streamlit."""
    performance_mensuelle = analyze_monthly_performance(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_mensuelle[["Revenu", "Bénéfice"]].plot(kind="line", marker="o", ax=ax)
    plt.title("Évolution mensuelle des revenus et bénéfices")
    plt.ylabel("Montant (€)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig

def plot_filling_rate_chart(df):
    """Crée un graphique des taux de remplissage par formation pour Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby("Formation")["TauxRemplissage"].mean().sort_values().plot(kind="barh", color="orange", ax=ax)
    plt.title("Taux de remplissage moyen par formation (%)")
    plt.xlabel("Taux de remplissage (%)")
    plt.tight_layout()
    return fig

def plot_revenue_distribution_chart(df):
    """Crée un graphique de répartition du chiffre d'affaires par catégorie pour Streamlit."""
    analysis_category = analyze_by_category(df)
    fig, ax = plt.subplots(figsize=(8, 8))
    revenues = analysis_category["Revenu"]
    plt.pie(revenues, labels=revenues.index, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition du chiffre d'affaires par catégorie")
    plt.axis('equal')
    plt.tight_layout()
    return fig

def plot_satisfaction_heatmap(df_satisfaction):
    """Crée une carte thermique de la satisfaction client pour Streamlit."""
    satisfaction_analysis = analyze_satisfaction(df_satisfaction)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(satisfaction_analysis, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".1f", ax=ax)
    plt.title("Carte de satisfaction client par formation")
    plt.tight_layout()
    return fig

def plot_site_comparison_chart(df):
    """Crée un graphique comparatif des performances des sites pour Streamlit."""
    site_perf = analyze_by_site(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    site_perf[["Revenu", "Bénéfice"]].plot(kind="bar", ax=ax)
    plt.title("Comparaison des performances financières par site")
    plt.ylabel("Montant (€)")
    plt.tight_layout()
    return fig

def plot_objectives_chart(df):
    """Crée un graphique de suivi des objectifs vs réalisations pour Streamlit."""
    objectives = analyze_objectives_vs_actuals(df)
    
    # Préparer les données pour le graphique
    sites = objectives.index
    x = np.arange(len(sites))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, objectives["RevenusRéalisation"], width, label='Réalisations')
    rects2 = ax.bar(x + width/2, objectives["RevenusObjectif"], width, label='Objectifs')
    
    ax.set_ylabel('Revenus (€)')
    ax.set_title('Objectifs vs Réalisations par site')
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend()
    
    # Ajouter les pourcentages d'atteinte
    for i, site in enumerate(sites):
        ax.annotate(f"{objectives.loc[site, 'RevenusAtteinte']}%", 
                   xy=(i - width/2, objectives.loc[site, 'RevenusRéalisation']),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# ============================================================
# PARTIE 4: APPLICATION STREAMLIT
# ============================================================

def main():
    # Titre et introduction
    st.title("📊 Tableau de bord - ECF BEQUET")
    st.markdown("### Analyse de performance des centres de formation")
    
    # Barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sélectionnez une page",
        ["Accueil", "Analyse par Formation", "Analyse par Site", "Satisfaction Client", "Rapport Complet"]
    )
    
    # Génération des données (une seule fois)
    if 'df_sessions' not in st.session_state:
        with st.spinner('Génération des données...'):
            st.session_state.df_sessions = generate_training_data(n_sessions=100)
            st.session_state.df_satisfaction = generate_satisfaction_data(st.session_state.df_sessions)
    
    df_sessions = st.session_state.df_sessions
    df_satisfaction = st.session_state.df_satisfaction
    
    # Page d'accueil
    if page == "Accueil":
        # Calcul des KPIs
        kpis = calculate_kpis(df_sessions)
        
        # Affichage des KPIs dans des colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chiffre d'affaires total", f"{kpis['Chiffre d\'affaires total']:,.2f} €")
            st.metric("Nombre total d'inscrits", f"{kpis['Nombre total d\'inscrits']:,}")
        
        with col2:
            st.metric("Bénéfice total", f"{kpis['Bénéfice total']:,.2f} €")
            st.metric("Taux de remplissage moyen", f"{kpis['Taux de remplissage moyen']:.1f}%")
        
        with col3:
            st.metric("Marge nette moyenne", f"{kpis['Marge nette moyenne']:.1f}%")
            st.metric("Satisfaction client moyenne", f"{kpis['Satisfaction client moyenne']:.1f}/10")
        
        # Graphiques principaux
        st.markdown("### Aperçu des performances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Répartition du chiffre d'affaires par catégorie")
            fig_revenue = plot_revenue_distribution_chart(df_sessions)
            st.pyplot(fig_revenue)
        
        with col2:
            st.markdown("#### Top 5 des formations les plus rentables")
            fig_profit = plot_profitability_chart(df_sessions)
            st.pyplot(fig_profit)
        
        # Évolution mensuelle
        st.markdown("### Évolution mensuelle des performances")
        fig_monthly = plot_monthly_trend_chart(df_sessions)
        st.pyplot(fig_monthly)
        
        # Tableau de données (versions expansibles)
        with st.expander("Voir les données brutes des sessions"):
            st.dataframe(df_sessions)
    
    # Page d'analyse par formation
    elif page == "Analyse par Formation":
        st.markdown("## Analyse des formations")
        
        # Graphique de rentabilité
        st.markdown("### Rentabilité par formation")
        fig_profit = plot_profitability_chart(df_sessions)
        st.pyplot(fig_profit)
        
        # Graphique de taux de remplissage
        st.markdown("### Taux de remplissage par formation")
        fig_filling = plot_filling_rate_chart(df_sessions)
        st.pyplot(fig_filling)
        
        # Tableau des formations avec sélection
        st.markdown("### Analyse détaillée par formation")
        
        rentabilite = analyze_profitability_by_training(df_sessions)
        formation_selection = st.selectbox(
            "Sélectionnez une formation pour plus de détails",
            options=rentabilite.index.tolist()
        )
        
        # Affichage des détails de la formation sélectionnée
        if formation_selection:
            formation_data = rentabilite.loc[formation_selection]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chiffre d'affaires", f"{formation_data['Revenu']:,.2f} €")
                st.metric("Nombre d'inscrits", f"{formation_data['Inscrits']:,}")
            
            with col2:
                st.metric("Bénéfice", f"{formation_data['Bénéfice']:,.2f} €")
                st.metric("Taux de remplissage", f"{formation_data['TauxRemplissage']:.1f}%")
            
            with col3:
                st.metric("Marge nette", f"{formation_data['MargeNette']:.1f}%")
                st.metric("Satisfaction moyenne", f"{formation_data['Satisfaction']:.1f}/10")
            
            # Sessions spécifiques à cette formation
            st.markdown("#### Sessions pour cette formation")
            formation_sessions = df_sessions[df_sessions["Formation"] == formation_selection]
            st.dataframe(formation_sessions)
        
        # Opportunités et risques
        st.markdown("### Opportunités et risques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Formations à fort potentiel")
            opportunities = identify_opportunities_and_risks(df_sessions)["opportunities"]
            if not opportunities.empty:
                st.dataframe(opportunities)
            else:
                st.info("Aucune formation ne répond actuellement aux critères de fort potentiel.")
        
        with col2:
            st.markdown("#### Formations à risque")
            risks = identify_opportunities_and_risks(df_sessions)["risks"]
            if not risks.empty:
                st.dataframe(risks)
            else:
                st.info("Aucune formation à risque identifiée.")
    
    # Page d'analyse par site
    elif page == "Analyse par Site":
        st.markdown("## Analyse par site")
        
        # Graphique de comparaison des sites
        st.markdown("### Comparaison des performances financières par site")
        fig_sites = plot_site_comparison_chart(df_sessions)
        st.pyplot(fig_sites)
        
        # Graphique des objectifs vs réalisations
        st.markdown("### Objectifs vs réalisations par site")
        fig_objectives = plot_objectives_chart(df_sessions)
        st.pyplot(fig_objectives)
        
        # Tableau détaillé par site
        st.markdown("### Performance détaillée par site")
        
        site_perf = analyze_by_site(df_sessions)
        site_selection = st.selectbox(
            "Sélectionnez un site pour plus de détails",
            options=site_perf.index.tolist()
        )
        
        if site_selection:
            site_data = site_perf.loc[site_selection]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chiffre d'affaires", f"{site_data['Revenu']:,.2f} €")
                st.metric("Nombre d'inscrits", f"{site_data['Inscrits']:,}")
            
            with col2:
                st.metric("Bénéfice", f"{site_data['Bénéfice']:,.2f} €")
                st.metric("Taux de remplissage", f"{site_data['TauxRemplissage']:.1f}%")
            
            with col3:
                st.metric("Marge nette", f"{site_data['MargeNette']:.1f}%")
                st.metric("Taux de présence", f"{site_data['TauxPrésence']:.1f}%")
            
            # Top formations pour ce site
            st.markdown(f"#### Top formations pour {site_selection}")
            top_trainings = analyze_top_trainings_by_site(df_sessions)[site_selection]
            st.dataframe(top_trainings)
            
            # Sessions spécifiques à ce site
            with st.expander(f"Voir toutes les sessions à {site_selection}"):
                site_sessions = df_sessions[df_sessions["Site"] == site_selection]
                st.dataframe(site_sessions)
    
    # Page de satisfaction client
    elif page == "Satisfaction Client":
        st.markdown("## Analyse de la satisfaction client")
        
        # Heatmap de satisfaction
        st.markdown("### Carte de satisfaction par formation")
        fig_satisfaction = plot_satisfaction_heatmap(df_satisfaction)
        st.pyplot(fig_satisfaction)
        
        # Analyse détaillée
        st.markdown("### Analyse détaillée de la satisfaction")
        
        satisfaction_data = analyze_satisfaction(df_satisfaction)
        formation_selection = st.selectbox(
            "Sélectionnez une formation pour les détails de satisfaction",
            options=satisfaction_data.index.tolist()
        )
        
        if formation_selection:
            satisfaction_details = satisfaction_data.loc[formation_selection]
            
            # Affichage sous forme de jauge
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Contenu de la formation")
                st.progress(satisfaction_details["ContenuFormation"] / 10)
                st.write(f"{satisfaction_details['ContenuFormation']:.1f}/10")
                
                st.markdown("##### Qualité du formateur")
                st.progress(satisfaction_details["QualitéFormateur"] / 10)
                st.write(f"{satisfaction_details['QualitéFormateur']:.1f}/10")
            
            with col2:
                st.markdown("##### Supports pédagogiques")
                st.progress(satisfaction_details["SupportsPédagogiques"] / 10)
                st.write(f"{satisfaction_details['SupportsPédagogiques']:.1f}/10")
                
                st.markdown("##### Environnement de formation")
                st.progress(satisfaction_details["EnvironnementFormation"] / 10)
                st.write(f"{satisfaction_details['EnvironnementFormation']:.1f}/10")
            
            with col3:
                st.markdown("##### Pertinence pratique")
                st.progress(satisfaction_details["PertinencePratique"] / 10)
                st.write(f"{satisfaction_details['PertinencePratique']:.1f}/10")
                
                st.markdown("##### Satisfaction globale")
                st.progress(satisfaction_details["SatisfactionGlobale"] / 10)
                st.write(f"{satisfaction_details['SatisfactionGlobale']:.1f}/10")
            
            # Évaluations pour cette formation
            with st.expander(f"Voir les évaluations détaillées pour {formation_selection}"):
                formation_evaluations = df_satisfaction[df_satisfaction["Formation"] == formation_selection]
                st.dataframe(formation_evaluations)
        
        # Points forts et axes d'amélioration
        st.markdown("### Points forts et axes d'amélioration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Points forts identifiés")
            st.markdown("""
            - **Qualité des formateurs** : Les formateurs sont particulièrement appréciés pour leur expertise et leur pédagogie
            - Contenu des formations : Les contenus sont jugés pertinents et adaptés aux besoins professionnels
            - Pertinence pratique : L'application pratique des connaissances est particulièrement valorisée
            """)      
        with col2:
            st.markdown("#### Axes d'amélioration")
            st.markdown("""
            - **Supports pédagogiques** : Les supports pourraient être modernisés et enrichis
            - **Environnement de formation** : L'amélioration des infrastructures d'accueil est suggérée
            - **Durée des sessions** : Certaines formations pourraient bénéficier d'un ajustement de leur durée
            """)

        # Page de rapport complet
    elif page == "Rapport Complet":
        st.markdown("## Rapport de synthèse pour la direction")
    
        # Résumé des KPIs
        kpis = calculate_kpis(df_sessions)
        
        st.markdown("### Résumé financier")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Chiffre d'affaires", f"{kpis['Chiffre d\'affaires total']:,.2f} €")
        with col2:
            st.metric("Bénéfice", f"{kpis['Bénéfice total']:,.2f} €")
        with col3:
            st.metric("Marge moyenne", f"{kpis['Marge nette moyenne']:.1f}%")
        with col4:
            st.metric("Taux remplissage", f"{kpis['Taux de remplissage moyen']:.1f}%")
        
        # Performance par site
        st.markdown("### Performance par site")
        site_perf = analyze_by_site(df_sessions)
        st.dataframe(site_perf)
        
        # Formations à potentiel et à risque
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Formations à développer (fort potentiel)")
            opportunities = identify_opportunities_and_risks(df_sessions)["opportunities"]
            if not opportunities.empty:
                st.dataframe(opportunities)
            else:
                st.info("Aucune formation ne répond actuellement aux critères de fort potentiel.")
        
        with col2:
            st.markdown("### Formations à surveiller (risques)")
            risks = identify_opportunities_and_risks(df_sessions)["risks"]
            if not risks.empty:
                st.dataframe(risks)
            else:
                st.info("Aucune formation à risque identifiée.")
        
        # Tendances récentes
        st.markdown("### Tendances des 3 derniers mois")
        df_copy = df_sessions.copy()
        df_copy["Mois"] = df_copy["Date"].dt.strftime('%Y-%m')
        last_3_months = df_copy.groupby("Mois").agg({
            "Revenu": "sum",
            "Bénéfice": "sum"
        }).tail(3)
        
        st.line_chart(last_3_months)
        
        # Performance commerciale
        st.markdown("### Performance commerciale")
        commercial_perf = analyze_commercial_performance(df_sessions)
        st.dataframe(commercial_perf)
        
        # Objectifs vs réalisations
        st.markdown("### Objectifs vs réalisations")
        objectives = analyze_objectives_vs_actuals(df_sessions)
        st.dataframe(objectives)
        
        # Recommandations stratégiques
        st.markdown("### Recommandations stratégiques")
        
        # Identifier les catégories les plus rentables
        category_analysis = analyze_by_category(df_sessions)
        if not category_analysis.empty:
            top_category = category_analysis.index[0]
            top_margin_category = category_analysis.sort_values("MargeNette", ascending=False).index[0]
            
            st.markdown(f"1. **Développer les formations de la catégorie '{top_category}'** qui génère le plus de bénéfices")
            st.markdown(f"2. **Optimiser les marges des formations de la catégorie '{top_margin_category}'** qui a la meilleure marge")
        
        # Identifier le site le plus performant
        if not site_perf.empty:
            top_site = site_perf.sort_values("Bénéfice", ascending=False).index[0]
            st.markdown(f"3. **Analyser et répliquer les bonnes pratiques du site '{top_site}'** qui performe le mieux")
        
        st.markdown("""
        4. **Augmenter le taux de remplissage** des formations les plus rentables via des actions marketing ciblées
        5. **Optimiser les coûts** des formations à faible marge mais forte demande
        6. **Envisager d'abandonner ou de restructurer** les formations à risque persistant
        7. **Développer des modules complémentaires** pour les formations à haute satisfaction
        """)
        
        # Export options
        st.markdown("### Export des données")
        
        def get_csv_download_link(df, filename):
            csv = df.to_csv(index=True)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Télécharger {filename}.csv</a>'
            return href
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(get_csv_download_link(df_sessions, "sessions_formation"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(get_csv_download_link(site_perf, "performance_sites"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(get_csv_download_link(analyze_profitability_by_training(df_sessions), "rentabilite_formations"), unsafe_allow_html=True)

    
if __name__ == "__main__":
    main()