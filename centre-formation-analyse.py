import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

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

def create_dashboard_summary(df_sessions):
    """Crée un résumé des données pour le tableau de bord."""
    df = df_sessions.copy()
    
    # KPIs généraux
    kpis = calculate_kpis(df)
    
    # Top 3 des formations les plus rentables
    top_formations = analyze_profitability_by_training(df).head(3)
    
    # Performance par site
    site_perf = analyze_by_site(df)
    
    # Tendance des 3 derniers mois
    df["Mois"] = df["Date"].dt.strftime('%Y-%m')
    last_3_months = df.groupby("Mois").agg({
        "Revenu": "sum",
        "Bénéfice": "sum"
    }).tail(3)
    
    summary = {
        "kpis": kpis,
        "top_formations": top_formations,
        "site_performance": site_perf,
        "recent_trend": last_3_months
    }
    
    return summary

def generate_management_report(df_sessions, df_satisfaction, analyses):
    """
    Génère un rapport de synthèse complet pour la direction
    avec analyses et recommandations stratégiques.
    """
    print("\n" + "="*80)
    print("RAPPORT DE SYNTHÈSE POUR LA DIRECTION - ECF BEQUET".center(80))
    print("="*80)
    
    # Résumé financier
    print("\nRÉSUMÉ FINANCIER:")
    print(f"- Chiffre d'affaires total: {analyses['kpis']['Chiffre d\'affaires total']:,.2f} €")
    print(f"- Bénéfice total: {analyses['kpis']['Bénéfice total']:,.2f} €")
    print(f"- Marge nette moyenne: {analyses['kpis']['Marge nette moyenne']:.1f}%")
    
    # Performance par site
    print("\nPERFORMANCE PAR SITE:")
    for site, data in analyses['site_performance'].iterrows():
        print(f"- {site}: CA = {data['Revenu']:,.2f} €, Bénéfice = {data['Bénéfice']:,.2f} €, Marge = {data['MargeNette']:.1f}%")
    
    # Formations à développer
    print("\nFORMATIONS À DÉVELOPPER (FORT POTENTIEL):")
    if not analyses['opportunities'].empty:
        for formation, data in analyses['opportunities'].iterrows():
            print(f"- {formation}: Marge = {data['MargeNette']:.1f}%, Remplissage = {data['TauxRemplissage']:.1f}%, Satisfaction = {data['Satisfaction']:.1f}/10")
    else:
        print("- Aucune formation ne répond actuellement aux critères de fort potentiel")
    
    # Formations à surveiller
    print("\nFORMATIONS À SURVEILLER (RISQUES):")
    if not analyses['risks'].empty:
        for formation, data in analyses['risks'].iterrows():
            print(f"- {formation}: Marge = {data['MargeNette']:.1f}%, Remplissage = {data['TauxRemplissage']:.1f}%, Satisfaction = {data['Satisfaction']:.1f}/10")
    else:
        print("- Aucune formation à risque identifiée")
    
    # Tendances récentes
    print("\nTENDANCES DES 3 DERNIERS MOIS:")
    trend_data = analyses['recent_trend']
    for month, data in trend_data.iterrows():
        print(f"- {month}: CA = {data['Revenu']:,.2f} €, Bénéfice = {data['Bénéfice']:,.2f} €")
    
    # Évolution du dernier mois par rapport au mois précédent
    if len(trend_data) >= 2:
        last_month = trend_data.index[-1]
        previous_month = trend_data.index[-2]
        revenue_change = (trend_data.loc[last_month, 'Revenu'] / trend_data.loc[previous_month, 'Revenu'] - 1) * 100
        profit_change = (trend_data.loc[last_month, 'Bénéfice'] / trend_data.loc[previous_month, 'Bénéfice'] - 1) * 100
        
        print(f"\nÉVOLUTION DERNIER MOIS:")
        print(f"- CA: {revenue_change:.1f}%")
        print(f"- Bénéfice: {profit_change:.1f}%")
    
    # Performance commerciale
    print("\nPERFORMANCE COMMERCIALE:")
    for site, data in analyses['commercial_performance'].iterrows():
        print(f"- {site}: Taux de conversion = {data['TauxConversion']:.1f}%, Taux de présence = {data['TauxPrésence']:.1f}%, Revenu/inscrit = {data['RevenuParInscrit']:.2f} €")
    
    # Objectifs vs réalisations
    print("\nOBJECTIFS VS RÉALISATIONS:")
    for site, data in analyses['objectives_analysis'].iterrows():
        print(f"- {site}: Revenus = {data['RevenusAtteinte']:.1f}% de l'objectif, Inscriptions = {data['InscriptionsAtteinte']:.1f}% de l'objectif")
    
    # Recommandations stratégiques
    print("\nRECOMMANDATIONS STRATÉGIQUES:")
    
    # Identifier les catégories les plus rentables
    category_analysis = analyze_by_category(df_sessions)
    if not category_analysis.empty:
        top_category = category_analysis.index[0]
        top_margin_category = category_analysis.sort_values("MargeNette", ascending=False).index[0]
        
        print(f"1. Développer les formations de la catégorie '{top_category}' qui génère le plus de bénéfices")
        print(f"2. Optimiser les marges des formations de la catégorie '{top_margin_category}' qui a la meilleure marge")
    
    # Identifier le site le plus performant
    if not analyses['site_performance'].empty:
        top_site = analyses['site_performance'].sort_values("Bénéfice", ascending=False).index[0]
        print(f"3. Analyser et répliquer les bonnes pratiques du site '{top_site}' qui performe le mieux")
    
    # Formations les plus rentables par site
    print("\nFORMATIONS LES PLUS RENTABLES PAR SITE:")
    for site, trainings in analyses['top_trainings_by_site'].items():
        print(f"\n{site}:")
        for _, row in trainings.iterrows():
            print(f"- {row['Formation']}: Bénéfice = {row['Bénéfice']:,.2f} €, Marge = {row['MargeNette']:.1f}%, Remplissage = {row['TauxRemplissage']:.1f}%")
    
    print("\nALERTES ET OPPORTUNITÉS:")
    
    # Alertes basées sur les tendances
    if len(trend_data) >= 3:
        if trend_data['Bénéfice'].iloc[-1] < trend_data['Bénéfice'].iloc[-2] < trend_data['Bénéfice'].iloc[-3]:
            print("! ALERTE: Baisse continue des bénéfices sur les 3 derniers mois")
    
    # Opportunités basées sur le taux de remplissage
    high_satisfaction_low_filling = df_sessions.groupby("Formation").agg({
        "Satisfaction": "mean",
        "TauxRemplissage": "mean"
    })
    
    opportunity_formations = high_satisfaction_low_filling[
        (high_satisfaction_low_filling["Satisfaction"] > 8) & 
        (high_satisfaction_low_filling["TauxRemplissage"] < 70)
    ].sort_values("Satisfaction", ascending=False)
    
    if not opportunity_formations.empty:
        print("\nOpportunités de développement (formations avec haute satisfaction mais faible remplissage):")
        for formation, data in opportunity_formations.iterrows():
            print(f"- {formation}: Satisfaction = {data['Satisfaction']:.1f}/10, Remplissage actuel = {data['TauxRemplissage']:.1f}%")
    
    # Suggestions pour améliorer la rentabilité
    print("\nSUGGESTIONS POUR AMÉLIORER LA RENTABILITÉ:")
    print("1. Augmenter le taux de remplissage des formations les plus rentables via des actions marketing ciblées")
    print("2. Optimiser les coûts des formations à faible marge mais forte demande")
    print("3. Envisager d'abandonner ou de restructurer les formations à risque persistant")
    print("4. Développer des modules complémentaires pour les formations à haute satisfaction")
    
    print("\n" + "="*80)
    print("FIN DU RAPPORT".center(80))
    print("="*80)

# ============================================================
# PARTIE 3: CRÉATION DE VISUALISATIONS
# ============================================================

def create_profitability_chart(df, output_file='rentabilite_formations.png'):
    """Crée un graphique de rentabilité des formations."""
    rentabilite = analyze_profitability_by_training(df)
    plt.figure(figsize=(12, 6))
    rentabilite_plot = rentabilite.sort_values("Bénéfice", ascending=True).tail(5)
    rentabilite_plot["Bénéfice"].plot(kind="barh", color="green")
    plt.title("Top 5 des formations les plus rentables (en €)")
    plt.xlabel("Bénéfice (€)")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_monthly_trend_chart(df, output_file='evolution_mensuelle.png'):
    """Crée un graphique d'évolution mensuelle des revenus et bénéfices."""
    performance_mensuelle = analyze_monthly_performance(df)
    plt.figure(figsize=(12, 6))
    performance_mensuelle[["Revenu", "Bénéfice"]].plot(kind="line", marker="o")
    plt.title("Évolution mensuelle des revenus et bénéfices")
    plt.ylabel("Montant (€)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_filling_rate_chart(df, output_file='taux_remplissage.png'):
    """Crée un graphique des taux de remplissage par formation."""
    plt.figure(figsize=(12, 6))
    df.groupby("Formation")["TauxRemplissage"].mean().sort_values().plot(kind="barh", color="orange")
    plt.title("Taux de remplissage moyen par formation (%)")
    plt.xlabel("Taux de remplissage (%)")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_revenue_distribution_chart(df, output_file='repartition_ca.png'):
    """Crée un graphique de répartition du chiffre d'affaires par catégorie."""
    analysis_category = analyze_by_category(df)
    plt.figure(figsize=(10, 6))
    revenues = analysis_category["Revenu"]
    plt.pie(revenues, labels=revenues.index, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition du chiffre d'affaires par catégorie")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_satisfaction_heatmap(df_satisfaction, output_file='satisfaction_client.png'):
    """Crée une carte thermique de la satisfaction client."""
    satisfaction_analysis = analyze_satisfaction(df_satisfaction)
    plt.figure(figsize=(12, 8))
    sns.heatmap(satisfaction_analysis, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".1f")
    plt.title("Carte de satisfaction client par formation")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_site_comparison_chart(df, output_file='comparaison_sites.png'):
    """Crée un graphique comparatif des performances des sites."""
    site_perf = analyze_by_site(df)
    plt.figure(figsize=(10, 6))
    site_perf[["Revenu", "Bénéfice"]].plot(kind="bar")
    plt.title("Comparaison des performances financières par site")
    plt.ylabel("Montant (€)")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_objectives_chart(df, output_file='objectifs_realisations.png'):
    """Crée un graphique de suivi des objectifs vs réalisations."""
    objectives = analyze_objectives_vs_actuals(df)
    plt.figure(figsize=(12, 6))
    
    # Préparer les données pour le graphique
    sites = objectives.index
    x = np.arange(len(sites))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
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
    plt.savefig(output_file)
    return output_file

# ============================================================
# PARTIE 4: EXÉCUTION DU SYSTÈME
# ============================================================

def run_performance_analysis_system():
    """Fonction principale qui exécute l'ensemble du système d'analyse."""
    print("Initialisation du système d'analyse de performance pour les centres de formation ECF BEQUET...")
    
    # Génération des données
    print("\nGénération des données de simulation...")
    df_sessions = generate_training_data(n_sessions=100)
    df_satisfaction = generate_satisfaction_data(df_sessions)
    
    print(f"Données générées: {len(df_sessions)} sessions de formation et {len(df_satisfaction)} évaluations de satisfaction")
    
    # Exécution des analyses
    print("\nExécution des analyses de performance...")
    rentabilite = analyze_profitability_by_training(df_sessions)
    performance_mensuelle = analyze_monthly_performance(df_sessions)
    analyse_site = analyze_by_site(df_sessions)
    analyse_categorie = analyze_by_category(df_sessions)
    analyse_satisfaction = analyze_satisfaction(df_satisfaction)
    
    # Calcul des KPIs
    kpis = calculate_kpis(df_sessions)
    
    # Identification des opportunités et risques
    analysis_results = identify_opportunities_and_risks(df_sessions)
    
    # Analyses commerciales
    commercial_performance = analyze_commercial_performance(df_sessions)
    objectives_analysis = analyze_objectives_vs_actuals(df_sessions)
    top_trainings_by_site = analyze_top_trainings_by_site(df_sessions)
    
    # Création du résumé pour le tableau de bord
    dashboard_summary = create_dashboard_summary(df_sessions)
    
    # Génération des visualisations
    print("\nGénération des visualisations pour le tableau de bord...")
    create_profitability_chart(df_sessions)
    create_monthly_trend_chart(df_sessions)
    create_filling_rate_chart(df_sessions)
    create_revenue_distribution_chart(df_sessions)
    create_satisfaction_heatmap(df_satisfaction)
    create_site_comparison_chart(df_sessions)
    create_objectives_chart(df_sessions)
    
    print("Visualisations créées avec succès!")
    
    # Compilation des analyses pour le rapport
    all_analyses = {
        "kpis": kpis,
        "site_performance": analyse_site,
        "opportunities": analysis_results["opportunities"],
        "risks": analysis_results["risks"],
        "recent_trend": dashboard_summary["recent_trend"],
        "commercial_performance": commercial_performance,
        "objectives_analysis": objectives_analysis,
        "top_trainings_by_site": top_trainings_by_site
    }
    
    # Génération du rapport final
    print("\nGénération du rapport de synthèse pour la direction...")
    generate_management_report(df_sessions, df_satisfaction, all_analyses)
    
    # Export des données pour le tableau de bord
    print("\nExport des données pour intégration dans un tableau de bord...")
    dashboard_summary['top_formations'].to_csv('top_formations.csv')
    analyse_site.to_csv('performance_sites.csv')
    performance_mensuelle.to_csv('performance_mensuelle.csv')
    analyse_satisfaction.to_csv('satisfaction_client.csv')
    
    print("\nSystème d'analyse de performance terminé avec succès.")
    print("""
Ce système permet de:
1. Suivre les KPIs essentiels des centres de formation
2. Analyser la rentabilité par formation, site et catégorie
3. Identifier les opportunités de développement et les risques
4. Comparer les performances entre sites
5. Générer des tableaux de bord visuels pour l'aide à la décision
6. Produire des rapports de synthèse pour la direction
""")

# Lancement du système
if __name__ == "__main__":
    run_performance_analysis_system()