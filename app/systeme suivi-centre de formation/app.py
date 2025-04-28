from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
from matplotlib.figure import Figure
import os

# Définition de toutes les fonctions d'analyse
def generate_training_data(n_formations=10, n_sessions=50):
    np.random.seed(42)
    
    # Liste des formations proposées
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
    
    # Sites de formation
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
        cost = formation_costs[formation] + np.random.randint(-100, 101)
        price = formation_prices[formation] + np.random.randint(-200, 201)
        revenue = attendance * price
        profit = revenue - cost
        satisfaction = round(np.random.normal(8, 1), 1)
        
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

def analyze_profitability_by_training(df):
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

def analyze_by_site(df):
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

def analyze_satisfaction(df_satisfaction):
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
    formation_analysis = df.groupby("Formation").agg({
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "TauxRemplissage": "mean",
        "Satisfaction": "mean"
    })
    
    risks = formation_analysis[
        (formation_analysis["MargeNette"] < 20) |
        (formation_analysis["TauxRemplissage"] < 60) |
        (formation_analysis["Satisfaction"] < 7)
    ]
    
    opportunities = formation_analysis[
        (formation_analysis["MargeNette"] > 40) &
        (formation_analysis["TauxRemplissage"] > 80) &
        (formation_analysis["Satisfaction"] > 8)
    ]
    
    return {
        "risks": risks.sort_values("Bénéfice"),
        "opportunities": opportunities.sort_values("Bénéfice", ascending=False)
    }

def analyze_commercial_performance(df):
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
    sites = df["Site"].unique()
    objectives = {}
    
    for site in sites:
        site_data = df[df["Site"] == site]
        revenue = site_data["Revenu"].sum()
        registrations = site_data["Inscrits"].sum()
        
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

def create_profitability_chart(df, output_file):
    rentabilite = analyze_profitability_by_training(df)
    plt.figure(figsize=(12, 6))
    rentabilite_plot = rentabilite.sort_values("Bénéfice", ascending=True).tail(5)
    rentabilite_plot["Bénéfice"].plot(kind="barh", color="green")
    plt.title("Top 5 des formations les plus rentables (en €)")
    plt.xlabel("Bénéfice (€)")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_revenue_distribution_chart(df, output_file):
    analysis_category = analyze_by_category(df)
    plt.figure(figsize=(10, 6))
    revenues = analysis_category["Revenu"]
    plt.pie(revenues, labels=revenues.index, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition du chiffre d'affaires par catégorie")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_filling_rate_chart(df, output_file):
    plt.figure(figsize=(12, 6))
    df.groupby("Formation")["TauxRemplissage"].mean().sort_values().plot(kind="barh", color="orange")
    plt.title("Taux de remplissage moyen par formation (%)")
    plt.xlabel("Taux de remplissage (%)")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_site_comparison_chart(df, output_file):
    site_perf = analyze_by_site(df)
    plt.figure(figsize=(10, 6))
    site_perf[["Revenu", "Bénéfice"]].plot(kind="bar")
    plt.title("Comparaison des performances financières par site")
    plt.ylabel("Montant (€)")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

def create_satisfaction_heatmap(df_satisfaction, output_file):
    satisfaction_analysis = analyze_satisfaction(df_satisfaction)
    plt.figure(figsize=(12, 8))
    sns.heatmap(satisfaction_analysis, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".1f")
    plt.title("Carte de satisfaction client par formation")
    plt.tight_layout()
    plt.savefig(output_file)
    return output_file

# Fonction manquante dans votre code
def analyze_by_category(df):
    category_analysis = df.groupby("Catégorie").agg({
        "Revenu": "sum",
        "Bénéfice": "sum",
        "MargeNette": "mean",
        "Inscrits": "sum",
        "TauxRemplissage": "mean",
        "Satisfaction": "mean"
    }).sort_values("Bénéfice", ascending=False)
    
    return category_analysis

# Initialisation de l'application Flask
app = Flask(__name__)

# Générez les données une seule fois au démarrage
df_sessions = generate_training_data(n_sessions=100)
df_satisfaction = generate_satisfaction_data(df_sessions)

# Créez un dossier pour stocker les graphiques temporaires
os.makedirs('static/images', exist_ok=True)

@app.route('/')
def home():
    # Page d'accueil avec KPIs principaux et navigation
    kpis = calculate_kpis(df_sessions)
    
    # Générez quelques graphiques pour la page d'accueil
    revenue_chart = create_revenue_distribution_chart(df_sessions, 'static/images/repartition_ca.png')
    profitability_chart = create_profitability_chart(df_sessions, 'static/images/rentabilite_formations.png')
    
    return render_template('index.html', 
                          kpis=kpis,
                          revenue_chart='images/repartition_ca.png',
                          profitability_chart='images/rentabilite_formations.png')

@app.route('/formations')
def formations():
    # Analyse par formation
    rentabilite = analyze_profitability_by_training(df_sessions)
    top_formations = rentabilite.head(5).to_dict(orient='records')
    
    # Création des graphiques
    filling_chart = create_filling_rate_chart(df_sessions, 'static/images/taux_remplissage.png')
    
    return render_template('formations.html', 
                          top_formations=top_formations,
                          filling_chart='images/taux_remplissage.png')

@app.route('/sites')
def sites():
    # Analyse par site
    analyse_site = analyze_by_site(df_sessions)
    site_data = analyse_site.reset_index().to_dict(orient='records')
    
    # Créer un graphique comparatif
    site_chart = create_site_comparison_chart(df_sessions, 'static/images/comparaison_sites.png')
    
    return render_template('sites.html', 
                          site_data=site_data,
                          site_chart='images/comparaison_sites.png')

@app.route('/satisfaction')
def satisfaction():
    # Analyse de satisfaction
    analyse_satisfaction = analyze_satisfaction(df_satisfaction)
    
    # Création de la heatmap
    satisfaction_chart = create_satisfaction_heatmap(df_satisfaction, 'static/images/satisfaction_client.png')
    
    return render_template('satisfaction.html',
                         satisfaction_chart='images/satisfaction_client.png')

@app.route('/report')
def report():
    # Préparer les données pour le rapport
    all_analyses = {
        "kpis": calculate_kpis(df_sessions),
        "site_performance": analyze_by_site(df_sessions),
        "opportunities": identify_opportunities_and_risks(df_sessions)["opportunities"],
        "risks": identify_opportunities_and_risks(df_sessions)["risks"],
        "commercial_performance": analyze_commercial_performance(df_sessions),
        "objectives_analysis": analyze_objectives_vs_actuals(df_sessions),
        "top_trainings_by_site": analyze_top_trainings_by_site(df_sessions)
    }
    
    return render_template('report.html', analyses=all_analyses, now=datetime.now)

if __name__ == '__main__':
    # Pour le développement local
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))