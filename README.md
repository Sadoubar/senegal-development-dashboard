# 🇸🇳 Dashboard Développement Sénégal

> **Analyse multidimensionnelle du développement du Sénégal (1960-2024)**  
> Exploration interactive de 1,516 indicateurs de la Banque Mondiale

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📊 Vue d'Ensemble

Ce dashboard interactif propose une analyse approfondie du développement du Sénégal à travers quatre dimensions clés :

- **💰 Économie** : PIB, croissance, investissements, commerce
- **👥 Social** : Santé, éducation, démographie, urbanisation  
- **⚡ Énergie** : Accès électricité, renouvelables, consommation
- **🏛️ Gouvernance** : Démocratie, efficacité, état de droit

## 🚀 Fonctionnalités

### 📈 Analyses Disponibles
- **Vue d'ensemble** : KPIs essentiels et tendances inter-domaines
- **Analyse sectorielle** : Focus approfondi par domaine
- **Matrices de corrélation** : Découverte d'interactions cachées
- **Tendances temporelles** : Comparaisons multi-indicateurs
- **Recherche libre** : Exploration des 1,516 indicateurs

### 🔍 Capacités Analytiques
- **Corrélations intelligentes** avec seuils ajustables
- **Visualisations interactives** (Plotly)
- **Lignes de tendance** polynomiales
- **Normalisation** pour comparaisons multi-échelles
- **Export données** CSV

## 💡 Insights Clés Découverts

### 🔗 Corrélations Remarquables
- **Santé ↔ Urbanisation** (r=0.958) : L'urbanisation améliore l'accès aux soins
- **PIB ↔ PIB/hab** (r=0.990) : Croissance économique inclusive
- **Mortalité ↔ Espérance de vie** (r=-0.980) : Efficacité des politiques de santé

### 📊 Modèle de Développement Identifié
1. **Transition démographique** réussie (baisse mortalité + urbanisation)
2. **Croissance inclusive** (PIB et bien-être progressent ensemble)  
3. **Cercle vertueux** : Santé → Urbanisation → Développement
4. **Investissements stratégiques** (IDE corrélés aux infrastructures)

## 🛠️ Installation Locale

### Prérequis
```bash
Python 3.8+
pip install -r requirements.txt
```

### Dépendances
```txt
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
numpy>=1.24.0
scipy>=1.10.0
openpyxl>=3.1.0
xlrd>=2.0.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

### Lancement
```bash
# Cloner le repo
git clone https://github.com/votre-username/senegal-dashboard
cd senegal-dashboard

# Installer les dépendances
pip install -r requirements.txt

# Placer le fichier de données
# API_SEN_DS2_en_excel_v2_21920.xls dans le dossier racine

# Lancer l'application
streamlit run main.py
```

## ☁️ Déploiement Streamlit Cloud

### Étapes de Déploiement

1. **Fork ce repository** sur votre GitHub

2. **Connectez-vous** à [share.streamlit.io](https://share.streamlit.io)

3. **Déployez** avec ces paramètres :
   - **Repository** : `votre-username/senegal-dashboard`
   - **Branch** : `main`
   - **Main file path** : `main.py`

4. **Ajoutez les secrets** si nécessaire (aucun requis pour cette app)

### Structure Recommandée
```
senegal-dashboard/
├── main.py                              # Application principale
├── requirements.txt                     # Dépendances Python
├── README.md                           # Documentation
├── API_SEN_DS2_en_excel_v2_21920.xls  # Données Banque Mondiale
└── .streamlit/
    └── config.toml                     # Configuration Streamlit
```

### Configuration Optimale (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#1e3d59"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false
```

## 📊 Sources de Données

- **Source principale** : [Banque Mondiale - Sénégal](https://data.worldbank.org/country/senegal)
- **Période couverte** : 1960-2024 (65 années)
- **Nombre d'indicateurs** : 1,516
- **Dernière mise à jour** : Juin 2025
- **Fréquence** : Annuelle

### Indicateurs Clés Analysés
- **Économiques** : PIB, chômage, IDE, inflation, commerce
- **Sociaux** : Espérance de vie, éducation, pauvreté, démographie
- **Énergétiques** : Accès électricité, renouvelables, consommation
- **Gouvernance** : Démocratie, efficacité, corruption, état de droit

## 🎯 Cas d'Usage

### 👨‍💼 Décideurs Politiques
- **Suivi KPIs** nationaux en temps réel
- **Identification** des leviers d'action prioritaires
- **Évaluation** de l'efficacité des politiques publiques

### 📚 Chercheurs & Académiques
- **Analyse** des patterns de développement
- **Validation** d'hypothèses de recherche
- **Données** pour publications scientifiques

### 🏢 Organisations Internationales
- **Benchmarking** régional et continental
- **Suivi** des Objectifs de Développement Durable
- **Évaluation** de l'impact des programmes

## 🔧 Méthodologie

### Traitement des Données
- **Nettoyage** : Suppression valeurs aberrantes et doublons
- **Normalisation** : Min-max scaling pour comparaisons
- **Interpolation** : Méthode linéaire pour données manquantes ponctuelles
- **Agrégation** : Moyennes pondérées pour indicateurs composites

### Calculs Statistiques
- **Corrélations** : Coefficient de Pearson (seuil |r| > 0.5)
- **Tendances** : Régression polynomiale degré 2
- **Significativité** : Test t de Student (p < 0.05)
- **Robustesse** : Bootstrap avec 1000 réplicats

## 👨‍💻 Auteur

**Sadou BARRY**
- 🔗 LinkedIn : [sadou-barry-881868164](https://www.linkedin.com/in/sadou-barry-881868164/)
- 📧 Contact : Via LinkedIn
- 🎓 Spécialisation : Data Science & ENERGIE

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. **Fork** le projet
2. **Créez** une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. **Committez** (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. **Push** (`git push origin feature/nouvelle-fonctionnalite`)
5. **Ouvrez** une Pull Request

### Idées de Contributions
- 🌍 **Comparaisons régionales** (Afrique de l'Ouest)
- 🤖 **Prédictions ML** pour projections futures
- 📱 **Version mobile** optimisée
- 🔄 **API temps réel** Banque Mondiale
- 📊 **Nouveaux indicateurs** (ODD, climat)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **Banque Mondiale** pour la qualité et l'accessibilité des données
- **Communauté Streamlit** pour la plateforme exceptionnelle
- **Gouvernement du Sénégal** pour la transparence des données publiques

---

<div align="center">

**🇸🇳 Fait avec ❤️ pour le développement du Sénégal**

[🚀 Lancer l'App](https://your-app-url.streamlit.app) • [📊 Données](https://data.worldbank.org/country/senegal) • [👨‍💻 Auteur](https://www.linkedin.com/in/sadou-barry-881868164/)

</div>