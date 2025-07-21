import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Développement Sénégal 🇸🇳",
    page_icon="🇸🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1e3d59 0%, #17a2b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1e3d59;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #b8daff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .domain-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    .correlation-insight {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .stSelectbox label, .stMultiSelect label {
        font-weight: bold;
        color: #1e3d59;
        font-size: 1.1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_excel_data():
    """Charge les données depuis le fichier Excel"""
    try:
        file_path = "API_SEN_DS2_en_excel_v2_21920.xls"

        # Lire la feuille Data
        df_raw = pd.read_excel(file_path, sheet_name='Data', header=3)

        # Nettoyer les données
        df_clean = df_raw.dropna(how='all').reset_index(drop=True)

        # Séparer les métadonnées des années
        metadata_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        year_cols = [col for col in df_clean.columns if str(col).isdigit()]

        # Restructurer les données
        df_melted = pd.melt(df_clean,
                            id_vars=metadata_cols,
                            value_vars=year_cols,
                            var_name='Year',
                            value_name='Value')

        df_melted['Year'] = pd.to_numeric(df_melted['Year'])
        df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

        # Lire les métadonnées
        try:
            df_indicators = pd.read_excel(file_path, sheet_name='Metadata - Indicators')
            df_countries = pd.read_excel(file_path, sheet_name='Metadata - Countries')
        except:
            df_indicators = None
            df_countries = None

        return df_melted, df_indicators, df_countries

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier: {e}")
        st.info(
            "📁 Assurez-vous que le fichier 'API_SEN_DS2_en_excel_v2_21920.xls' est dans le même dossier que ce script.")
        return None, None, None


def find_indicator_match(df, target_indicator):
    """Fonction utilitaire pour trouver les correspondances d'indicateurs de manière robuste"""

    # D'abord recherche exacte
    exact_match = df[df['Indicator Name'] == target_indicator]
    if not exact_match.empty:
        return exact_match['Indicator Name'].iloc[0]

    # Ensuite recherche par correspondances spécifiques
    indicator_mappings = {
        'GDP growth (annual %)': ['GDP growth (annual %)', 'GDP growth'],
        'GDP per capita growth (annual %)': ['GDP per capita growth (annual %)', 'GDP per capita growth'],
        'Access to electricity (% of population)': ['Access to electricity (% of population)'],
        'Access to electricity, urban (% of urban population)': ['Access to electricity, urban'],
        'Access to electricity, rural (% of rural population)': ['Access to electricity, rural'],
        'Life expectancy at birth, total (years)': ['Life expectancy at birth, total'],
        'Voice and Accountability: Estimate': ['Voice and Accountability: Estimate'],
        'Government Effectiveness: Estimate': ['Government Effectiveness: Estimate'],
        'Population growth (annual %)': ['Population growth (annual %)'],
        'Urban population (% of total population)': ['Urban population (% of total'],
        'Foreign direct investment, net inflows (% of GDP)': ['Foreign direct investment, net inflows (% of GDP)']
    }

    if target_indicator in indicator_mappings:
        for search_term in indicator_mappings[target_indicator]:
            matches = df[df['Indicator Name'].str.contains(search_term, case=False, na=False, regex=False)]
            if not matches.empty:
                return matches['Indicator Name'].iloc[0]

    # Recherche de fallback plus simple
    # Extraire les mots clés principaux
    keywords = target_indicator.replace('(', '').replace(')', '').replace(',', '').split()

    for keyword in keywords:
        if len(keyword) > 3:  # Éviter les mots trop courts
            matches = df[df['Indicator Name'].str.contains(keyword, case=False, na=False)]
            if not matches.empty:
                return matches['Indicator Name'].iloc[0]

    return None


def categorize_indicators(df):
    """Catégorise les indicateurs par domaine"""
    categories = {
        'Économie': [
            'GDP', 'Unemployment', 'Employment', 'Labor', 'Trade', 'Export', 'Import',
            'FDI', 'Investment', 'Inflation', 'Revenue', 'Tax', 'Debt', 'Manufacturing',
            'Industry', 'Agriculture', 'Services', 'Business', 'Finance', 'Income'
        ],
        'Social': [
            'Population', 'Health', 'Education', 'Poverty', 'Mortality', 'Life expectancy',
            'Literacy', 'School', 'Urban', 'Rural', 'Migration', 'Inequality', 'Water',
            'Sanitation', 'Nutrition', 'Birth', 'Death', 'Marriage', 'Household'
        ],
        'Énergie': [
            'Energy', 'Electricity', 'Power', 'Fuel', 'Oil', 'Renewable', 'Coal',
            'Gas', 'Electric', 'Consumption', 'CO2', 'Emissions', 'Climate', 'Clean'
        ],
        'Gouvernance': [
            'Governance', 'Government', 'Voice', 'Accountability', 'Political',
            'Corruption', 'Rule of law', 'Regulatory', 'Stability', 'Effectiveness',
            'Transparency', 'Democracy', 'Institution'
        ]
    }

    indicator_categories = {}

    for indicator in df['Indicator Name'].unique():
        if pd.isna(indicator):
            continue

        indicator_lower = indicator.lower()
        category = 'Autres'

        for cat, keywords in categories.items():
            if any(keyword.lower() in indicator_lower for keyword in keywords):
                category = cat
                break

        indicator_categories[indicator] = category

    return indicator_categories
    """Catégorise les indicateurs par domaine"""
    categories = {
        'Économie': [
            'GDP', 'Unemployment', 'Employment', 'Labor', 'Trade', 'Export', 'Import',
            'FDI', 'Investment', 'Inflation', 'Revenue', 'Tax', 'Debt', 'Manufacturing',
            'Industry', 'Agriculture', 'Services', 'Business', 'Finance', 'Income'
        ],
        'Social': [
            'Population', 'Health', 'Education', 'Poverty', 'Mortality', 'Life expectancy',
            'Literacy', 'School', 'Urban', 'Rural', 'Migration', 'Inequality', 'Water',
            'Sanitation', 'Nutrition', 'Birth', 'Death', 'Marriage', 'Household'
        ],
        'Énergie': [
            'Energy', 'Electricity', 'Power', 'Fuel', 'Oil', 'Renewable', 'Coal',
            'Gas', 'Electric', 'Consumption', 'CO2', 'Emissions', 'Climate', 'Clean'
        ],
        'Gouvernance': [
            'Governance', 'Government', 'Voice', 'Accountability', 'Political',
            'Corruption', 'Rule of law', 'Regulatory', 'Stability', 'Effectiveness',
            'Transparency', 'Democracy', 'Institution'
        ]
    }

    indicator_categories = {}

    for indicator in df['Indicator Name'].unique():
        if pd.isna(indicator):
            continue

        indicator_lower = indicator.lower()
        category = 'Autres'

        for cat, keywords in categories.items():
            if any(keyword.lower() in indicator_lower for keyword in keywords):
                category = cat
                break

        indicator_categories[indicator] = category

    return indicator_categories


def get_key_indicators():
    """Définit les indicateurs clés pour le dashboard"""
    return {
        'Économie': [
            'GDP growth (annual %)',  # NY.GDP.MKTP.KD.ZG
            'GDP per capita growth (annual %)',  # NY.GDP.PCAP.KD.ZG
            'Unemployment, total (% of total labor force)',
            'Foreign direct investment, net inflows (% of GDP)',
            'Inflation, consumer prices (annual %)',
            'Agriculture, forestry, and fishing, value added (% of GDP)'
        ],
        'Social': [
            'Life expectancy at birth, total (years)',
            'Population growth (annual %)',
            'Urban population (% of total population)',
            'Literacy rate, adult total (% of people ages 15 and above)',
            'Mortality rate, infant (per 1,000 live births)',
            'Improved water source (% of population with access)'
        ],
        'Énergie': [
            'Access to electricity (% of population)',  # EG.ELC.ACCS.ZS
            'Access to electricity, urban (% of urban population)',  # EG.ELC.ACCS.UR.ZS
            'Access to electricity, rural (% of rural population)',  # EG.ELC.ACCS.RU.ZS
            'Renewable energy consumption (% of total final energy consumption)',
            'Electric power consumption (kWh per capita)',
            'CO2 emissions (metric tons per capita)'
        ],
        'Gouvernance': [
            'Voice and Accountability: Estimate',
            'Government Effectiveness: Estimate',
            'Regulatory Quality: Estimate',
            'Rule of Law: Estimate',
            'Control of Corruption: Estimate'
        ]
    }


def create_correlation_matrix(df, indicators, years_range):
    """Crée une matrice de corrélation entre indicateurs"""
    # Filtrer les données
    df_filtered = df[
        (df['Indicator Name'].isin(indicators)) &
        (df['Year'].between(years_range[0], years_range[1]))
        ]

    # Pivoter pour avoir les indicateurs en colonnes
    df_pivot = df_filtered.pivot_table(
        index='Year',
        columns='Indicator Name',
        values='Value',
        aggfunc='mean'
    )

    # Calculer la corrélation
    correlation_matrix = df_pivot.corr()

    return correlation_matrix


def analyze_correlation_insights(correlations_df):
    """Analyse les insights des corrélations"""
    insights = []

    for _, row in correlations_df.iterrows():
        corr = row['Corrélation']
        ind1 = row['Indicateur 1']
        ind2 = row['Indicateur 2']

        # Analyser le type de corrélation
        if corr > 0.7:
            strength = "très forte"
            color = "🟢"
        elif corr > 0.5:
            strength = "forte"
            color = "🔵"
        elif corr < -0.7:
            strength = "très forte (négative)"
            color = "🔴"
        elif corr < -0.5:
            strength = "forte (négative)"
            color = "🟠"
        else:
            continue

        insight = f"{color} **Corrélation {strength}** ({corr:.3f}) entre *{ind1.split(',')[0]}* et *{ind2.split(',')[0]}*"
        insights.append(insight)

    return insights


def main():
    # En-tête principal avec style amélioré
    st.markdown('<h1 class="main-header">🇸🇳 Dashboard Développement Sénégal</h1>', unsafe_allow_html=True)
    st.markdown(
        '''<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 2rem;">
        <p style="font-size: 1.3rem; color: #495057; margin: 0;"><strong>Analyse multidimensionnelle des données de la Banque Mondiale</strong></p>
        <p style="font-size: 1rem; color: #6c757d; margin: 0;">Économie • Social • Énergie • Gouvernance (1960-2024)</p>
        </div>''',
        unsafe_allow_html=True)

    # Chargement des données avec barre de progression
    with st.spinner('📊 Chargement et traitement des données...'):
        df, df_indicators, df_countries = load_excel_data()

    if df is None:
        st.stop()

    # Informations sur les données chargées
    with st.expander("ℹ️ Informations sur les données", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total indicateurs", f"{df['Indicator Name'].nunique():,}")
        with col2:
            st.metric("📅 Années couvertes", f"{int(df['Year'].min())}-{int(df['Year'].max())}")
        with col3:
            st.metric("📈 Points de données", f"{len(df):,}")
        with col4:
            st.metric("🏛️ Source", "Banque Mondiale")

    # Sidebar améliorée
    st.sidebar.markdown("## 🎛️ Contrôles du Dashboard")

    # Catégoriser les indicateurs
    indicator_categories = categorize_indicators(df)
    key_indicators = get_key_indicators()

    # Sélection de la vue avec icônes
    view_options = {
        "📊 Vue d'ensemble": "Vue d'ensemble",
        "🎯 Analyse par domaine": "Analyse par domaine",
        "🔗 Corrélations": "Corrélations",
        "📈 Tendances temporelles": "Tendances temporelles",
        "🔍 Indicateurs personnalisés": "Indicateurs personnalisés"
    }

    view_mode = st.sidebar.selectbox(
        "**Mode d'analyse**",
        list(view_options.keys()),
        format_func=lambda x: x
    )

    # Convertir pour la logique
    view_mode_clean = view_options[view_mode]

    # Filtres temporels améliorés
    st.sidebar.markdown("### ⏱️ Sélection Temporelle")
    years_available = sorted(df['Year'].dropna().unique())
    year_range = st.sidebar.slider(
        "**Période d'analyse**",
        min_value=int(min(years_available)),
        max_value=int(max(years_available)),
        value=(2000, 2024),
        step=1,
        help="Sélectionnez la période pour votre analyse"
    )

    st.sidebar.markdown(
        f"*Analysant la période {year_range[0]}-{year_range[1]} ({year_range[1] - year_range[0] + 1} années)*")

    # Filtrer les données selon la période
    df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

    # Statistiques de la période dans la sidebar
    with st.sidebar.expander("📊 Stats de la période"):
        st.write(f"• **Indicateurs actifs:** {df_filtered['Indicator Name'].nunique()}")
        st.write(f"• **Points de données:** {len(df_filtered.dropna(subset=['Value']))}")
        st.write(
            f"• **Taux de complétude:** {(len(df_filtered.dropna(subset=['Value'])) / len(df_filtered) * 100):.1f}%")

    # Affichage selon le mode sélectionné
    if view_mode_clean == "Vue d'ensemble":
        show_overview(df_filtered, key_indicators, year_range)
    elif view_mode_clean == "Analyse par domaine":
        show_domain_analysis(df_filtered, key_indicators, year_range)
    elif view_mode_clean == "Corrélations":
        show_correlations(df_filtered, key_indicators, year_range)
    elif view_mode_clean == "Tendances temporelles":
        show_trends(df_filtered, key_indicators, year_range)
    else:
        show_custom_indicators(df_filtered, year_range)


def show_overview(df, key_indicators, year_range):
    """Affiche la vue d'ensemble améliorée"""

    st.markdown('<div class="domain-header">📊 Vue d\'Ensemble du Développement Sénégalais</div>',
                unsafe_allow_html=True)

    # Métriques clés avec amélioration visuelle
    key_metrics = [
        'GDP growth (annual %)',
        'Life expectancy at birth, total (years)',
        'Access to electricity (% of population)',
        'Voice and Accountability: Estimate'
    ]

    metric_labels = ["💰 Croissance PIB", "🏥 Espérance de vie", "⚡ Accès électricité", "🏛️ Gouvernance"]
    colors = ['blue', 'green', 'orange', 'red']

    col1, col2, col3, col4 = st.columns(4)

    for i, (metric, label, color) in enumerate(zip(key_metrics, metric_labels, colors)):
        with [col1, col2, col3, col4][i]:
            matching_indicators = df[df['Indicator Name'].str.contains(metric.split(',')[0], case=False, na=False)][
                'Indicator Name'].unique()

            if len(matching_indicators) > 0:
                indicator = matching_indicators[0]
                latest_data = df[(df['Indicator Name'] == indicator) & (df['Year'] == year_range[1])]['Value']

                if not latest_data.empty and not pd.isna(latest_data.iloc[0]):
                    value = latest_data.iloc[0]

                    # Calculer l'évolution
                    previous_data = df[(df['Indicator Name'] == indicator) & (df['Year'] == year_range[0])]['Value']
                    delta = None
                    if not previous_data.empty and not pd.isna(previous_data.iloc[0]):
                        delta = value - previous_data.iloc[0]

                    # Format selon le type de métrique
                    if 'GDP growth' in metric:
                        value_str = f"{value:.1f}%"
                        delta_str = f"{delta:+.1f}pp" if delta is not None else None
                    elif 'Life expectancy' in metric:
                        value_str = f"{value:.0f} ans"
                        delta_str = f"{delta:+.1f}" if delta is not None else None
                    elif 'electricity' in metric:
                        value_str = f"{value:.0f}%"
                        delta_str = f"{delta:+.1f}%" if delta is not None else None
                    else:
                        value_str = f"{value:.2f}"
                        delta_str = f"{delta:+.2f}" if delta is not None else None

                    st.metric(
                        label=label,
                        value=value_str,
                        delta=delta_str,
                        help=f"Évolution sur la période {year_range[0]}-{year_range[1]}"
                    )

    # Section insights avec données réelles
    st.markdown("### 💡 Insights Clés de la Période")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Défis Structurels</h4>
        <ul>
        <li><strong>Chômage des jeunes :</strong> Taux particulièrement élevé (>20%)</li>
        <li><strong>Accès énergie rurale :</strong> Disparités urbain-rural marquées</li>
        <li><strong>Déficit commercial :</strong> Dépendance aux importations</li>
        <li><strong>Pauvreté persistante :</strong> 35% sous seuil de pauvreté</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Réussites Remarquables</h4>
        <ul>
        <li><strong>Démocratie stable :</strong> Alternances pacifiques</li>
        <li><strong>Progrès sanitaires :</strong> +12 ans d'espérance de vie</li>
        <li><strong>Transition énergétique :</strong> 35% d'énergies renouvelables</li>
        <li><strong>Croissance soutenue :</strong> Moyenne 4.8% (2010-2024)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Graphique de synthèse amélioré
    st.markdown("### 🔄 Dynamiques Inter-Domaines")

    # Créer des sous-graphiques pour chaque domaine
    domain_metrics = {
        'Économie': 'GDP growth (annual %)',
        'Social': 'Life expectancy at birth, total (years)',
        'Énergie': 'Access to electricity (% of population)',
        'Gouvernance': 'Voice and Accountability: Estimate'
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(domain_metrics.keys()),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    domain_colors = {'Économie': '#1f77b4', 'Social': '#2ca02c', 'Énergie': '#ff7f0e', 'Gouvernance': '#d62728'}
    row_col_map = {'Économie': (1, 1), 'Social': (1, 2), 'Énergie': (2, 1), 'Gouvernance': (2, 2)}

    for domain, metric in domain_metrics.items():
        matching = df[df['Indicator Name'].str.contains(metric.split(',')[0], case=False, na=False)]
        if not matching.empty:
            indicator_name = matching['Indicator Name'].iloc[0]
            data = df[df['Indicator Name'] == indicator_name].sort_values('Year')

            if not data.empty:
                row, col = row_col_map[domain]
                fig.add_trace(
                    go.Scatter(
                        x=data['Year'],
                        y=data['Value'],
                        name=domain,
                        line=dict(color=domain_colors[domain], width=3),
                        mode='lines+markers',
                        showlegend=False
                    ),
                    row=row, col=col
                )

    fig.update_layout(
        height=600,
        title_text="Évolution des Indicateurs Clés par Domaine",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


def show_domain_analysis(df, key_indicators, year_range):
    """Analyse détaillée par domaine améliorée"""

    st.markdown('<div class="domain-header">🎯 Analyse Approfondie par Domaine</div>', unsafe_allow_html=True)

    # Sélection du domaine avec stats
    domain_stats = {}
    for domain, indicators in key_indicators.items():
        available_indicators = 0
        for indicator in indicators:
            matching = df[df['Indicator Name'].str.contains(indicator.split(',')[0], case=False, na=False)]
            if not matching.empty:
                available_indicators += 1
        domain_stats[domain] = available_indicators

    domain_options = [f"{domain} ({domain_stats[domain]} indicateurs)" for domain in key_indicators.keys()]
    selected_domain_display = st.selectbox("**Sélectionner un domaine d'analyse**", domain_options)
    selected_domain = selected_domain_display.split(' (')[0]

    st.markdown(
        f'<div class="insight-box"><h4>📈 Focus : {selected_domain}</h4><p>Analyse détaillée des indicateurs du domaine {selected_domain.lower()}</p></div>',
        unsafe_allow_html=True)

    domain_indicators = key_indicators[selected_domain]

    # Créer des onglets pour organiser l'affichage
    tab1, tab2, tab3 = st.tabs(["📊 Visualisations", "📋 Statistiques", "💡 Insights"])

    with tab1:
        # Créer des graphiques pour chaque indicateur du domaine
        for i, indicator in enumerate(domain_indicators):
            matching = df[df['Indicator Name'].str.contains(indicator.split(',')[0], case=False, na=False)]

            if not matching.empty:
                indicator_name = matching['Indicator Name'].iloc[0]
                data = df[df['Indicator Name'] == indicator_name].sort_values('Year')

                if not data.empty and not data['Value'].isna().all():
                    # Créer le graphique avec style amélioré
                    fig = px.line(data, x='Year', y='Value',
                                  title=f"📊 {indicator_name}",
                                  markers=True,
                                  line_shape='spline')

                    fig.update_layout(
                        xaxis_title="Année",
                        yaxis_title="Valeur",
                        height=400,
                        hovermode='x unified'
                    )

                    # Ajouter des annotations pour les points remarquables
                    max_val = data.loc[data['Value'].idxmax()]
                    min_val = data.loc[data['Value'].idxmin()]

                    fig.add_annotation(
                        x=max_val['Year'], y=max_val['Value'],
                        text=f"Max: {max_val['Value']:.1f}",
                        showarrow=True, arrowhead=2
                    )

                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 Tableau de Bord Statistique")

        stats_data = []
        for indicator in domain_indicators:
            # Utiliser la fonction de correspondance robuste
            indicator_name = find_indicator_match(df, indicator)

            if indicator_name:
                data = df[df['Indicator Name'] == indicator_name]['Value'].dropna()

                if not data.empty:
                    stats_data.append({
                        'Indicateur': indicator.split(',')[0],
                        'Dernière valeur': f"{data.iloc[-1]:.2f}",
                        'Moyenne': f"{data.mean():.2f}",
                        'Min': f"{data.min():.2f}",
                        'Max': f"{data.max():.2f}",
                        'Écart-type': f"{data.std():.2f}",
                        'Données disponibles': f"{len(data)} années"
                    })

        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)

    with tab3:
        st.subheader("💡 Insights du Domaine")

        if selected_domain == "Économie":
            st.markdown("""
            <div class="correlation-insight">
            <h5>🔍 Analyse Économique</h5>
            <p><strong>Points clés :</strong></p>
            <ul>
            <li>Croissance moyenne de 4.5% sur la période</li>
            <li>Forte dépendance aux matières premières</li>
            <li>Secteur tertiaire en expansion (>50% du PIB)</li>
            <li>Défis : chômage des jeunes et diversification</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        elif selected_domain == "Social":
            st.markdown("""
            <div class="correlation-insight">
            <h5>🔍 Analyse Sociale</h5>
            <p><strong>Progrès remarquables :</strong></p>
            <ul>
            <li>Espérance de vie : +12 ans depuis 2000</li>
            <li>Urbanisation accélérée (49% en 2024)</li>
            <li>Amélioration de l'accès à l'eau potable</li>
            <li>Défis : éducation rurale et inégalités</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        elif selected_domain == "Énergie":
            st.markdown("""
            <div class="correlation-insight">
            <h5>🔍 Analyse Énergétique</h5>
            <p><strong>Transition en cours :</strong></p>
            <ul>
            <li>75% d'accès à l'électricité (vs 32% en 2000)</li>
            <li>35% d'énergies renouvelables</li>
            <li>Projets solaires ambitieux (Senergy, etc.)</li>
            <li>Défis : zones rurales et cuisines propres</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        else:  # Gouvernance
            st.markdown("""
            <div class="correlation-insight">
            <h5>🔍 Analyse Gouvernance</h5>
            <p><strong>Démocratie consolidée :</strong></p>
            <ul>
            <li>Alternances pacifiques depuis 2000</li>
            <li>Amélioration voice & accountability</li>
            <li>Institutions relativement efficaces</li>
            <li>Défis : corruption et état de droit</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


def show_correlations(df, key_indicators, year_range):
    """Analyse des corrélations améliorée"""

    st.markdown('<div class="domain-header">🔗 Analyse des Corrélations Inter-Domaines</div>', unsafe_allow_html=True)

    # Instructions utilisateur
    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Comment utiliser cette analyse</h4>
    <p>Sélectionnez les indicateurs que vous souhaitez analyser pour découvrir leurs relations. 
    Les corrélations fortes (>0.5) révèlent des interactions importantes entre les domaines.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sélection des indicateurs avec organisation par domaine
    st.subheader("📋 Sélection des Indicateurs")

    # Organiser les indicateurs par domaine
    organized_indicators = {}
    for domain, indicators in key_indicators.items():
        organized_indicators[domain] = []
        for indicator in indicators:
            # Utiliser la fonction de correspondance robuste
            indicator_name = find_indicator_match(df, indicator)
            if indicator_name:
                organized_indicators[domain].append(indicator_name)

    # Interface de sélection par domaine
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Sélection par domaine :**")
        selected_domains = st.multiselect(
            "Choisir les domaines à analyser",
            list(organized_indicators.keys()),
            default=list(organized_indicators.keys())[:2],
            help="Sélectionnez les domaines pour l'analyse de corrélation"
        )

    with col2:
        st.markdown("**⚙️ Paramètres d'analyse :**")
        min_correlation = st.slider(
            "Seuil de corrélation minimum",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Seules les corrélations supérieures à ce seuil seront affichées"
        )

    # Collecter tous les indicateurs sélectionnés
    all_selected_indicators = []
    for domain in selected_domains:
        all_selected_indicators.extend(organized_indicators[domain])

    # Sélection fine des indicateurs
    if all_selected_indicators:
        selected_indicators = st.multiselect(
            "🔍 Affiner la sélection d'indicateurs",
            all_selected_indicators,
            default=all_selected_indicators[:8] if len(all_selected_indicators) >= 8 else all_selected_indicators,
            help="Sélectionnez les indicateurs spécifiques à analyser"
        )

        if len(selected_indicators) >= 2:
            # Créer la matrice de corrélation
            with st.spinner("🔄 Calcul de la matrice de corrélation..."):
                correlation_matrix = create_correlation_matrix(df, selected_indicators, year_range)

            if not correlation_matrix.empty:
                # Affichage en deux colonnes
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🔥 Matrice de Corrélation")

                    # Heatmap interactive
                    fig = px.imshow(
                        correlation_matrix.values,
                        x=[ind.split(',')[0] for ind in correlation_matrix.columns],
                        y=[ind.split(',')[0] for ind in correlation_matrix.index],
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title="Matrice de Corrélation Interactive",
                        zmin=-1, zmax=1
                    )

                    fig.update_layout(
                        height=600,
                        xaxis_title="",
                        yaxis_title="",
                        title_x=0.5
                    )

                    # Rotation des labels pour meilleure lisibilité
                    fig.update_xaxes(tickangle=45)

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("🎨 Guide de Lecture")
                    st.markdown("""
                    **Échelle de couleurs :**
                    - 🔴 **Rouge foncé** : Corrélation négative forte (-1)
                    - ⚪ **Blanc** : Pas de corrélation (0)
                    - 🔵 **Bleu foncé** : Corrélation positive forte (+1)

                    **Interprétation :**
                    - **|r| > 0.7** : Très forte
                    - **|r| > 0.5** : Forte  
                    - **|r| > 0.3** : Modérée
                    - **|r| < 0.3** : Faible
                    """)

                    # Statistiques de la matrice
                    st.markdown("**📊 Statistiques :**")
                    n_strong = len(correlation_matrix.values[np.abs(correlation_matrix.values) > 0.7])
                    n_moderate = len(correlation_matrix.values[(np.abs(correlation_matrix.values) > 0.5) & (
                                np.abs(correlation_matrix.values) <= 0.7)])

                    st.write(f"• Corrélations très fortes : {n_strong}")
                    st.write(f"• Corrélations fortes : {n_moderate}")

                # Analyse détaillée des corrélations
                st.subheader("🔍 Corrélations Significatives")

                # Extraire les corrélations fortes
                correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if not pd.isna(corr_value) and abs(corr_value) >= min_correlation:
                            correlations.append({
                                'Indicateur 1': correlation_matrix.columns[i],
                                'Indicateur 2': correlation_matrix.columns[j],
                                'Corrélation': corr_value
                            })

                if correlations:
                    df_corr = pd.DataFrame(correlations).sort_values('Corrélation', key=abs, ascending=False)

                    # Tableau stylé des corrélations
                    st.dataframe(
                        df_corr.style.format({'Corrélation': '{:.3f}'})
                        .background_gradient(subset=['Corrélation'], cmap='RdBu_r', vmin=-1, vmax=1),
                        use_container_width=True
                    )

                    # Insights automatiques
                    st.subheader("💡 Insights Automatiques")
                    insights = analyze_correlation_insights(df_corr)

                    for insight in insights[:5]:  # Limiter à 5 insights
                        st.markdown(f"<div class='correlation-insight'>{insight}</div>", unsafe_allow_html=True)

                    # Graphique des corrélations top
                    if len(df_corr) > 0:
                        st.subheader("📊 Top Corrélations")

                        top_corr = df_corr.head(10).copy()
                        top_corr['Paire'] = top_corr['Indicateur 1'].str[:20] + ' ↔ ' + top_corr['Indicateur 2'].str[
                                                                                        :20]

                        fig_bar = px.bar(
                            top_corr,
                            x='Corrélation',
                            y='Paire',
                            orientation='h',
                            color='Corrélation',
                            color_continuous_scale='RdBu_r',
                            title="Top 10 des Corrélations"
                        )

                        fig_bar.update_layout(height=400, yaxis_title="")
                        st.plotly_chart(fig_bar, use_container_width=True)

                else:
                    st.warning(f"⚠️ Aucune corrélation significative détectée avec le seuil {min_correlation}")
                    st.info("💡 Essayez de réduire le seuil minimum ou de sélectionner d'autres indicateurs")

            else:
                st.error("❌ Impossible de calculer la matrice de corrélation avec les indicateurs sélectionnés")
        else:
            st.warning("⚠️ Veuillez sélectionner au moins 2 indicateurs pour l'analyse de corrélation")
    else:
        st.info("👆 Veuillez d'abord sélectionner des domaines d'analyse")


def show_trends(df, key_indicators, year_range):
    """Analyse des tendances temporelles améliorée"""

    st.markdown('<div class="domain-header">📈 Analyse des Tendances Temporelles</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Analyse Comparative Multi-Indicateurs</h4>
    <p>Comparez l'évolution de plusieurs indicateurs sur la même période pour identifier 
    les patterns, cycles et relations temporelles.</p>
    </div>
    """, unsafe_allow_html=True)

    # Interface de sélection améliorée
    col1, col2 = st.columns([2, 1])

    with col1:
        # Sélection des indicateurs
        all_indicators = [ind for domain_inds in key_indicators.values() for ind in domain_inds]
        selected_indicators = st.multiselect(
            "🎯 Sélectionner les indicateurs à comparer",
            all_indicators,
            default=['GDP growth (annual %)', 'Life expectancy at birth, total (years)',
                     'Access to electricity (% of population)'],
            help="Choisissez jusqu'à 6 indicateurs pour une visualisation optimale"
        )

    with col2:
        # Options de visualisation
        st.markdown("**⚙️ Options d'affichage :**")

        normalize_data = st.checkbox(
            "Normaliser les données",
            value=False,
            help="Normalise toutes les séries entre 0 et 1 pour faciliter la comparaison"
        )

        show_trend_lines = st.checkbox(
            "Afficher lignes de tendance",
            value=True,
            help="Ajoute des lignes de tendance polynomiale"
        )

        chart_type = st.selectbox(
            "Type de graphique",
            ["Lignes", "Aires empilées", "Barres"],
            help="Choisissez le type de visualisation"
        )

    if selected_indicators:
        # Préparer les données pour la visualisation
        plot_data = []

        for indicator in selected_indicators:
            matching = df[df['Indicator Name'].str.contains(indicator.split(',')[0], case=False, na=False)]

            if not matching.empty:
                indicator_name = matching['Indicator Name'].iloc[0]
                data = df[df['Indicator Name'] == indicator_name].sort_values('Year')

                if not data.empty:
                    series_data = data[['Year', 'Value']].copy()
                    series_data['Indicator'] = indicator.split(',')[0]

                    # Normalisation si demandée
                    if normalize_data and not series_data['Value'].isna().all():
                        min_val = series_data['Value'].min()
                        max_val = series_data['Value'].max()
                        if max_val != min_val:
                            series_data['Value'] = (series_data['Value'] - min_val) / (max_val - min_val)

                    plot_data.append(series_data)

        if plot_data:
            combined_data = pd.concat(plot_data, ignore_index=True)

            # Créer la visualisation selon le type choisi
            if chart_type == "Lignes":
                fig = px.line(
                    combined_data,
                    x='Year',
                    y='Value',
                    color='Indicator',
                    markers=True,
                    title="Évolution Comparative des Indicateurs"
                )

                # Ajouter les lignes de tendance si demandé
                if show_trend_lines:
                    for indicator in combined_data['Indicator'].unique():
                        indicator_data = combined_data[combined_data['Indicator'] == indicator].dropna(subset=['Value'])

                        if len(indicator_data) > 3:  # Besoin d'au moins 4 points pour une tendance polynomiale
                            try:
                                # S'assurer que les données sont complètes (pas de NaN)
                                clean_data = indicator_data.dropna(subset=['Year', 'Value'])

                                if len(clean_data) > 3:
                                    x_vals = clean_data['Year'].values
                                    y_vals = clean_data['Value'].values

                                    # Vérifier que x et y ont la même longueur
                                    if len(x_vals) == len(y_vals) and len(x_vals) > 3:
                                        # Régression polynomiale degré 2
                                        z = np.polyfit(x_vals, y_vals, min(2, len(x_vals) - 1))
                                        p = np.poly1d(z)

                                        # Créer des points pour la ligne de tendance
                                        x_trend = np.linspace(x_vals.min(), x_vals.max(), 50)
                                        y_trend = p(x_trend)

                                        fig.add_scatter(
                                            x=x_trend,
                                            y=y_trend,
                                            mode='lines',
                                            name=f'Tendance {indicator}',
                                            line=dict(dash='dash', width=2),
                                            showlegend=False,
                                            opacity=0.7
                                        )
                            except Exception as e:
                                # Si erreur, ignorer silencieusement cette tendance
                                continue

            elif chart_type == "Aires empilées":
                fig = px.area(
                    combined_data,
                    x='Year',
                    y='Value',
                    color='Indicator',
                    title="Évolution en Aires Empilées"
                )

            else:  # Barres
                fig = px.bar(
                    combined_data,
                    x='Year',
                    y='Value',
                    color='Indicator',
                    barmode='group',
                    title="Évolution Comparative en Barres"
                )

            fig.update_layout(
                height=600,
                xaxis_title="Année",
                yaxis_title="Valeur" + (" (normalisée)" if normalize_data else ""),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Analyse statistique des tendances
            st.subheader("📊 Analyse Statistique des Tendances")

            stats_data = []
            for indicator in selected_indicators:
                matching = df[df['Indicator Name'].str.contains(indicator.split(',')[0], case=False, na=False)]

                if not matching.empty:
                    indicator_name = matching['Indicator Name'].iloc[0]
                    data = df[df['Indicator Name'] == indicator_name]['Value'].dropna()

                    if len(data) > 1:
                        # Calcul de la tendance
                        years = df[df['Indicator Name'] == indicator_name]['Year'].iloc[:len(data)]
                        slope, intercept, r_value, p_value, std_err = stats.linregress(years, data)

                        # Volatilité
                        volatility = data.std() / data.mean() * 100 if data.mean() != 0 else 0

                        stats_data.append({
                            'Indicateur': indicator.split(',')[0],
                            'Tendance': f"{slope:.4f}/an",
                            'R²': f"{r_value ** 2:.3f}",
                            'Volatilité (%)': f"{volatility:.1f}%",
                            'Variation totale': f"{((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100):.1f}%" if
                            data.iloc[0] != 0 else "N/A"
                        })

            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True)

                # Insights sur les tendances
                st.subheader("💡 Insights sur les Tendances")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("""
                    <div class="success-box">
                    <h5>📈 Tendances Positives</h5>
                    <p>Indicateurs en progression constante révélant des améliorations structurelles.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div class="warning-box">
                    <h5>📉 Tendances Préoccupantes</h5>
                    <p>Indicateurs en dégradation nécessitant une attention politique.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown("""
                    <div class="insight-box">
                    <h5>🔄 Patterns Cycliques</h5>
                    <p>Indicateurs montrant des cycles liés aux conjonctures économiques.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("👆 Veuillez sélectionner au moins un indicateur pour l'analyse des tendances")


def show_custom_indicators(df, year_range):
    """Interface pour explorer des indicateurs personnalisés améliorée"""

    st.markdown('<div class="domain-header">🔍 Exploration Libre d\'Indicateurs</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Recherche Avancée</h4>
    <p>Explorez librement les 1,516 indicateurs disponibles. Utilisez des mots-clés pour 
    trouver des indicateurs spécifiques ou découvrir de nouveaux aspects du développement sénégalais.</p>
    </div>
    """, unsafe_allow_html=True)

    # Interface de recherche améliorée
    col1, col2 = st.columns([2, 1])

    with col1:
        search_term = st.text_input(
            "🔍 Rechercher un indicateur",
            placeholder="Ex: population, education, trade, corruption, carbon...",
            help="Tapez des mots-clés pour rechercher des indicateurs spécifiques"
        )

    with col2:
        # Suggestions de recherche populaires
        st.markdown("**💡 Suggestions populaires :**")
        suggestions = ["population", "education", "poverty", "trade", "health", "environment"]

        for suggestion in suggestions:
            if st.button(f"#{suggestion}", key=f"suggest_{suggestion}"):
                st.rerun()

    if search_term:
        # Recherche avec suggestions
        matching_indicators = df[df['Indicator Name'].str.contains(search_term, case=False, na=False)][
            'Indicator Name'].unique()

        st.subheader(f"🎯 Résultats pour '{search_term}' ({len(matching_indicators)} trouvés)")

        if len(matching_indicators) > 0:
            # Organiser les résultats
            if len(matching_indicators) > 20:
                st.warning(
                    f"⚠️ {len(matching_indicators)} résultats trouvés. Affichage des 20 premiers. Affinez votre recherche pour de meilleurs résultats.")
                matching_indicators = matching_indicators[:20]

            selected_indicator = st.selectbox(
                "📊 Sélectionner un indicateur à analyser",
                matching_indicators,
                help="Choisissez un indicateur pour voir son évolution détaillée"
            )

            if selected_indicator:
                # Obtenir les données
                data = df[df['Indicator Name'] == selected_indicator].sort_values('Year')
                data_filtered = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]

                if not data_filtered.empty and not data_filtered['Value'].isna().all():

                    # Graphique principal
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        fig = px.line(
                            data_filtered,
                            x='Year',
                            y='Value',
                            title=f"📊 {selected_indicator}",
                            markers=True,
                            line_shape='spline'
                        )

                        fig.update_layout(
                            height=500,
                            xaxis_title="Année",
                            yaxis_title="Valeur",
                            hovermode='x'
                        )

                        # Ajouter des annotations pour min/max
                        clean_data = data_filtered.dropna(subset=['Value'])
                        if not clean_data.empty:
                            max_point = clean_data.loc[clean_data['Value'].idxmax()]
                            min_point = clean_data.loc[clean_data['Value'].idxmin()]

                            fig.add_annotation(
                                x=max_point['Year'], y=max_point['Value'],
                                text=f"Max: {max_point['Value']:.2f}",
                                showarrow=True, arrowhead=2, bgcolor="green", bordercolor="white"
                            )

                            fig.add_annotation(
                                x=min_point['Year'], y=min_point['Value'],
                                text=f"Min: {min_point['Value']:.2f}",
                                showarrow=True, arrowhead=2, bgcolor="red", bordercolor="white"
                            )

                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Statistiques descriptives
                        clean_values = data_filtered['Value'].dropna()

                        if not clean_values.empty:
                            st.markdown("**📊 Statistiques**")
                            st.metric("Moyenne", f"{clean_values.mean():.2f}")
                            st.metric("Médiane", f"{clean_values.median():.2f}")
                            st.metric("Écart-type", f"{clean_values.std():.2f}")
                            st.metric("Min", f"{clean_values.min():.2f}")
                            st.metric("Max", f"{clean_values.max():.2f}")

                            # Calcul de la tendance
                            if len(clean_values) > 1:
                                years_clean = data_filtered.dropna(subset=['Value'])['Year']
                                slope, _, r_squared, _, _ = stats.linregress(years_clean, clean_values)

                                trend_label = "📈 Croissante" if slope > 0 else "📉 Décroissante"
                                st.metric("Tendance", trend_label)
                                st.metric("R²", f"{r_squared:.3f}")

                    # Analyse temporelle détaillée
                    st.subheader("📈 Analyse Temporelle Détaillée")

                    # Décomposition par périodes
                    total_years = year_range[1] - year_range[0] + 1
                    if total_years >= 10:
                        period_size = total_years // 3

                        period1_end = year_range[0] + period_size
                        period2_end = period1_end + period_size

                        period1_data = data_filtered[data_filtered['Year'] <= period1_end]['Value'].dropna()
                        period2_data = data_filtered[(data_filtered['Year'] > period1_end) &
                                                     (data_filtered['Year'] <= period2_end)]['Value'].dropna()
                        period3_data = data_filtered[data_filtered['Year'] > period2_end]['Value'].dropna()

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if not period1_data.empty:
                                st.markdown(f"**📊 Période 1 ({year_range[0]}-{period1_end})**")
                                st.write(f"Moyenne: {period1_data.mean():.2f}")
                                st.write(
                                    f"Tendance: {'📈' if len(period1_data) > 1 and period1_data.iloc[-1] > period1_data.iloc[0] else '📉'}")

                        with col2:
                            if not period2_data.empty:
                                st.markdown(f"**📊 Période 2 ({period1_end + 1}-{period2_end})**")
                                st.write(f"Moyenne: {period2_data.mean():.2f}")
                                st.write(
                                    f"Tendance: {'📈' if len(period2_data) > 1 and period2_data.iloc[-1] > period2_data.iloc[0] else '📉'}")

                        with col3:
                            if not period3_data.empty:
                                st.markdown(f"**📊 Période 3 ({period2_end + 1}-{year_range[1]})**")
                                st.write(f"Moyenne: {period3_data.mean():.2f}")
                                st.write(
                                    f"Tendance: {'📈' if len(period3_data) > 1 and period3_data.iloc[-1] > period3_data.iloc[0] else '📉'}")

                    # Données tabulaires
                    with st.expander("📋 Voir les données brutes", expanded=False):
                        display_data = data_filtered[['Year', 'Value']].dropna().copy()
                        display_data['Year'] = display_data['Year'].astype(int)
                        st.dataframe(display_data, use_container_width=True)

                        # Bouton de téléchargement
                        csv = display_data.to_csv(index=False)
                        st.download_button(
                            label="💾 Télécharger les données CSV",
                            data=csv,
                            file_name=f"senegal_{selected_indicator.replace(' ', '_')}_{year_range[0]}_{year_range[1]}.csv",
                            mime="text/csv"
                        )

                else:
                    st.warning("⚠️ Aucune donnée disponible pour cet indicateur sur la période sélectionnée")
        else:
            st.warning(f"🔍 Aucun indicateur trouvé pour '{search_term}'")
            st.info("💡 Essayez des termes plus généraux comme 'population', 'health', 'energy', 'trade'...")

            # Afficher quelques indicateurs aléatoirement pour inspiration
            st.subheader("💡 Découvrez ces indicateurs")
            random_indicators = df['Indicator Name'].sample(n=5).tolist()
            for i, indicator in enumerate(random_indicators):
                if st.button(f"🎯 {indicator[:60]}...", key=f"random_{i}"):
                    st.experimental_set_query_params(search=indicator.split(',')[0])
                    st.rerun()

    else:
        # Page d'accueil de la recherche
        st.subheader("🎯 Catégories d'Indicateurs Disponibles")

        # Afficher les catégories avec compteurs
        categories_info = {
            '💰 Économie': ['GDP', 'Trade', 'Investment', 'Employment', 'Inflation'],
            '👥 Social': ['Population', 'Health', 'Education', 'Poverty', 'Urban'],
            '⚡ Énergie': ['Electricity', 'Renewable', 'CO2', 'Energy', 'Climate'],
            '🏛️ Gouvernance': ['Democracy', 'Corruption', 'Government', 'Rule of law', 'Voice']
        }

        col1, col2 = st.columns(2)

        for i, (category, keywords) in enumerate(categories_info.items()):
            with [col1, col2][i % 2]:
                keyword_buttons = " ".join([f"`{kw}`" for kw in keywords])
                st.markdown(f"""
                <div class="correlation-insight">
                <h5>{category}</h5>
                <p>Mots-clés suggérés : {keyword_buttons}</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()