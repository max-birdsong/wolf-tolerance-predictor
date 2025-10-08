import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Page configuration
st.set_page_config(page_title="Wildlife Stakeholder Segmentation", layout="wide", page_icon="🐺")

# Title and introduction
st.title("🐺 Wildlife Stakeholder Segmentation Dashboard")
st.markdown("""
**Decision Support Tool for Wolf Management Strategies**  
This dashboard identifies distinct stakeholder segments based on wildlife value orientations and tolerance levels,
enabling targeted engagement and communication strategies.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔍 Cluster Explorer",
    "👥 Segment Profiles",
    "🤝 Experience & Trust",
    "📈 Model Performance",
    "💡 Strategic Insights"
])


# Load data function
@st.cache_data
def load_data():
        df = pd.read_csv('FINAL Quantitative Respondent Data Combined Long_v2.csv')  # Match your actual filename
        df['tolerance'] = (df['Q3'] > 4).astype(int)
        return df

    # Core clustering variables
    df = pd.DataFrame({
        'MUT_WVO': np.random.uniform(1, 7, n),
        'UT_WVO': np.random.uniform(1, 7, n),
        'Q3': np.random.uniform(1, 7, n),
        'group': np.random.choice(['Deer', 'GenPop', 'Land', 'Wolf'], n, p=[0.15, 0.45, 0.30, 0.10])
    })

    # Add tolerance based on mutualism
    df['tolerance'] = (df['MUT_WVO'] > 4.5).astype(int)

    # Demographics
    df['Q36'] = np.random.randint(18, 80, n)  # Age
    df['Q37'] = np.random.choice([1, 2], n, p=[0.48, 0.52])  # Sex (1=female, 2=male)
    df['Q34'] = np.random.choice([1, 2], n, p=[0.70, 0.30])  # Land ownership (1=no, 2=yes)
    df['Q35a'] = np.random.choice([1, 2], n, p=[0.85, 0.15])  # Livestock (1=no, 2=yes)

    # Wolf attitudes
    df['Q1a'] = np.random.uniform(1, 5, n)  # Wolves are beautiful
    df['Q1b'] = np.random.uniform(1, 5, n)  # Safety risk
    df['Q1c'] = np.random.uniform(1, 5, n)  # Ecosystem importance
    df['Q1d'] = np.random.uniform(1, 5, n)  # Economic impact
    df['Q1e'] = np.random.uniform(1, 5, n)  # Enjoy knowing they exist
    df['Q1f'] = np.random.uniform(1, 5, n)  # Wolves are burden

    # Experience variables
    df['Q17a1'] = np.random.choice([1, 2], n, p=[0.60, 0.40])  # Seen tracks
    df['Q17a2'] = np.random.choice([1, 2], n, p=[0.70, 0.30])  # Heard howls
    df['Q17a3'] = np.random.choice([1, 2], n, p=[0.80, 0.20])  # Watched wolves
    df['Q17a4'] = np.random.choice([1, 2], n, p=[0.90, 0.10])  # Seen close to home
    df['Q17a5'] = np.random.choice([1, 2], n, p=[0.95, 0.05])  # Property damage
    df['Q17a8'] = np.random.choice([1, 2], n, p=[0.85, 0.15])  # Enjoyable interaction

    # Agency trust
    df['Q18'] = np.random.uniform(1, 5, n)  # FWP interaction (wolf issues)
    df['Q19'] = np.random.uniform(1, 5, n)  # FWP interaction (other issues)
    df['Q24'] = np.random.uniform(1, 5, n)  # Satisfaction with wolf management
    df['Q25'] = np.random.uniform(1, 5, n)  # Confidence in FWP

    # Political efficacy
    df['Q22a'] = np.random.uniform(1, 5, n)  # Citizens can influence decisions
    df['Q22c'] = np.random.uniform(1, 5, n)  # Have opportunity to provide input
    df['Q22d'] = np.random.uniform(1, 5, n)  # Agencies listen to input
    df['Q22e'] = np.random.uniform(1, 5, n)  # Agencies respect way of life

    return df


@st.cache_data
def perform_clustering(df, n_clusters=3):
    features = ['MUT_WVO', 'UT_WVO', 'Q3']
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df[features].dropna().copy()
    df_clustered['cluster'] = clusters

    # Add all other variables back
    for col in df.columns:
        if col not in df_clustered.columns:
            df_clustered[col] = df.loc[df_clustered.index, col]

    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    calinski = calinski_harabasz_score(X_scaled, clusters)

    return df_clustered, kmeans, scaler, silhouette, davies_bouldin, calinski


# Load data
df = load_data()
df_clustered, model, scaler, sil_score, db_score, ch_score = perform_clustering(df, n_clusters=3)


# Helper functions
def get_cluster_name(cluster_id):
    names = {
        0: "Utilitarian Landowners",
        1: "Mutualist General Public",
        2: "Tolerant Moderates"
    }
    return names.get(cluster_id, f"Cluster {cluster_id}")


def format_percentage(series):
    """Calculate percentage of 'yes' responses (value=2)"""
    return (series == 2).sum() / len(series) * 100


# ==================== PAGE 1: OVERVIEW ====================
if page == "📊 Overview":
    st.header("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Respondents", f"{len(df_clustered):,}")
    with col2:
        st.metric("Clusters Identified", "3")
    with col3:
        tolerance_pct = (df_clustered['tolerance'].sum() / len(df_clustered)) * 100
        st.metric("Overall Tolerance Rate", f"{tolerance_pct:.1f}%")
    with col4:
        st.metric("Model Quality (Silhouette)", f"{sil_score:.3f}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = df_clustered['cluster'].value_counts().sort_index()
        cluster_pcts = (cluster_counts / cluster_counts.sum() * 100).round(1)

        fig = go.Figure(data=[go.Pie(
            labels=[get_cluster_name(i) for i in cluster_counts.index],
            values=cluster_counts.values,
            text=[f'{pct}%' for pct in cluster_pcts],
            textposition='inside',
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#95E1D3'])
        )])
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Quick Cluster Overview")

        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            size = len(cluster_data)
            tolerance_rate = (cluster_data['tolerance'].sum() / size) * 100

            with st.expander(f"**{get_cluster_name(cluster_id)}** (n={size})"):
                st.write(f"**Tolerance Rate:** {tolerance_rate:.1f}%")
                st.write(f"**Mutualism:** {cluster_data['MUT_WVO'].mean():.2f}")
                st.write(f"**Utilitarianism:** {cluster_data['UT_WVO'].mean():.2f}")
                st.write(f"**Wolf Acceptance:** {cluster_data['Q3'].mean():.2f}")
                st.write(f"**Avg Age:** {cluster_data['Q36'].mean():.0f} years")

    st.markdown("---")

    # Demographics comparison
    st.subheader("Demographic Snapshot by Cluster")

    demo_data = []
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        demo_data.append({
            'Cluster': get_cluster_name(cluster_id),
            'Avg Age': f"{cluster_data['Q36'].mean():.0f}",
            '% Male': f"{(cluster_data['Q37'] == 2).sum() / len(cluster_data) * 100:.0f}%",
            '% Own Land': f"{(cluster_data['Q34'] == 2).sum() / len(cluster_data) * 100:.0f}%",
            '% Raise Livestock': f"{(cluster_data['Q35a'] == 2).sum() / len(cluster_data) * 100:.0f}%"
        })

    st.dataframe(pd.DataFrame(demo_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Key Takeaways")
    st.markdown("""
    - **Three distinct stakeholder segments** identified with meaningful differences in values and tolerance
    - **Mutualism strongly predicts tolerance** - no high-mutualist, low-tolerance segment exists
    - **Landowners concentrate in low-tolerance cluster**, suggesting targeted engagement needed
    - **General public shows highest tolerance**, representing opportunity for advocacy support
    - **Demographics vary significantly** across clusters, enabling targeted outreach strategies
    """)

# ==================== PAGE 2: CLUSTER EXPLORER ====================
elif page == "🔍 Cluster Explorer":
    st.header("Interactive Cluster Exploration")

    st.subheader("Value Orientation & Acceptance Profiles")

    cluster_means = df_clustered.groupby('cluster')[['MUT_WVO', 'UT_WVO', 'Q3']].mean()

    categories = ['Mutualism', 'Utilitarianism', 'Wolf Acceptance']

    fig = go.Figure()

    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    for i, cluster_id in enumerate(sorted(cluster_means.index)):
        values = [
            cluster_means.loc[cluster_id, 'MUT_WVO'],
            cluster_means.loc[cluster_id, 'UT_WVO'],
            cluster_means.loc[cluster_id, 'Q3']
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=get_cluster_name(cluster_id),
            line=dict(color=colors[i])
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 7])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Wolf attitude comparison
    st.subheader("Wolf Attitudes Across Clusters")

    attitude_vars = {
        'Q1a': 'Wolves are beautiful',
        'Q1c': 'Important for ecosystem',
        'Q1e': 'Enjoy knowing they exist',
        'Q1b': 'Pose safety risk (reversed)',
        'Q1f': 'Wolves are burden (reversed)'
    }

    attitude_data = []
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        for var, label in attitude_vars.items():
            value = cluster_data[var].mean()
            # Reverse scoring for negative items
            if 'reversed' in label:
                value = 6 - value
                label = label.replace(' (reversed)', '')
            attitude_data.append({
                'Cluster': get_cluster_name(cluster_id),
                'Attitude': label,
                'Mean Score': value
            })

    attitude_df = pd.DataFrame(attitude_data)

    fig = px.bar(attitude_df, x='Attitude', y='Mean Score', color='Cluster',
                 barmode='group', height=400,
                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Stakeholder group distribution
    st.subheader("Stakeholder Group Distribution by Cluster")

    group_dist = pd.crosstab(df_clustered['cluster'], df_clustered['group'], normalize='index') * 100
    group_dist = group_dist.round(1)

    fig = go.Figure()

    group_names = {'Deer': 'Deer Hunters', 'GenPop': 'General Public', 'Land': 'Landowners', 'Wolf': 'Wolf Hunters'}
    colors_groups = {'Deer': '#8B4513', 'GenPop': '#4169E1', 'Land': '#228B22', 'Wolf': '#696969'}

    for group in group_dist.columns:
        fig.add_trace(go.Bar(
            name=group_names.get(group, group),
            x=[get_cluster_name(i) for i in group_dist.index],
            y=group_dist[group],
            marker_color=colors_groups.get(group, '#999999'),
            text=[f'{val:.1f}%' for val in group_dist[group]],
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        yaxis_title='Percentage',
        xaxis_title='Cluster',
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 3: SEGMENT PROFILES ====================
elif page == "👥 Segment Profiles":
    st.header("Detailed Segment Profiles")

    cluster_id = st.selectbox("Select Cluster to Explore",
                              sorted(df_clustered['cluster'].unique()),
                              format_func=get_cluster_name)

    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

    cluster_descriptions = {
        0: "This segment exhibits strong utilitarian values with low mutualism. Predominantly composed of landowners and deer hunters who view wildlife through a resource management lens. Low tolerance for wolves reflects concerns about impacts on property and game species.",
        1: "Characterized by high mutualism and emotional connection to wildlife. This segment, largely general public members, views wolves through a relational rather than utilitarian framework. Strong support for wolf presence and conservation.",
        2: "A moderate segment balancing utilitarian values with surprising tolerance. Despite practical wildlife perspectives, this diverse group demonstrates acceptance of wolf presence, suggesting tolerance extends beyond value orientations alone."
    }

    st.subheader(f"{get_cluster_name(cluster_id)}")
    st.info(cluster_descriptions.get(cluster_id, ""))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Segment Size", f"{len(cluster_data):,}",
                  f"{len(cluster_data) / len(df_clustered) * 100:.1f}% of sample")
    with col2:
        tolerance_rate = (cluster_data['tolerance'].sum() / len(cluster_data)) * 100
        st.metric("Tolerance Rate", f"{tolerance_rate:.1f}%")
    with col3:
        primary_group = cluster_data['group'].value_counts().index[0]
        group_names = {'Deer': 'Deer Hunters', 'GenPop': 'General Public', 'Land': 'Landowners', 'Wolf': 'Wolf Hunters'}
        st.metric("Primary Stakeholder", group_names.get(primary_group, primary_group))

    st.markdown("---")

    # Demographics
    st.subheader("Demographics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Age", f"{cluster_data['Q36'].mean():.0f} years")
    with col2:
        male_pct = (cluster_data['Q37'] == 2).sum() / len(cluster_data) * 100
        st.metric("% Male", f"{male_pct:.0f}%")
    with col3:
        land_pct = (cluster_data['Q34'] == 2).sum() / len(cluster_data) * 100
        st.metric("% Own 160+ Acres", f"{land_pct:.0f}%")
    with col4:
        livestock_pct = (cluster_data['Q35a'] == 2).sum() / len(cluster_data) * 100
        st.metric("% Raise Livestock", f"{livestock_pct:.0f}%")

    st.markdown("---")

    # Value orientations and attitudes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Value Orientations")

        values_df = pd.DataFrame({
            'Dimension': ['Mutualism', 'Utilitarianism', 'Wolf Acceptance'],
            'Score': [
                cluster_data['MUT_WVO'].mean(),
                cluster_data['UT_WVO'].mean(),
                cluster_data['Q3'].mean()
            ]
        })

        fig = px.bar(values_df, x='Dimension', y='Score',
                     color='Score',
                     color_continuous_scale='Viridis',
                     range_y=[0, 7])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Stakeholder Composition")

        group_counts = cluster_data['group'].value_counts()

        fig = px.pie(
            values=group_counts.values,
            names=[group_names.get(g, g) for g in group_counts.index],
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Wolf attitudes breakdown
    st.subheader("Wolf Attitudes Profile")

    attitudes = {
        'Q1a': ('Wolves are beautiful', False),
        'Q1c': ('Ecosystem importance', False),
        'Q1e': ('Enjoy knowing they exist', False),
        'Q1b': ('Pose safety risk', True),
        'Q1d': ('Economic harm', True),
        'Q1f': ('Wolves are burden', True)
    }

    attitude_scores = []
    for var, (label, reverse) in attitudes.items():
        score = cluster_data[var].mean()
        if reverse:
            score = 6 - score  # Reverse negative items
        attitude_scores.append({'Attitude': label, 'Score': score})

    attitude_df = pd.DataFrame(attitude_scores).sort_values('Score', ascending=True)

    fig = px.bar(attitude_df, y='Attitude', x='Score', orientation='h',
                 color='Score', color_continuous_scale='RdYlGn',
                 range_x=[0, 5])
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 4: EXPERIENCE & TRUST ====================
elif page == "🤝 Experience & Trust":
    st.header("Wolf Experience & Agency Trust Analysis")

    st.markdown("""
    Understanding how direct wolf experiences and trust in management agencies vary across segments 
    provides insights into *why* tolerance differs and how to build credibility.
    """)

    st.markdown("---")

    # Wolf Experience by Cluster
    st.subheader("Direct Wolf Experience by Cluster")

    experience_vars = {
        'Q17a1': 'Seen wolf tracks',
        'Q17a2': 'Heard wolf howls',
        'Q17a3': 'Watched wolves from afar',
        'Q17a4': 'Seen wolves close to home',
        'Q17a5': 'Wolves damaged property',
        'Q17a8': 'Enjoyable wolf interaction'
    }

    exp_data = []
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        for var, label in experience_vars.items():
            pct = format_percentage(cluster_data[var])
            exp_data.append({
                'Cluster': get_cluster_name(cluster_id),
                'Experience Type': label,
                'Percentage': pct
            })

    exp_df = pd.DataFrame(exp_data)

    fig = px.bar(exp_df, x='Experience Type', y='Percentage', color='Cluster',
                 barmode='group', height=450,
                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    fig.update_layout(xaxis_tickangle=-45, yaxis_title='% Experienced')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Trust and Satisfaction Metrics
    st.subheader("Agency Trust & Satisfaction by Cluster")

    col1, col2 = st.columns(2)

    with col1:
        trust_vars = {
            'Q24': 'Satisfaction with wolf mgmt',
            'Q25': 'Confidence in FWP',
            'Q18': 'FWP interaction (wolf issues)',
            'Q19': 'FWP interaction (other issues)'
        }

        trust_data = []
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            for var, label in trust_vars.items():
                trust_data.append({
                    'Cluster': get_cluster_name(cluster_id),
                    'Metric': label,
                    'Score': cluster_data[var].mean()
                })

        trust_df = pd.DataFrame(trust_data)

        fig = px.bar(trust_df, x='Metric', y='Score', color='Cluster',
                     barmode='group', height=350,
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3'],
                     range_y=[0, 5])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Political efficacy
        efficacy_vars = {
            'Q22a': 'Citizens can influence decisions',
            'Q22c': 'Have opportunity for input',
            'Q22d': 'Agencies listen to input',
            'Q22e': 'Agencies respect way of life'
        }

        efficacy_data = []
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            for var, label in efficacy_vars.items():
                efficacy_data.append({
                    'Cluster': get_cluster_name(cluster_id),
                    'Metric': label,
                    'Score': cluster_data[var].mean()
                })

        efficacy_df = pd.DataFrame(efficacy_data)

        fig = px.bar(efficacy_df, x='Metric', y='Score', color='Cluster',
                     barmode='group', height=350,
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3'],
                     range_y=[0, 5])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Key Insights
    st.subheader("Experience & Trust Insights by Cluster")

    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

        with st.expander(f"**{get_cluster_name(cluster_id)}**"):

            # Calculate key metrics
            any_experience = format_percentage(cluster_data['Q17a1'])
            property_damage = format_percentage(cluster_data['Q17a5'])
            enjoyable = format_percentage(cluster_data['Q17a8'])
            satisfaction = cluster_data['Q24'].mean()
            confidence = cluster_data['Q25'].mean()
            influence = cluster_data['Q22a'].mean()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Wolf Experience Rate", f"{any_experience:.0f}%")
                st.metric("Property Damage", f"{property_damage:.0f}%")

            with col2:
                st.metric("Management Satisfaction", f"{satisfaction:.2f}/5")
                st.metric("Confidence in FWP", f"{confidence:.2f}/5")

            with col3:
                st.metric("Enjoyable Interactions", f"{enjoyable:.0f}%")
                st.metric("Perceived Influence", f"{influence:.2f}/5")

            # Interpretation
            if cluster_id == 0:
                st.markdown("""
                **Key Patterns:**
                - Higher rates of negative experiences (property concerns)
                - Lower satisfaction with wolf management
                - Feel less heard by agencies
                - Trust deficit is barrier to tolerance
                """)
            elif cluster_id == 1:
                st.markdown("""
                **Key Patterns:**
                - More positive/observational wolf experiences
                - Higher satisfaction with management (when it protects wolves)
                - Feel moderately empowered in process
                - Experience reinforces pro-wolf attitudes
                """)
            else:
                st.markdown("""
                **Key Patterns:**
                - Moderate experience levels
                - Pragmatic trust in agency competence
                - Balanced view of influence opportunities
                - Trust may enable tolerance despite utilitarian values
                """)

    st.markdown("---")

    st.subheader("Strategic Implications")
    st.markdown("""
    **Experience Effects:**
    - Negative experiences (property damage) concentrate in low-tolerance cluster
    - Positive/observational experiences more common among tolerant segments
    - Experience type matters more than experience frequency

    **Trust Dynamics:**
    - Trust in FWP correlates with tolerance but relationship varies by cluster
    - Low-tolerance cluster shows trust deficit - engagement barrier
    - High perceived influence may buffer negative experiences

    **Recommendations:**
    - Address trust deficit with Cluster 0 through responsive conflict management
    - Leverage positive experiences in Cluster 1 for advocacy
    - Maintain credibility with Cluster 2 through science-based, adaptive management
    """)

# ==================== PAGE 5: MODEL PERFORMANCE ====================
elif page == "📈 Model Performance":
    st.header("Model Validation & Performance")

    st.markdown("""
    The clustering model was validated using multiple metrics to ensure robust segment identification.
    The 3-cluster solution was selected based on optimal balance of statistical performance and interpretability.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{sil_score:.3f}",
                  help="Measures cluster cohesion. Range: -1 to 1. Higher is better. >0.3 indicates reasonable structure.")
    with col2:
        st.metric("Davies-Bouldin Index", f"{db_score:.3f}",
                  help="Measures cluster separation. Lower is better. <1.5 indicates good separation.")
    with col3:
        st.metric("Calinski-Harabasz", f"{ch_score:.1f}",
                  help="Ratio of between-cluster to within-cluster variance. Higher is better.")

    st.markdown("---")

    st.subheader("Cluster Selection Analysis")

    st.markdown("Performance metrics across different numbers of clusters:")

    k_values = range(2, 7)
    metrics_data = {'k': [], 'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}

    for k in k_values:
        _, _, _, sil, db, ch = perform_clustering(df, n_clusters=k)
        metrics_data['k'].append(k)
        metrics_data['Silhouette'].append(sil)
        metrics_data['Davies-Bouldin'].append(db)
        metrics_data['Calinski-Harabasz'].append(ch)

    metrics_df = pd.DataFrame(metrics_data)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df['k'], y=metrics_df['Silhouette'],
                                 mode='lines+markers', name='Silhouette',
                                 line=dict(color='#4ECDC4', width=3),
                                 marker=dict(size=10)))
        fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Selected k=3")
        fig.update_layout(title="Silhouette Score by k",
                          xaxis_title="Number of Clusters",
                          yaxis_title="Silhouette Score",
                          height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df['k'], y=metrics_df['Davies-Bouldin'],
                                 mode='lines+markers', name='Davies-Bouldin',
                                 line=dict(color='#FF6B6B', width=3),
                                 marker=dict(size=10)))
        fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Selected k=3")
        fig.update_layout(title="Davies-Bouldin Index by k (lower is better)",
                          xaxis_title="Number of Clusters",
                          yaxis_title="Davies-Bouldin Index",
                          height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metrics_df.round(3), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Why k=3?")
    st.markdown("""
    The 3-cluster solution was selected because:
    - **Highest silhouette score (0.314)** indicates best-defined clusters
    - **Reasonable Davies-Bouldin score** shows adequate separation
    - **Interpretable segments** with clear stakeholder distinctions
    - **Actionable insights** - three segments are manageable for strategy development
    - Beyond k=3, marginal metric improvements don't justify added complexity
    """)

# ==================== PAGE 6: STRATEGIC INSIGHTS ====================
elif page == "💡 Strategic Insights":
    st.header("Strategic Recommendations for Wildlife Managers")

    st.markdown("""
    These insights translate cluster analysis into actionable engagement strategies for each stakeholder segment.
    """)

    # Cluster 0: Utilitarian Landowners
    with st.expander("🎯 **Cluster 0: Utilitarian Landowners** - High Priority, High Resistance", expanded=True):
        st.markdown("""
        **Profile:** Low mutualism (3.08), high utilitarianism (5.96), 90.6% intolerant. Dominated by landowners (43.1%) and deer hunters (18.0%).

        **Key Challenge:** Lowest acceptance of wolves due to concerns about property impacts and game populations.

        **Recommended Strategies:**

        1. **Compensation & Mitigation Programs**
           - Emphasize livestock depredation compensation
           - Provide technical assistance for non-lethal deterrents
           - Share data on actual vs. perceived impacts to deer populations

        2. **Practical Communication Frame**
           - Focus on coexistence tools, not conservation values
           - Highlight agency responsiveness to conflict situations
           - Provide clear protocols for reporting problems

        3. **Stakeholder Engagement**
           - Direct outreach through agricultural extension offices
           - Partner with Farm Bureau and hunting organizations
           - Offer workshops on wolf behavior and deterrence methods

        4. **Build Trust Through Action**
           - This cluster shows lowest trust in FWP
           - Demonstrate responsiveness to conflict calls
           - Show respect for their economic concerns and way of life
           - Provide timely follow-up on depredation reports

        5. **Risk Management**
           - This segment's resistance could drive policy opposition
           - Proactive engagement critical to prevent organized resistance
           - Consider pilot programs demonstrating effective coexistence

        **Success Metrics:** Increased use of deterrents, reduced depredation complaints, improved perception of agency support, higher trust scores
        """)

    # Cluster 1: Mutualist General Public
    with st.expander("🎯 **Cluster 1: Mutualist General Public** - High Support, Advocacy Potential", expanded=True):
        st.markdown("""
        **Profile:** High mutualism (5.53), low utilitarianism (3.64), 94.1% tolerant. Primarily general public (65.5%).

        **Key Opportunity:** Strong support base for wolf conservation initiatives with positive wolf experiences.

        **Recommended Strategies:**

        1. **Leverage Support for Policy**
           - Mobilize for public comment periods
           - Develop citizen science monitoring programs
           - Create volunteer ambassador network

        2. **Educational Programming**
           - Wildlife viewing opportunities and ecotourism
           - Social media campaigns highlighting wolf ecology
           - School programs emphasizing ecosystem benefits
           - Capitalize on their positive wolf experiences

        3. **Counter-Balance Opposition**
           - Provide talking points for public advocacy
           - Organize letter-writing campaigns during policy debates
           - Build coalition with conservation organizations

        4. **Maintain Engagement**
           - This group shows moderate political efficacy
           - Strengthen belief that their voice matters
           - Provide clear pathways for meaningful participation
           - Share how their input influenced decisions

        5. **Economic Development**
           - Promote wolf-watching tourism revenue
           - Partner with local businesses on wildlife tourism
           - Document economic benefits for rural communities

        **Success Metrics:** Volunteer participation rates, public comment submissions, tourism revenue growth, sustained high satisfaction with FWP
        """)

    # Cluster 2: Tolerant Moderates
    with st.expander("🎯 **Cluster 2: Tolerant Moderates** - Strategic Bridge Group", expanded=True):
        st.markdown("""
        **Profile:** Moderate mutualism (3.84), high utilitarianism (5.33), 97.5% tolerant. Diverse stakeholder mix.

        **Key Insight:** This segment defies expectations - high tolerance despite utilitarian values. Their moderate trust and balanced experiences make them credible messengers.

        **Recommended Strategies:**

        1. **Identify Success Factors**
           - Research why tolerance persists despite utilitarian views
           - Document what makes coexistence work for this group
           - Use findings to inform outreach to Cluster 0
           - Their moderate agency trust suggests realistic expectations work

        2. **Peer-to-Peer Messengers**
           - Recruit landowners from this cluster as spokespeople
           - Share success stories within agricultural community
           - More credible than agency messaging to resistant groups
           - Their practical perspective resonates across stakeholder types

        3. **Balanced Communication**
           - Acknowledge both ecological and practical perspectives
           - Emphasize adaptive management approach
           - Show responsiveness to legitimate concerns
           - Model the pragmatic tolerance this group demonstrates

        4. **Coalition Building**
           - This diverse group can bridge conservation/agriculture divide
           - Support dialogue between different stakeholder groups
           - Model productive engagement across value differences
           - Leverage their balanced wolf experiences (both positive and practical)

        5. **Maintain Their Tolerance**
           - Don't take this group for granted
           - Continue demonstrating management competence
           - Keep trust levels stable through transparent communication
           - Address conflicts quickly to prevent erosion of tolerance

        **Success Metrics:** Peer testimonials developed, cross-stakeholder dialogue events, replication of tolerance factors, sustained high tolerance rates
        """)

    st.markdown("---")

    # Overall strategic framework
    st.subheader("Integrated Strategy Framework")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Priority Actions:**
        1. Focus resources on Cluster 0 mitigation/compensation
        2. Mobilize Cluster 1 for policy support
        3. Study Cluster 2 to understand tolerance mechanisms
        4. Develop cluster-specific communication materials
        5. Train field staff on segment-appropriate engagement
        6. Monitor trust metrics as leading indicators
        """)

    with col2:
        st.markdown("""
        **Key Success Factors:**
        - Tailor messaging to value orientations
        - Avoid one-size-fits-all communication
        - Build trust through responsive action
        - Balance biological and social objectives
        - Monitor segment-specific outcomes
        - Maintain flexibility as attitudes evolve
        """)

    st.markdown("---")

    # Resource allocation
    st.subheader("Suggested Resource Allocation")

    resource_data = pd.DataFrame({
        'Cluster': ['Cluster 0: Utilitarian Landowners',
                    'Cluster 1: Mutualist General Public',
                    'Cluster 2: Tolerant Moderates'],
        'Resource Priority': ['High (40%)', 'Medium (30%)', 'Medium (30%)'],
        'Focus Area': ['Conflict mitigation, trust-building',
                       'Advocacy mobilization, education',
                       'Research, peer messaging'],
        'Engagement Type': ['Direct, problem-solving',
                            'Inspirational, participatory',
                            'Collaborative, bridge-building'],
        'Trust Challenge': ['Critical - major deficit',
                            'Low - already satisfied',
                            'Moderate - maintain balance']
    })

    st.dataframe(resource_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Implementation roadmap
    st.subheader("Implementation Roadmap")

    st.markdown("""
    **Phase 1: Immediate (0-3 months)**
    - Audit current conflict response protocols
    - Develop cluster-specific fact sheets and talking points
    - Begin trust-building outreach with Cluster 0 stakeholders
    - Identify Cluster 2 peer messengers

    **Phase 2: Short-term (3-6 months)**
    - Launch compensation/deterrent program enhancements
    - Train field staff on segment-appropriate engagement
    - Pilot peer-to-peer messaging program
    - Activate Cluster 1 volunteer network

    **Phase 3: Medium-term (6-12 months)**
    - Evaluate trust metrics and adjust strategies
    - Scale successful pilot programs
    - Document best practices from Cluster 2
    - Conduct cross-cluster dialogue sessions

    **Phase 4: Long-term (12+ months)**
    - Reassess cluster characteristics (attitudes may shift)
    - Refine strategies based on outcome data
    - Institutionalize segment-based approach
    - Share lessons learned with other wildlife agencies
    """)

    st.markdown("---")

    st.subheader("Monitoring & Evaluation")

    st.markdown("""
    **Key Performance Indicators by Cluster:**

    **Cluster 0 (Utilitarian Landowners):**
    - % using non-lethal deterrents
    - Depredation complaint rates
    - Trust in FWP scores
    - Satisfaction with conflict response time

    **Cluster 1 (Mutualist General Public):**
    - Public comment participation
    - Volunteer program enrollment
    - Social media engagement metrics
    - Tourism revenue linked to wolves

    **Cluster 2 (Tolerant Moderates):**
    - Number of peer testimonials collected
    - Cross-stakeholder dialogue attendance
    - Tolerance rate stability
    - Trust score maintenance

    **Overall Program Success:**
    - Movement between clusters over time
    - Statewide tolerance trends
    - Conflict incident rates
    - Public satisfaction with wolf management
    """)

# Footer
st.markdown("---")
st.markdown("""
*Dashboard developed for wildlife management decision support | Data reflects stakeholder survey responses on wildlife value orientations, wolf tolerance, experiences, and agency trust*

**Technical Note:** Replace synthetic data with actual dataset by updating the `load_data()` function with `pd.read_csv('your_data.csv')`
""")