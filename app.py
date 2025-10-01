import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Wolf Tolerance Prediction System",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1a472a;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #5a6c57;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .banner {
        background: linear-gradient(135deg, #1a472a 0%, #2d5f3f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .banner-title {
        color: white;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    .banner-subtitle {
        color: #e8f5e9;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1a472a;
    }
    .stButton>button {
        background-color: #1a472a;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2d5f3f;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #e8f5e9;
        border-left: 5px solid #1a472a;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="banner">
    <p class="banner-title">🐺 Wolf Tolerance Prediction System</p>
    <p class="banner-subtitle">Machine Learning Decision Support for Wildlife Management</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="text-align: center; color: #5a6c57; font-size: 1.1rem; margin-bottom: 2rem;">'
    'A portfolio demonstration of predictive modeling for human-wildlife coexistence'
    '</p>',
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('wolf_tolerance_model.pkl')

model = load_model()

# Sidebar navigation with personal branding
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "Select Tool:",
    ["🎯 Individual Assessment", "📊 Batch Processing", "📈 Model Documentation"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# About the project
with st.sidebar.expander("💼 About This Project", expanded=False):
    st.markdown("""
    **Project Type:** Data Science Portfolio Demonstration
    
    **Purpose:** Showcase production-level ML system design for wildlife management agencies
    
    **Key Demonstrations:**
    - End-to-end ML pipeline (data → model → deployment)
    - Agency-appropriate UI/UX design
    - Interpretable predictions with actionable insights
    - Professional documentation standards
    - Operational decision support workflows
    
    **Use Case:** This type of tool could support wildlife agencies in:
    - Resource allocation for conflict mitigation
    - Targeted community outreach
    - Evidence-based policy planning
    
    **Disclaimer:** This is an independent portfolio project using publicly available research data. Not affiliated with any government agency.
    """)

# Educational content in sidebar with professional framing
with st.sidebar.expander("📚 About Wildlife Value Orientations"):
    st.markdown("""
    **Mutualism Values (1-7)**
    
    Viewing wildlife as companions deserving of care and protection. Individuals with high mutualism values tend to:
    - Support wildlife conservation
    - Emphasize animal welfare
    - View wildlife as having intrinsic rights
    
    *Example: "Wolves have a right to exist in Montana"*
    
    ---
    
    **Utilitarian Values (1-7)**
    
    Viewing wildlife primarily as resources for human benefit. Individuals with high utilitarian values tend to:
    - Prioritize human needs over wildlife
    - Support management focused on economic benefits
    - View wildlife through a cost-benefit lens
    
    *Example: "Wildlife management should prioritize human safety and economic interests"*
    
    ---
    
    **Research Foundation**
    
    Peer-reviewed research demonstrates that wildlife value orientations are **2-16× stronger predictors** of tolerance than demographic factors alone, making them essential for targeted outreach and conflict mitigation strategies.
    """)

with st.sidebar.expander("ℹ️ Technical Details"):
    st.markdown("""
    **Model Specifications**
    
    - **Algorithm**: Logistic Regression with ADASYN
    - **Accuracy**: 70% on test data
    - **Training Data**: 2,146 Montana residents (2023)
    - **Key Predictors**: Value orientations, stakeholder group
    - **Framework**: Scikit-learn + Streamlit
    
    **Model Philosophy**
    
    This tool provides probabilistic assessments to inform management strategies. It demonstrates how ML can supplement (not replace) direct community engagement and field expertise.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("**Portfolio Project**")
st.sidebar.caption("Data Science | Conservation Technology")
st.sidebar.caption("Built with Python, Scikit-learn, Streamlit")

# Page 1: Individual Assessment
if page == "🎯 Individual Assessment":
    st.markdown("### Individual Tolerance Assessment")
    st.markdown("Evaluate wolf tolerance for a specific individual or community member")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Demographic Information")
        age = st.slider("Age", 18, 90, 45, help="Respondent age in years")
        gender = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x: "Female" if x == 1 else "Male"
        )
        group = st.selectbox(
            "Stakeholder Group",
            ["GenPop", "Land", "Wolf", "Deer"],
            help="**GenPop**: General public | **Land**: Landowner/Rancher | **Wolf**: Wolf hunter | **Deer**: Deer hunter"
        )
        
        st.markdown("---")
        st.markdown("#### Wildlife Value Orientations")
        st.caption("Based on validated survey scales (Teel & Manfredo, 2010)")
        
        mutualism = st.slider(
            "Mutualism Score",
            1.0, 7.0, 4.0, 0.1,
            help="How strongly does this person view wildlife as companions? (1=Low, 7=High)"
        )
        utilitarianism = st.slider(
            "Utilitarianism Score",
            1.0, 7.0, 4.0, 0.1,
            help="How strongly does this person view wildlife as resources? (1=Low, 7=High)"
        )
        
        st.markdown("---")
        predict_button = st.button("🔍 Run Assessment", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            input_data = pd.DataFrame({
                'Q36': [age],
                'Q37': [gender],
                'group': [group],
                'MUT_WVO': [mutualism],
                'UT_WVO': [utilitarianism]
            })
            
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            confidence = max(proba)
            
            # Professional results display
            st.markdown("### Assessment Results")
            
            # Primary result with color-coded styling
            if prediction == "Tolerant":
                st.success("#### ✅ TOLERANT CLASSIFICATION")
                st.markdown(f"**Probability of Tolerance:** {proba[1]:.1%}")
            else:
                st.error("#### ⚠️ NOT TOLERANT CLASSIFICATION")
                st.markdown(f"**Probability of Intolerance:** {proba[0]:.1%}")
            
            # Confidence metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Classification", prediction)
            with metric_col2:
                st.metric("Model Confidence", f"{confidence:.1%}")
            with metric_col3:
                certainty_level = "High" if confidence > 0.75 else "Moderate" if confidence > 0.65 else "Low"
                st.metric("Certainty Level", certainty_level)
            
            st.markdown("---")
            
            # Probability visualization
            st.markdown("#### Classification Probabilities")
            prob_df = pd.DataFrame({
                'Classification': ['Not Tolerant', 'Tolerant'],
                'Probability': [proba[0], proba[1]]
            })
            
            fig, ax = plt.subplots(figsize=(10, 3))
            colors = ['#c62828', '#2e7d32']
            bars = ax.barh(prob_df['Classification'], prob_df['Probability'], color=colors, height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
                ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
            
            # Management recommendations
            st.markdown("---")
            st.markdown("#### 📋 Example Management Recommendations")
            st.caption("*Demonstrating how agencies could operationalize predictions*")
            
            if prediction == "Tolerant" and confidence > 0.75:
                st.markdown("""
                <div class="info-box">
                <strong>Low Risk Profile</strong><br>
                Example agency actions for individuals with this profile:
                <ul>
                <li>Maintain standard engagement levels</li>
                <li>Potential ambassador for coexistence programs</li>
                <li>Low priority for intensive outreach resources</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Not Tolerant" and confidence > 0.75:
                st.markdown("""
                <div class="info-box">
                <strong>High Risk Profile</strong><br>
                Example agency actions for individuals with this profile:
                <ul>
                <li><strong>Priority for proactive engagement</strong></li>
                <li>Deploy conflict mitigation resources (range riders, compensation programs)</li>
                <li>Focus on utilitarian value framing in communications</li>
                <li>Increase monitoring for potential human-wildlife conflicts</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                <strong>Moderate/Uncertain Profile</strong><br>
                Example agency actions for borderline classifications:
                <ul>
                <li>Conduct direct surveys for accurate assessment</li>
                <li>Monitor situation closely</li>
                <li>Apply standard engagement protocols</li>
                <li>Re-assess after community interactions</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed factor analysis
            with st.expander("🔬 Factor Analysis: What Drives This Assessment?"):
                st.markdown("**Influential Factors** (based on logistic regression coefficients):")
                
                factors = []
                if utilitarianism > 5:
                    factors.append("- 🔴 **High utilitarian values** (coef: -0.84): Strong predictor of intolerance")
                elif utilitarianism < 3:
                    factors.append("- 🟢 **Low utilitarian values** (coef: -0.84): Increases tolerance likelihood")
                else:
                    factors.append("- 🟡 **Moderate utilitarian values**: Neutral effect")
                
                if mutualism > 5:
                    factors.append("- 🟢 **High mutualist values** (coef: +0.32): Increases tolerance")
                elif mutualism < 3:
                    factors.append("- 🔴 **Low mutualist values** (coef: +0.32): Decreases tolerance")
                else:
                    factors.append("- 🟡 **Moderate mutualist values**: Neutral effect")
                
                if group == "Land":
                    factors.append("- 🔴 **Landowner status** (coef: -0.35): Decreases tolerance significantly")
                elif group == "GenPop":
                    factors.append("- 🟢 **General public** (coef: +0.32): Increases tolerance baseline")
                elif group == "Wolf":
                    factors.append("- 🔴 **Wolf hunter status**: Likely decreases tolerance")
                else:
                    factors.append("- 🟡 **Deer hunter status**: Moderate effect")
                
                factors.append(f"- 🟡 **Demographics** (age: {age}, gender: {'Female' if gender==1 else 'Male'}): Minimal impact (age coef: -0.05, gender coef: +0.09)")
                
                for factor in factors:
                    st.markdown(factor)
                
                st.markdown("---")
                st.caption("**Note:** Value orientations are 2-16× more influential than demographic factors in predicting tolerance.")
        
        else:
            # Placeholder when no prediction yet
            st.info("👈 Enter individual characteristics and click **Run Assessment** to generate predictions and recommendations.")
            st.markdown("---")
            st.markdown("#### Assessment Output Includes:")
            st.markdown("""
            - **Tolerance classification** with probability scores
            - **Confidence metrics** for decision reliability
            - **Management recommendations** tailored to risk level
            - **Factor analysis** explaining the prediction
            - **Actionable next steps** for FWP staff
            """)

# Page 2: Batch Processing
elif page == "📊 Batch Processing":
    st.markdown("### Batch Assessment Tool")
    st.markdown("Process multiple individuals simultaneously for community-level or regional analysis")
    
    # Enhanced instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📁 Upload Data File")
        st.markdown("Upload a CSV file containing assessment data for multiple individuals.")
    
    with col2:
        st.markdown("#### 📄 Template")
        st.markdown("Download our formatted template to ensure compatibility.")
    
    # Sample CSV template with better presentation
    with st.expander("📋 CSV Format Requirements & Template", expanded=True):
        st.markdown("""
        **Required Columns:**
        
        | Column | Description | Valid Values |
        |--------|-------------|--------------|
        | `Q36` | Age | 18-90 |
        | `Q37` | Gender | 1 (Female), 2 (Male) |
        | `group` | Stakeholder Group | GenPop, Land, Wolf, Deer |
        | `MUT_WVO` | Mutualism Score | 1.0-7.0 |
        | `UT_WVO` | Utilitarianism Score | 1.0-7.0 |
        
        **Optional Columns:**
        - Any identifier columns (e.g., `ID`, `Name`, `Location`) will be preserved in output
        """)
        
        st.markdown("---")
        st.markdown("**Sample Data Template:**")
        
        # Enhanced sample data with identifiers
        sample_df = pd.DataFrame({
            'ID': ['R001', 'R002', 'R003', 'R004'],
            'Location': ['Missoula', 'Bozeman', 'Great Falls', 'Kalispell'],
            'Q36': [45, 32, 67, 51],
            'Q37': [1, 2, 1, 2],
            'group': ['GenPop', 'Land', 'Wolf', 'Deer'],
            'MUT_WVO': [5.2, 3.1, 6.8, 4.5],
            'UT_WVO': [3.5, 6.2, 2.1, 5.0]
        })
        st.dataframe(sample_df, use_container_width=True)
        
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            "📥 Download Template CSV",
            csv_sample,
            "FWP_Wolf_Assessment_Template.csv",
            "text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type="csv",
        help="Upload a CSV file with the required columns listed above"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_cols = ['Q36', 'Q37', 'group', 'MUT_WVO', 'UT_WVO']
            
            if all(col in df.columns for col in required_cols):
                
                with st.spinner("🔄 Processing assessments... This may take a moment for large datasets."):
                    predictions = model.predict(df[required_cols])
                    probas = model.predict_proba(df[required_cols])
                    
                    df['Predicted_Tolerance'] = predictions
                    df['Tolerant_Probability'] = probas[:, 1]
                    df['Confidence'] = probas.max(axis=1)
                    df['Risk_Level'] = df.apply(
                        lambda x: 'High Risk' if x['Predicted_Tolerance'] == 'Not Tolerant' and x['Confidence'] > 0.75
                        else 'Low Risk' if x['Predicted_Tolerance'] == 'Tolerant' and x['Confidence'] > 0.75
                        else 'Moderate Risk',
                        axis=1
                    )
                
                st.success(f"✅ Successfully processed {len(df)} assessments!")
                
                st.markdown("---")
                st.markdown("### 📊 Assessment Summary")
                
                # Enhanced summary statistics
                col1, col2, col3, col4 = st.columns(4)
                tolerant_pct = (predictions == 'Tolerant').mean()
                avg_confidence = probas.max(axis=1).mean()
                high_risk_count = (df['Risk_Level'] == 'High Risk').sum()
                
                col1.metric("Total Assessed", f"{len(df):,}")
                col2.metric("Predicted Tolerant", f"{tolerant_pct:.1%}", 
                           delta=f"{int(tolerant_pct * len(df))} individuals")
                col3.metric("High Risk Cases", f"{high_risk_count}", 
                           delta="Requires attention" if high_risk_count > 0 else "None identified")
                col4.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Risk breakdown
                st.markdown("---")
                st.markdown("### 🎯 Risk Distribution")
                risk_counts = df['Risk_Level'].value_counts()
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                with risk_col1:
                    high_risk = risk_counts.get('High Risk', 0)
                    st.markdown(f"""
                    <div style='background-color: #ffebee; padding: 1rem; border-radius: 8px; border-left: 4px solid #c62828;'>
                    <h4 style='color: #c62828; margin: 0;'>🔴 High Risk</h4>
                    <h2 style='margin: 0.5rem 0;'>{high_risk}</h2>
                    <p style='margin: 0; color: #666;'>Priority for outreach</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_col2:
                    mod_risk = risk_counts.get('Moderate Risk', 0)
                    st.markdown(f"""
                    <div style='background-color: #fff8e1; padding: 1rem; border-radius: 8px; border-left: 4px solid #f57c00;'>
                    <h4 style='color: #f57c00; margin: 0;'>🟡 Moderate Risk</h4>
                    <h2 style='margin: 0.5rem 0;'>{mod_risk}</h2>
                    <p style='margin: 0; color: #666;'>Monitor situation</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_col3:
                    low_risk = risk_counts.get('Low Risk', 0)
                    st.markdown(f"""
                    <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #2e7d32;'>
                    <h4 style='color: #2e7d32; margin: 0;'>🟢 Low Risk</h4>
                    <h2 style='margin: 0.5rem 0;'>{low_risk}</h2>
                    <p style='margin: 0; color: #666;'>Standard protocols</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display results table
                st.markdown("---")
                st.markdown("### 📋 Detailed Results")
                
                # Add filtering options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Level:",
                        options=['High Risk', 'Moderate Risk', 'Low Risk'],
                        default=['High Risk', 'Moderate Risk', 'Low Risk']
                    )
                with filter_col2:
                    tolerance_filter = st.multiselect(
                        "Filter by Predicted Tolerance:",
                        options=['Tolerant', 'Not Tolerant'],
                        default=['Tolerant', 'Not Tolerant']
                    )
                
                filtered_df = df[
                    df['Risk_Level'].isin(risk_filter) & 
                    df['Predicted_Tolerance'].isin(tolerance_filter)
                ]
                
                st.dataframe(filtered_df, height=400, use_container_width=True)
                st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
                
                # Download options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_full = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Complete Results",
                        csv_full,
                        f"FWP_Wolf_Assessment_Results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        type="primary",
                        use_container_width=True
                    )
                
                with col2:
                    # High risk subset for priority action
                    high_risk_df = df[df['Risk_Level'] == 'High Risk']
                    if len(high_risk_df) > 0:
                        csv_high_risk = high_risk_df.to_csv(index=False)
                        st.download_button(
                            "🔴 Download High Risk Cases Only",
                            csv_high_risk,
                            f"FWP_High_Risk_Cases_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("No high risk cases identified")
                
                # Visualization section
                st.markdown("---")
                st.markdown("### 📈 Visual Analytics")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Tolerance distribution
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    pred_counts = df['Predicted_Tolerance'].value_counts()
                    colors_pred = ['#c62828', '#2e7d32']
                    bars = ax1.bar(pred_counts.index, pred_counts.values, color=colors_pred, edgecolor='black', linewidth=1.5)
                    ax1.set_title('Tolerance Classification Distribution', fontsize=14, fontweight='bold', pad=15)
                    ax1.set_ylabel('Number of Individuals', fontsize=11, fontweight='bold')
                    ax1.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # Add count labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}',
                                ha='center', va='bottom', fontweight='bold', fontsize=12)
                    
                    st.pyplot(fig1)
                    plt.close()
                
                with viz_col2:
                    # Confidence distribution
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    ax2.hist(df['Confidence'], bins=20, color='#1a472a', edgecolor='black', linewidth=1.2)
                    ax2.axvline(df['Confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Confidence"].mean():.1%}')
                    ax2.set_title('Model Confidence Distribution', fontsize=14, fontweight='bold', pad=15)
                    ax2.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
                    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    ax2.legend(fontsize=10)
                    ax2.grid(axis='y', alpha=0.3, linestyle='--')
                    st.pyplot(fig2)
                    plt.close()
                
                # Stakeholder group analysis
                st.markdown("---")
                st.markdown("### 👥 Stakeholder Group Analysis")
                
                group_analysis = df.groupby('group').agg({
                    'Predicted_Tolerance': lambda x: (x == 'Tolerant').mean(),
                    'Confidence': 'mean',
                    'Q36': 'count'
                }).round(3)
                group_analysis.columns = ['Tolerance Rate', 'Avg Confidence', 'Sample Size']
                group_analysis['Tolerance Rate'] = (group_analysis['Tolerance Rate'] * 100).round(1).astype(str) + '%'
                group_analysis['Avg Confidence'] = (group_analysis['Avg Confidence'] * 100).round(1).astype(str) + '%'
                
                st.dataframe(group_analysis, use_container_width=True)
                
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                group_tolerance = df.groupby(['group', 'Predicted_Tolerance']).size().unstack(fill_value=0)
                group_tolerance.plot(kind='bar', ax=ax3, color=['#c62828', '#2e7d32'], edgecolor='black', linewidth=1.2)
                ax3.set_title('Tolerance by Stakeholder Group', fontsize=14, fontweight='bold', pad=15)
                ax3.set_xlabel('Stakeholder Group', fontsize=11, fontweight='bold')
                ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
                ax3.legend(title='Classification', fontsize=10)
                ax3.grid(axis='y', alpha=0.3, linestyle='--')
                plt.xticks(rotation=0)
                st.pyplot(fig3)
                plt.close()
                
                # Management recommendations
                st.markdown("---")
                st.markdown("### 📋 Example Agency Actions")
                st.caption("*Demonstrating operational use cases for wildlife management agencies*")
                
                if high_risk_count > 0:
                    st.error(f"""
                    **🔴 HIGH PRIORITY SCENARIO:** {high_risk_count} individuals classified as high risk for intolerance
                    
                    **Example Agency Response:**
                    1. Deploy conflict prevention resources to high-risk areas
                    2. Initiate targeted outreach emphasizing coexistence benefits
                    3. Consider range rider programs or livestock protection measures
                    4. Schedule community meetings for high-risk stakeholder groups
                    5. Monitor for potential human-wolf conflicts
                    """)
                
                if mod_risk > 0:
                    st.warning(f"""
                    **🟡 MODERATE PRIORITY SCENARIO:** {mod_risk} individuals with uncertain classifications
                    
                    **Example Agency Response:**
                    1. Conduct follow-up surveys for accurate assessment
                    2. Standard engagement and education protocols
                    3. Monitor attitude shifts over time
                    4. Provide balanced information on wolf management
                    """)
                
                if low_risk > 0:
                    st.success(f"""
                    **🟢 POSITIVE OUTLOOK:** {low_risk} individuals show strong tolerance indicators
                    
                    **Example Opportunities:**
                    1. Recruit as community ambassadors for coexistence
                    2. Leverage for positive messaging and case studies
                    3. Maintain engagement with minimal resource allocation
                    4. Use as success stories in outreach materials
                    """)
                
            else:
                st.error(f"❌ **Invalid CSV Format**")
                st.markdown(f"""
                Your file is missing required columns. 
                
                **Required:** {', '.join(required_cols)}
                
                **Found in your file:** {', '.join(df.columns)}
                
                Please download the template above and ensure your data matches the format.
                """)
        
        except Exception as e:
            st.error(f"❌ **Error Processing File**")
            st.exception(e)
            st.info("Please check that your CSV is properly formatted and try again. Download the template above if needed.")

# Page 3: Model Documentation
elif page == "📈 Model Documentation":
    st.markdown("### Model Performance & Technical Documentation")
    st.markdown("Comprehensive overview of model development, validation, and operational characteristics")
    
    # Executive summary
    st.markdown("---")
    st.markdown("#### 📊 Executive Summary")
    
    summary_col1, summary_col2 = st.columns([2, 1])
    
    with summary_col1:
        st.markdown("""
        This project demonstrates a **Logistic Regression model with ADASYN** (Adaptive Synthetic Sampling) 
        for predicting wolf tolerance among Montana residents. The model was developed using publicly available 
        survey data from 2,146 Montana residents collected in 2023, representing diverse stakeholder groups 
        including general public, landowners, wolf hunters, and deer hunters.
        
        **Key Findings:**
        - Wildlife value orientations are **2-16× more predictive** than demographic factors
        - Utilitarian values show the strongest negative effect (coefficient: -0.84)
        - Model achieves 70% accuracy with balanced performance across tolerance classes
        - ADASYN balancing strategy optimizes for minority class detection (intolerant individuals)
        
        **Portfolio Demonstration:**
        - Production-quality ML system architecture
        - Agency-appropriate UI/UX design
        - Interpretable model with actionable outputs
        - Professional documentation standards
        """)
    
    with summary_col2:
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 1.5rem; border-radius: 8px; border-left: 5px solid #1a472a;'>
        <h4 style='color: #1a472a; margin-top: 0;'>Model Specifications</h4>
        <p><strong>Algorithm:</strong> Logistic Regression</p>
        <p><strong>Balancing:</strong> ADASYN</p>
        <p><strong>Test Accuracy:</strong> 70%</p>
        <p><strong>F1-Macro:</strong> 0.691</p>
        <p><strong>Training N:</strong> 2,146</p>
        <p><strong>Features:</strong> 5</p>
        <p><strong>Framework:</strong> Scikit-learn</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed performance metrics
    st.markdown("---")
    st.markdown("#### 🎯 Model Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Test Accuracy", "70.0%", help="Overall classification accuracy on held-out test set")
    with perf_col2:
        st.metric("F1-Macro Score", "0.691", help="Balanced F1 score across both classes")
    with perf_col3:
        st.metric("Minority Class F1", "0.65", help="F1 score for 'Not Tolerant' class")
    with perf_col4:
        st.metric("Majority Class F1", "0.73", help="F1 score for 'Tolerant' class")
    
    # Class-specific performance table
    st.markdown("---")
    st.markdown("#### 📋 Performance by Classification")
    
    metrics_df = pd.DataFrame({
        'Classification': ['Not Tolerant (Minority)', 'Tolerant (Majority)'],
        'Precision': [0.59, 0.79],
        'Recall': [0.73, 0.68],
        'F1-Score': [0.65, 0.73],
        'Test Set Support': [211, 325]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **Performance Interpretation:**
    
    - **Precision (Not Tolerant: 59%)**: When model predicts intolerance, it's correct 59% of the time. This conservative approach minimizes false alarms while capturing genuine high-risk cases.
    
    - **Recall (Not Tolerant: 73%)**: Model successfully identifies 73% of truly intolerant individuals, making it effective for proactive conflict prevention.
    
    - **Balanced Approach**: The model maintains reasonable performance across both classes, avoiding excessive bias toward the majority class.
    """)
    
    # Model development rationale
    st.markdown("---")
    st.markdown("#### 🔬 Model Development & Strategy Selection")
    
    with st.expander("Why ADASYN? Comparing Balancing Strategies", expanded=True):
        st.markdown("""
        **Challenge:** Original dataset showed class imbalance (60% Tolerant, 40% Not Tolerant), risking 
        poor performance on the minority class (intolerant individuals) which are most critical for FWP to identify.
        
        **Solution:** Four balancing strategies were compared:
        """)
        
        strategy_comparison = pd.DataFrame({
            'Strategy': ['No Balancing', 'Class Weights', 'SMOTE', 'ADASYN (Selected)'],
            'Test Accuracy': ['72%', '68%', '69%', '70%'],
            'F1-Macro': [0.666, 0.677, 0.685, 0.691],
            'Minority F1': [0.60, 0.63, 0.64, 0.65],
            'Minority Recall': [0.62, 0.71, 0.70, 0.73]
        })
        
        st.dataframe(strategy_comparison, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **ADASYN Selected Because:**
        1. **Highest F1-Macro (0.691)**: Best balanced performance across both classes
        2. **Best Minority Class F1 (0.65)**: Most effective at identifying intolerant individuals
        3. **Highest Minority Recall (73%)**: Captures more high-risk cases for proactive management
        4. **Adaptive Synthesis**: Generates more synthetic samples in difficult-to-classify regions
        
        **Trade-off:** Slight reduction in overall accuracy (70% vs 72%) is acceptable given the 
        substantial improvement in detecting the operationally-critical minority class.
        """)
    
    # Feature importance
    st.markdown("---")
    st.markdown("#### 🔍 Feature Importance Analysis")
    
    st.markdown("""
    Logistic regression coefficients indicate the change in log-odds of tolerance for each unit increase 
    in the predictor variable. Larger absolute values indicate stronger influence on predictions.
    """)
    
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#c62828' if c < 0 else '#2e7d32' for c in coef_df['Coefficient']]
    bars = ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['Feature'], fontsize=11)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Coefficient Value (Effect on Log-Odds of Tolerance)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Predictors of Wolf Tolerance', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add coefficient values on bars
    for i, (bar, coef) in enumerate(zip(bars, coef_df['Coefficient'])):
        ax.text(coef + (0.03 if coef > 0 else -0.03), i, f'{coef:.3f}',
                va='center', ha='left' if coef > 0 else 'right', fontweight='bold', fontsize=10)
    
    # Enhanced legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2e7d32', label='Increases Tolerance (Positive Coefficient)'),
        Patch(facecolor='#c62828', label='Decreases Tolerance (Negative Coefficient)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Interpretation of coefficients
    st.markdown("---")
    st.markdown("#### 💡 Key Insights from Feature Analysis")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        **🔴 Strongest Negative Predictors (Decrease Tolerance):**
        
        1. **Utilitarian Values (coef: -0.84)**
           - By far the strongest predictor
           - Each 1-point increase decreases tolerance log-odds by 0.84
           - Practical implication: Individuals viewing wildlife primarily as resources show significantly lower tolerance
        
        2. **Landowner Status (coef: -0.35)**
           - Second strongest predictor
           - Landowners show ~40% lower tolerance than general public
           - Likely reflects concerns about livestock depredation and property rights
        
        3. **Deer Hunter Status (coef: -0.21)**
           - Moderate negative effect
           - May reflect competition concerns over prey species
        """)
    
    with insight_col2:
        st.markdown("""
        **🟢 Strongest Positive Predictors (Increase Tolerance):**
        
        1. **General Public Baseline (coef: +0.32)**
           - Non-stakeholder residents show higher tolerance
           - Less direct interaction/conflict with wolves
        
        2. **Mutualist Values (coef: +0.32)**
           - Strong positive effect
           - Viewing wildlife as companions increases acceptance
           - Each 1-point increase raises tolerance log-odds by 0.32
        
        3. **Wolf Hunter Status**
           - Variable effect depending on other factors
           - Direct interaction with wolves through hunting
        
        **📊 Demographic Effects (Minimal):**
        - Age (coef: -0.05): Negligible effect
        - Gender (coef: +0.09): Very small effect
        """)
    
    st.success("""
    **🎯 Strategic Insight for Wildlife Agencies:**
    
    Wildlife value orientations (mutualism and utilitarianism) are **2-16× more influential** than demographic 
    characteristics. This finding suggests that educational outreach emphasizing:
    1. Ecological benefits of wolves (appeals to mutualist values)
    2. Economic coexistence strategies (addresses utilitarian concerns)
    3. Science-based wolf management (builds trust across value orientations)
    
    ...would likely be far more effective than demographic-targeted campaigns alone.
    
    **Portfolio Note:** This demonstrates how data science can inform evidence-based policy and resource allocation.
    """)
    
    # Model limitations and best practices
    st.markdown("---")
    st.markdown("#### ⚠️ Limitations & Best Practices")
    
    limit_col1, limit_col2 = st.columns(2)
    
    with limit_col1:
        st.markdown("""
        **Model Limitations:**
        
        🔸 **70% Accuracy Ceiling**
        - Human attitudes are complex and multifaceted
        - 30% error rate indicates predictions should supplement, not replace, direct assessment
        
        🔸 **Limited Feature Set**
        - Only 5 predictors captured
        - Missing factors: personal wolf encounters, economic dependence on livestock, trust in agencies, prior conflict history
        
        🔸 **Moderate Confidence Zone (60-65%)**
        - ~25% of predictions fall in this range
        - Borderline cases require follow-up assessment
        
        🔸 **Temporal Validity**
        - Model trained on 2023 data
        - Public attitudes may shift with events (wolf attacks, policy changes)
        - Recommend annual retraining
        
        🔸 **Geographic Generalizability**
        - Trained exclusively on Montana residents
        - May not apply to other wolf range states without local validation
        """)
    
    with limit_col2:
        st.markdown("""
        **Best Practice Recommendations:**
        
        ✅ **Appropriate Use Cases:**
        - Initial screening of new residents in wolf habitat
        - Community-level risk assessment for resource allocation
        - Identifying high-risk areas for proactive outreach
        - Prioritizing limited staff time and funding
        
        ✅ **Deployment Guidelines:**
        - Use predictions as **decision support**, not definitive labels
        - Always combine with local field knowledge
        - For high-stakes decisions (e.g., compensation programs), validate with direct surveys
        - Monitor prediction accuracy over time and retrain as needed
        
        ✅ **Confidence Thresholds:**
        - **High confidence (>75%)**: Reliable for resource allocation
        - **Moderate confidence (65-75%)**: Reasonable guidance, monitor closely
        - **Low confidence (<65%)**: Conduct direct assessment
        
        ✅ **Ethical Considerations:**
        - Predictions should inform outreach, not discriminate
        - Avoid labeling individuals without their knowledge
        - Focus on community-level patterns, not individual targeting
        - Respect privacy and data confidentiality
        """)
    
    # Technical specifications
    st.markdown("---")
    with st.expander("🛠️ Technical Specifications"):
        st.markdown("""
        **Model Architecture:**
        - **Algorithm**: Scikit-learn Logistic Regression (`LogisticRegression`)
        - **Solver**: lbfgs (Limited-memory BFGS)
        - **Regularization**: L2 penalty (Ridge)
        - **Max Iterations**: 1000
        - **Random State**: 42 (reproducibility)
        
        **Preprocessing Pipeline:**
        - **Numerical Features**: Standard scaling (mean=0, std=1)
        - **Categorical Features**: One-hot encoding
        - **Balancing**: ADASYN applied to training set only
        
        **Training Configuration:**
        - **Training Set**: 70% (n=1,502)
        - **Test Set**: 30% (n=536)
        - **Cross-Validation**: 5-fold stratified CV on training set
        - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
        
        **Model Persistence:**
        - **Format**: Joblib pickle (`.pkl`)
        - **Version**: Scikit-learn 1.3.x compatible
        - **File Size**: ~50KB
        - **Load Time**: <100ms
        
        **Deployment Environment:**
        - **Framework**: Streamlit 1.x
        - **Python**: 3.9+
        - **Dependencies**: pandas, numpy, matplotlib, scikit-learn, joblib
        """)
    
    # Future enhancements
    st.markdown("---")
    st.markdown("#### 🚀 Future Enhancement Roadmap")
    
    st.markdown("""
    **Potential Model Improvements:**
    
    1. **Feature Expansion**
       - Incorporate spatial data (distance to wolf packs, land use type)
       - Add temporal features (seasonality, recent wolf activity)
       - Include trust in FWP/government agencies
       - Capture personal experience with wolves (sightings, conflicts)
    
    2. **Model Sophistication**
       - Experiment with ensemble methods (Random Forest, Gradient Boosting)
       - Test neural networks for complex interaction effects
       - Develop separate models for each stakeholder group
       - Implement time-series forecasting for attitude trends
    
    3. **Operational Enhancements**
       - Real-time data integration (incident reports, social media sentiment)
       - Geographic visualization dashboard (risk heat maps)
       - Automated alert system for emerging high-risk communities
       - Mobile application for field staff data collection
    
    4. **Research Extensions**
       - Validate model in other wolf range states (Idaho, Wyoming, Wisconsin)
       - Longitudinal studies tracking attitude changes
       - Intervention effectiveness studies (does outreach improve tolerance?)
       - Cost-benefit analysis of prediction-guided resource allocation
    """)

# Footer with professional branding
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; background-color: #f5f5f5; border-radius: 10px; margin-top: 3rem;'>
    <p style='font-size: 1.1rem; font-weight: 600; color: #1a472a; margin-bottom: 0.5rem;'>
        Wolf Tolerance Prediction System
    </p>
    <p style='font-size: 0.95rem; color: #5a6c57; margin-bottom: 1rem;'>
        Portfolio Project | Data Science for Conservation
    </p>
    <p style='font-size: 0.85rem; color: #7a8a77;'>
        <strong>Model Type:</strong> Logistic Regression + ADASYN | 
        <strong>Framework:</strong> Scikit-learn + Streamlit | 
        <strong>Data Source:</strong> Montana Wolf Attitude Survey (2023, Public Dataset)
    </p>
    <p style='font-size: 0.8rem; color: #999; margin-top: 1rem;'>
        <strong>Disclaimer:</strong> Independent portfolio project demonstrating ML applications for wildlife management.<br>
        Not affiliated with or endorsed by any government agency. Built for educational and professional demonstration purposes.
    </p>
</div>
""", unsafe_allow_html=True)