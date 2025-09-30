import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Wolf Tolerance Predictor", page_icon="🐺", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #2c3e50; font-weight: bold;}
    .sub-header {font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🐺 Wolf Tolerance Prediction Tool</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predict Montana residents\' tolerance toward wolves based on wildlife values and demographics</p>',
    unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    return joblib.load('wolf_tolerance_model.pkl')


model = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Single Prediction", "Batch Predictions", "Model Insights"])

# Educational content in sidebar
with st.sidebar.expander("ℹ️ Understanding Value Orientations"):
    st.write("""
    **Mutualism (1-7 scale)**
    - Viewing wildlife as companions deserving care
    - Example: "Wolves have a right to exist"

    **Utilitarianism (1-7 scale)**
    - Viewing wildlife as resources for human use
    - Example: "Wildlife management should prioritize human needs"

    Research shows these values are 2-16× stronger predictors than demographics.
    """)

# Page 1: Single Prediction
if page == "Single Prediction":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Features")
        age = st.slider("Age", 18, 90, 45)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        group = st.selectbox("Stakeholder Group", ["GenPop", "Land", "Wolf", "Deer"],
                             help="GenPop=General Public, Land=Landowner, Wolf=Wolf Advocate, Deer=Deer Hunter")

        st.markdown("---")
        st.subheader("Wildlife Value Orientations")
        mutualism = st.slider("Mutualism (wildlife as companions)", 1.0, 7.0, 4.0, 0.1)
        utilitarianism = st.slider("Utilitarianism (wildlife as resources)", 1.0, 7.0, 4.0, 0.1)

        predict_button = st.button("🔮 Predict Tolerance", type="primary")

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

            # Display results
            st.subheader("Prediction Results")

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Prediction", prediction,
                          delta="Positive" if prediction == "Tolerant" else "Negative",
                          delta_color="normal" if prediction == "Tolerant" else "inverse")
            with metric_col2:
                st.metric("Confidence", f"{confidence:.1%}")

            # Probability visualization
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Not Tolerant', 'Tolerant'],
                'Probability': [proba[0], proba[1]]
            })
            st.bar_chart(prob_df.set_index('Class'), height=200)

            # Interpretation
            st.markdown("---")
            if prediction == "Tolerant":
                st.success(
                    f"✅ This individual is predicted to be **tolerant** toward wolves ({proba[1]:.1%} probability)")
            else:
                st.warning(
                    f"⚠️ This individual is predicted to be **not tolerant** toward wolves ({proba[0]:.1%} probability)")

            # Confidence warning
            if confidence < 0.65:
                st.info(f"""
                📊 **Moderate Confidence ({confidence:.0%})**

                This prediction has lower confidence, suggesting the individual's attitudes may be borderline 
                or their profile is unusual. Consider direct surveying for accurate assessment.
                """)

            # Explanation of prediction
            with st.expander("🔍 Why this prediction?"):
                st.write("**Key factors influencing this prediction:**")

                factors = []
                if utilitarianism > 5:
                    factors.append("- ⚠️ High utilitarian values (coefficient: -0.84) strongly predict intolerance")
                elif utilitarianism < 3:
                    factors.append("- ✅ Low utilitarian values increase tolerance likelihood")

                if mutualism > 5:
                    factors.append("- ✅ High mutualist values (coefficient: +0.32) increase tolerance")
                elif mutualism < 3:
                    factors.append("- ⚠️ Low mutualist values decrease tolerance likelihood")

                if group == "Land":
                    factors.append("- ⚠️ Landowner status (coefficient: -0.35) decreases tolerance")
                elif group == "GenPop":
                    factors.append("- ✅ General public baseline (coefficient: +0.32) increases tolerance")

                if gender == 2 and age > 50:
                    factors.append("- Demographic effects are minimal (gender: +0.09, age: -0.05)")

                for factor in factors:
                    st.write(factor)

# Page 2: Batch Predictions
elif page == "Batch Predictions":
    st.header("Batch Predictions")
    st.write("Upload a CSV file to predict tolerance for multiple individuals at once")

    # Sample CSV template
    with st.expander("📄 CSV Format Requirements"):
        st.write("""
        Your CSV must contain these columns:
        - **Q36**: Age (numeric, 18-90)
        - **Q37**: Gender (1=Female, 2=Male)
        - **group**: Stakeholder group (GenPop, Land, Wolf, or Deer)
        - **MUT_WVO**: Mutualism score (numeric, 1-7)
        - **UT_WVO**: Utilitarianism score (numeric, 1-7)
        """)

        # Sample data
        sample_df = pd.DataFrame({
            'Q36': [45, 32, 67],
            'Q37': [1, 2, 1],
            'group': ['GenPop', 'Land', 'Wolf'],
            'MUT_WVO': [5.2, 3.1, 6.8],
            'UT_WVO': [3.5, 6.2, 2.1]
        })
        st.dataframe(sample_df)

        csv_sample = sample_df.to_csv(index=False)
        st.download_button("Download Sample CSV", csv_sample, "sample_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            required_cols = ['Q36', 'Q37', 'group', 'MUT_WVO', 'UT_WVO']
            if all(col in df.columns for col in required_cols):

                with st.spinner("Processing predictions..."):
                    predictions = model.predict(df[required_cols])
                    probas = model.predict_proba(df[required_cols])

                    df['Predicted_Tolerance'] = predictions
                    df['Tolerant_Probability'] = probas[:, 1]
                    df['Confidence'] = probas.max(axis=1)

                st.success(f"✅ Processed {len(df)} records successfully!")

                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                tolerant_pct = (predictions == 'Tolerant').mean()
                avg_confidence = probas.max(axis=1).mean()

                col1.metric("Total Records", len(df))
                col2.metric("Predicted Tolerant", f"{tolerant_pct:.1%}")
                col3.metric("Predicted Intolerant", f"{(1 - tolerant_pct):.1%}")
                col4.metric("Avg Confidence", f"{avg_confidence:.1%}")

                # Display results
                st.subheader("Results Preview")
                st.dataframe(df, height=400)

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Complete Results",
                    csv,
                    "predictions_output.csv",
                    "text/csv",
                    type="primary"
                )

                # Visualization
                st.subheader("Distribution of Predictions")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Prediction counts
                pred_counts = df['Predicted_Tolerance'].value_counts()
                ax1.bar(pred_counts.index, pred_counts.values, color=['#e74c3c', '#2ecc71'])
                ax1.set_title('Prediction Counts')
                ax1.set_ylabel('Count')

                # Confidence distribution
                ax2.hist(df['Confidence'], bins=20, color='#3498db', edgecolor='black')
                ax2.set_title('Confidence Distribution')
                ax2.set_xlabel('Confidence')
                ax2.set_ylabel('Frequency')

                st.pyplot(fig)

            else:
                st.error(f"❌ CSV must contain columns: {required_cols}")
                st.write("Your CSV contains:", list(df.columns))

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Page 3: Model Insights
elif page == "Model Insights":
    st.header("Model Performance & Insights")

    # Performance metrics
    st.subheader("Test Set Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "70%")
    col2.metric("F1-Macro", "0.691")
    col3.metric("Training Samples", "2,146")
    col4.metric("Features", "5")

    # Class-specific performance
    st.subheader("Performance by Class")
    metrics_df = pd.DataFrame({
        'Class': ['Not Tolerant (Minority)', 'Tolerant (Majority)'],
        'Precision': [0.59, 0.79],
        'Recall': [0.73, 0.68],
        'F1-Score': [0.65, 0.73],
        'Support': [211, 325]
    })
    st.dataframe(metrics_df, hide_index=True)

    st.info("""
    **Model Strategy:** Logistic Regression with ADASYN (Adaptive Synthetic Sampling)

    ADASYN was selected after comparing 4 balancing strategies, achieving the best test F1-macro (0.691) 
    and best minority class F1 (0.65). The strategy improved minority class recall to 73% while maintaining 
    reasonable overall accuracy.
    """)

    # Feature importance
    st.subheader("Feature Importance (Logistic Regression Coefficients)")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coefficient']]
    bars = ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['Feature'])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Coefficient (Effect on Log-Odds of Tolerance)', fontsize=12)
    ax.set_title('Top 10 Predictors of Wolf Tolerance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#2ecc71', label='Increases Tolerance'),
        Patch(facecolor='#e74c3c', label='Decreases Tolerance')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    st.pyplot(fig)

    # Interpretation
    st.subheader("Key Findings")
    st.markdown("""
    1. **Value orientations dominate**: Utilitarianism (coef: -0.84) is 2.4× stronger than landowner status (-0.35)
    2. **Stakeholder groups matter**: Landowners show significantly lower tolerance; general public shows higher tolerance
    3. **Demographics are weak**: Gender (+0.09) and age (-0.05) have minimal predictive power
    4. **Actionable insight**: Education targeting wildlife value orientations may be 16× more effective than demographic-based outreach
    """)

    # Model limitations
    with st.expander("⚠️ Model Limitations & Considerations"):
        st.markdown("""
        **Limitations:**
        - 70% accuracy indicates human attitudes are complex and difficult to predict
        - Only 5 features; missing factors like personal wolf encounters, economic dependence, trust in agencies
        - Moderate confidence predictions (60-65%) suggest borderline cases may need direct assessment

        **Design Trade-offs:**
        - Prioritized simple, readily-obtainable features for practical deployment
        - Sacrificed some accuracy for ability to classify populations without extensive surveys
        - Logistic regression chosen for interpretability over slightly higher-performing black-box models

        **Use Cases:**
        - Classify new residents moving into wolf habitat
        - Target outreach based on predicted tolerance
        - Identify high-risk areas for proactive coexistence programs
        """)

# Footer
st.markdown("---")
st.caption("🎓 Portfolio Project | Model: Logistic Regression with ADASYN | Data: Montana Wolf Attitude Survey (2023)")