import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Load model and preprocessing objects ---
@st.cache_resource
def load_models():
    """
    Load all model artifacts needed for ADHD prediction.
    
    Returns:
    --------
    tuple
        All loaded artifacts needed for prediction including:
        - scaler: StandardScaler for quantitative features
        - imputer: KNNImputer for missing value handling
        - pca: PCA transformer for connectome dimensionality reduction
        - trained_model: Final optimized Logistic Regression model
        - feature lists: Column names for different feature types
        - model_report: Performance metrics and model information
    """
    try:
        # Load preprocessing objects that were fitted during training
        scaler = joblib.load('scaler.joblib')
        imputer = joblib.load('imputer.joblib')
        pca = joblib.load('pca_connectome.joblib')
        
        # Load model performance report and configuration
        with open('adhd_model_report.json', 'r') as f:
            model_report = json.load(f)
        
        # Extract model information from report
        model_info = model_report['model_info']
        model_name = model_info['name']  # "Logistic Regression"
       
        # Load the trained model using dynamic filename from report
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        trained_model = joblib.load(model_filename)
        
        # Load feature column definitions to ensure correct preprocessing order
        with open('quant_cols.json', 'r') as f:
            quant_cols = json.load(f)
        with open('cat_cols.json', 'r') as f:
            cat_cols = json.load(f)
        with open('conn_cols.json', 'r') as f:
            conn_cols = json.load(f)
        
        return (scaler, imputer, pca, trained_model, 
                quant_cols, cat_cols, conn_cols, model_report)

    except FileNotFoundError as e:
        st.error(f"Required model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load all artifacts needed for prediction pipeline
(scaler, imputer, pca, model, quant_cols, cat_cols, conn_cols, model_report) = load_models()

# --- Streamlit UI ---
st.title("🧠 ADHD Prediction App")
st.markdown("---")

# Display model information and performance metrics in sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.info(f"""
**Algorithm:** {model_report['model_info']['name']}  
**Features:** {len(quant_cols + cat_cols + conn_cols)} total  
**Quantitative:** {len(quant_cols)}  
**Categorical:** {len(cat_cols)}  
**Connectome:** {len(conn_cols)}  

**Performance Metrics:**
- Test F1-Score: {model_report['performance']['test_f1_macro']:.3f}
- Test ROC AUC: {model_report['performance']['test_roc_auc']:.3f}
- ADHD Precision: {model_report['performance']['test_precision_adhd']:.3f}
- ADHD Recall: {model_report['performance']['test_recall_adhd']:.3f}
""")

# Optional data dictionary for user reference
with st.expander("ℹ️ Data Dictionary"):
    try:
        # Load data dictionary to help users understand feature meanings
        data_dict_df = pd.read_excel("C:\\Users\\04ama\\OneDrive\\chemistry\\ADHD_SEX_Prediction\\data\\Data Dictionary\\Data Dictionary (1).xlsx")
        st.dataframe(data_dict_df)
    except Exception as e:
        st.warning(f"Data dictionary not found: {e}")

st.write("Please provide the following information:")

# --- User Inputs with organized layout ---
st.subheader("📋 Questionnaire Data")

# Create two-column layout for better space utilization
col1, col2 = st.columns(2)

# Split quantitative columns for two-column input
mid_point = len(quant_cols) // 2

user_quant = {}
with col1:
    st.write("**Quantitative Features (Part 1):**")
    for col in quant_cols[:mid_point]:
        # Number inputs for continuous variables with default value 0
        user_quant[col] = st.number_input(f"{col}", value=0.0, key=f"quant_{col}")

with col2:
    st.write("**Quantitative Features (Part 2):**")
    for col in quant_cols[mid_point:]:
        user_quant[col] = st.number_input(f"{col}", value=0.0, key=f"quant_{col}")

st.subheader("📝 Categorical Data")
user_cat = {}
for col in cat_cols:
    # Binary categorical variables with clear Yes/No labels
    if 'Sex_F' in col:
        user_cat[col] = st.selectbox(
            f"{col}", 
            options=[0, 1], 
            format_func=lambda x: "Female" if x ==1 else "Male",
            key=f"cat_{col}"
        )
    else:
        user_cat[col] = st.number_input(
            f"{col}", 
            min_value = 0,
            max_value = 20,
            value =0,
            step =1,
            key=f"cat_{col}"
        )

st.subheader("🧠 Connectome Data")
st.write("Upload your connectome data as a CSV file:")
conn_file = st.file_uploader("Upload connectome CSV", type=["csv"])

# Handle connectome data upload or use defaults
if conn_file is not None:
    try:
        conn_df = pd.read_csv(conn_file)
        
        # Map uploaded connectome data to expected features
        user_conn = {}
        missing_cols = []
        for col in conn_cols:
            if col in conn_df.columns:
                # Use first row of uploaded data, convert to float for consistency
                user_conn[col] = float(conn_df.iloc[0][col])
            else:
                # Use default value for missing features
                user_conn[col] = 0.0
                missing_cols.append(col)
        
        # Inform user about data completeness
        if missing_cols:
            st.warning(f"Missing {len(missing_cols)} connectome features. Using default values.")
        else:
            st.success(f"Loaded connectome data with {len(conn_df.columns)} features")
            
    except Exception as e:
        st.error(f"Error loading connectome file: {e}")
        # Fallback to default values if upload fails
        user_conn = {col: 0.0 for col in conn_cols}
else:
    # Use default values when no file is uploaded
    user_conn = {col: 0.0 for col in conn_cols}
    st.info("ℹ️ Using default values (0.0) for connectome data")

# --- Prepare input data for model prediction ---
# Combine all user inputs into single DataFrame row
input_df = pd.DataFrame([{**user_quant, **user_cat, **user_conn}])

# --- Prediction Pipeline ---
st.markdown("---")
if st.button("🎯 Predict ADHD", type="primary"):
    try:
        with st.spinner("Processing data"):
            
            # Ensure input has all expected columns in correct order
            all_expected_cols = quant_cols + cat_cols + conn_cols
            processed_df = input_df.reindex(columns=all_expected_cols, fill_value=0.0)
            
            # Step 1: Scale quantitative features using fitted scaler
            if len(quant_cols) > 0:
                processed_df[quant_cols] = scaler.transform(processed_df[quant_cols])
            
            # Step 2: Impute missing values for questionnaire data
            non_conn_cols = quant_cols + cat_cols
            if len(non_conn_cols) > 0:
                processed_df[non_conn_cols] = imputer.transform(processed_df[non_conn_cols])
            
            # Step 3: Apply PCA dimensionality reduction to connectome features
            if len(conn_cols) > 0:
                conn_pca = pca.transform(processed_df[conn_cols])
                # Create DataFrame with PCA component names
                pca_cols = [f'conn_pca_{i+1}' for i in range(conn_pca.shape[1])]
                conn_pca_df = pd.DataFrame(conn_pca, columns=pca_cols, index=processed_df.index)
                
                # Replace original connectome columns with PCA components
                processed_df = processed_df.drop(columns=conn_cols)
                processed_df = pd.concat([processed_df, conn_pca_df], axis=1)
            
            # Step 4: Generate prediction probability and classification
            proba = model.predict_proba(processed_df)[0][1]  # Probability of ADHD class
            prediction = int(proba >= 0.5)  # Binary classification using 0.5 threshold
        
        # Display prediction results with comprehensive metrics
        st.success("Prediction completed!")
        
        st.subheader("🎯 Prediction Results")
        
        # Create metrics display in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ADHD Probability", f"{proba:.1%}")
        
        with col2:
            st.metric("Threshold Used", "50.0%")
        
        with col3:
            # Confidence is the maximum of predicted probability and its complement
            confidence = max(proba, 1-proba)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Main prediction result with clinical interpretation
        if prediction == 1:
            st.error("🔴 **Prediction: LIKELY ADHD**")
            st.write(f"The model predicts this individual **likely has ADHD** with **{proba:.1%}** probability.")
            
            # Provide confidence-based clinical guidance
            if proba > 0.8:
                st.write("⚠️ **High confidence prediction** - Consider clinical evaluation.")
            elif proba > 0.6:
                st.write("📋 **Moderate confidence** - Additional assessment recommended.")
            else:
                st.write("❓ **Low confidence** - Consider additional data or assessment.")
                
        else:
            st.success("🟢 **Prediction: UNLIKELY ADHD**")
            st.write(f"The model predicts this individual **unlikely has ADHD** with **{(1-proba):.1%}** probability.")
            
            # Provide guidance for negative predictions
            if (1-proba) > 0.8:
                st.write("High confidence prediction - Low ADHD likelihood.")
            elif (1-proba) > 0.6:
                st.write("📊 **Moderate confidence** - Monitor if symptoms develop.")
            else:
                st.write("❓ **Low confidence** - Consider additional assessment if symptoms present.")
        
        # Display model reliability information from validation
        st.info(f"""
        **Model Reliability:** 
        - Stability: {model_report['stability']['mean_f1_macro']:.3f} ± {model_report['stability']['std_f1_macro']:.3f} F1-Score
        - Cross-validation tested with multiple random seeds for robustness
        """)
        
        # Technical details for transparency and debugging
        with st.expander("🔧 Technical Details"):
            st.write(f"**Model:** {model_report['model_info']['name']}")
            st.write(f"**Features processed:** {processed_df.shape[1]}")
            st.write(f"**Raw probability:** {proba:.4f}")
            st.write(f"**Decision threshold:** 0.5000")
            st.write(f"**Final prediction:** {'ADHD' if prediction == 1 else 'No ADHD'}")
            
            # Show comprehensive model performance metrics
            st.write("**Model Performance on Test Set:**")
            perf = model_report['performance']
            st.write(f"- Test F1-Macro: {perf['test_f1_macro']:.4f}")
            st.write(f"- Test ROC AUC: {perf['test_roc_auc']:.4f}")
            st.write(f"- ADHD Precision: {perf['test_precision_adhd']:.4f}")
            st.write(f"- ADHD Recall: {perf['test_recall_adhd']:.4f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Please check that all inputs are valid and try again.")
        
        # Debug information for troubleshooting
        with st.expander("🐛 Debug Information"):
            st.write(f"Input DataFrame shape: {input_df.shape}")
            st.write(f"Input DataFrame columns: {list(input_df.columns)}")
            st.write("Error details:", str(e))

# Medical disclaimer for ethical AI deployment
st.markdown("---")
st.markdown("""
**⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. 
It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of qualified healthcare providers with questions about medical conditions.
""")