import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import shap
import matplotlib.pyplot as plt
from Pages import model_inspection
import threading
from explainerdashboard import ClassifierExplainer, ExplainerDashboard



with open("output_folder/physician_conversion.pkl", "rb") as f:
    conversion_classifer = pickle.load(f)

df_validation = pd.read_csv('output_folder/Validation_data.csv')
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)

def inference_output(df_inference, df_validation, conversion_classifier,n):
    df_inference.drop(['Unnamed: 0'], axis=1, inplace=True)

    # id col not to be considered while training
    drop_id_col_list = ['NPI_ID', 'HCP_ID']
    X_inference = df_inference.drop(drop_id_col_list, axis=1)

    # Predictions
    df_inference['Prediction'] = conversion_classifier.predict(X_inference)

    # Creating validation set (assuming 'TARGET' is the target column)
    X_validation = df_validation.drop(['NPI_ID', 'HCP_ID', 'TARGET'], axis=1)
    X_validation = X_validation.astype(float)

    # Explain the model using a TreeExplainer
    explainer = shap.TreeExplainer(conversion_classifier)
    
    # Calculate SHAP values for inference data
    shap_values = explainer.shap_values(X_inference)

    # Plot feature importance using a bar plot
    shap.summary_plot(shap_values, X_inference, plot_type='bar')
    plt.savefig('shap_bar_plot.png', bbox_inches='tight')
    fig_1 = Image.open('shap_bar_plot.png')

    # shap_target_positive = shap_values[1]
    # shap.plots.bar(shap_values)
    # plt.savefig('shap_bar_plot_fig_2.png', bbox_inches='tight')
    # fig_2 = Image.open('shap_bar_plot_fig_2.png')

    shap.summary_plot(shap_values, X_inference)
    plt.savefig('shap_bar_plot_fig_3.png', bbox_inches='tight')
    fig_3 = Image.open('shap_bar_plot_fig_3.png')

    #Extracting top features for each Physician

    # Create a new DataFrame to store the top features for each row
    df = df_inference.loc[df_inference['Prediction'] == 1.0].reset_index(drop=True)

    id_col_list = ['NPI_ID', 'HCP_ID','Prediction']
    top_features_df = pd.DataFrame(index=df.index)
    
    # Iterate through rows and extract top n features
    for row_idx in range(len(df)):
        shap_values_row = shap_values[row_idx]
        
        # Get the absolute SHAP values
        abs_shap_values = abs(shap_values_row)
        
        # Get indices of top n features
        top_feature_indices = abs_shap_values.argsort()[-n:][::-1]
        
        # Get corresponding feature names
        top_feature_names = df.drop(id_col_list, axis=1).columns[top_feature_indices]
        
        # Add the id_col_list column values to the new DataFrame
        for col in id_col_list:
            top_features_df.loc[row_idx, col] = df.loc[row_idx, col]
        
        # Add the top feature names to the new DataFrame
        for i in range(n):
            top_features_df.loc[row_idx, f'REASON{i+1}'] = top_feature_names[i]

    return df_inference, fig_1, fig_3, top_features_df

# Get the explainer object from model_inspection.py
explainer = model_inspection.get_explainer()


def main():
    st.set_page_config(
        page_title="Physician Conversion Application",
        layout="wide"
    )

    st.sidebar.header("Let's explore some data")
    st.sidebar.markdown("""
    Welcome to this Exploratory Data Analysis and Model Testing app.
    """)
    
    PAGE_MAP = {
    "Model Inspection": model_inspection
    }

    st.sidebar.header("Page Navigation")
    current_page = st.sidebar.radio("Go To", list(PAGE_MAP), key='sidebar')


    custom_style = """
    <style>
    body {
        background-color: #00000; /* Set your desired background color */
    }
    .side-by-side {
        display: flex;
    }
    .side-by-side > * {
        flex: 1;
        padding: 10px;
    }
    </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)

    logo_path = "Input_data\propensity-chart.gif"  # Specify the path to your logo image
    logo_image = Image.open(logo_path)

    st.image(logo_image, use_column_width=True)

    st.title("Physician Conversion Application")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("Uploaded CSV Data"):
            st.write("Uploaded CSV Data:")
            st.write(df)

        n = st.selectbox("Select the number of top features", [1, 2, 3, 4, 5])

        df_output, fig_1, fig_3, df_recommendation = inference_output(df, df_validation, conversion_classifer, n)

        st.write("Inference Data with Prediction:")
        st.write(df_output)

        st.write("SHAP PLOTS:")

        # Arrange images side by side using CSS
        st.markdown('<div class="side-by-side">', unsafe_allow_html=True)
        
        st.markdown('<div>', unsafe_allow_html=True)
        st.write("SHAP BAR PLOT:")
        st.image(fig_1, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div>', unsafe_allow_html=True)
        st.write("SHAP SUMMARY PLOT:")
        st.image(fig_3, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("Inference Data with Recommendation:")
        st.write(df_recommendation)

        # Start the ExplainerDashboard in a separate thread
    # Show the Explainer Dashboard
    st.write("Explainer Dashboard:")
    #st.write(explainer)

    # Launch the Explainer Dashboard
    if st.button("Launch Explainer Dashboard"):
        dashboard = ExplainerDashboard(explainer, title="Physician Conversion Dashboard")
        dashboard.run()
    # if st.button("Launch Explainer Dashboard"):
    #     st.markdown("Click [here](http://192.168.1.7:8050) to open the Explainer Dashboard in a new tab.")
if __name__ == "__main__":
    main()