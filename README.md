# Physician_Conversion_Model
ML_Engineering_repo

The Physician Conversion Model aims to predict the likelihood of a healthcare provider (HCP) becoming a first-time writer for a particular company or brand. This is a key use case in the lifesciences and healthcare industry, enabling marketing and sales teams to identify high-potential HCPs who have not yet prescribed a company’s product but exhibit characteristics of future converters.

This repository is not only focused on building the predictive model but also on developing an end-to-end scalable, production-ready AI/ML solution.

# Tasks
- Harshit: to review the notebooks pipeline and add comments so that it becomes easy to teach
- Harshit: to prepare a flow for the streamlit application 
- Asheesh: Once flow is finalized, modify the streamlit application to enhance the UX

# MLFLOW Command:
mlflow server --backend-store-uri sqlite:///E:/Harshit/proj_1/Physician_Conversion_Model/mlflow/mlflow.db --default-artifact-root file:///E:/Harshit/proj_1/Physician_Conversion_Model/mlflow/mlruns --host 0.0.0.0 --port 5000