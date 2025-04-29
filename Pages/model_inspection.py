import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pickle

with open("output_folder/physician_conversion.pkl", "rb") as f:
    conversion_classifer = pickle.load(f)

df_validation = pd.read_csv('output_folder/Validation_data.csv')
df_validation.drop(['Unnamed: 0'], axis=1, inplace = True)

# Creating validation set (assuming 'TARGET' is the target column)
X_validation = df_validation.drop(['NPI_ID', 'HCP_ID', 'TARGET'], axis=1)
X_validation = X_validation.astype(float)
y_validation = df_validation['TARGET']
print(X_validation.shape)
print(y_validation.shape)



# Get the Explainer
def get_explainer():
    explainer = ClassifierExplainer(conversion_classifer, X_validation, y_validation)
    return explainer