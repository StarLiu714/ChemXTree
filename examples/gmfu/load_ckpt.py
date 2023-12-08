import pandas as pd
from ChemXTree.GMFU import TabularModel


# Loading and preparing the data...
dataset_name = "BBBP"
train_df = pd.read_csv("/ChemXTree/"+dataset_name+"/train_fingerprint.csv").iloc[:, 1:]
train_df = train_df.rename(columns={"targets": "y"})
valid_df = pd.read_csv("/ChemXTree/"+dataset_name+"/valid_fingerprint.csv").iloc[:, 1:]
valid_df = valid_df.rename(columns={"targets": "y"})
test_df = pd.read_csv("/ChemXTree/"+dataset_name+"/test_fingerprint.csv").iloc[:, 1:]
test_df = test_df.rename(columns={"targets": "y"})

best_model =TabularModel.load_model(
    "model_auc_0_75488968108681",
    strict=False)


X_test = test_df.drop('y', axis=1)
y_test = test_df['y']

# prediction on validation set
test_pred_df = best_model.predict(X_test)
print(test_pred_df)