from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import roc_auc_score

run = Run.get_context()
ws = run.experiment.workspace

key = "creditcardfraud"
if key in ws.datasets.keys(): 
    dataset = ws.datasets[key] 
else:
    print("Please create a dataset")
    
df = dataset.to_pandas_dataframe()
y = df.pop("Class")

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(x_train)
x_train = imp.transform(x_train)
x_test = imp.transform(x_test)

imp = imp.fit(pd.DataFrame(y_train))
y_train = imp.transform(pd.DataFrame(y_train))
y_test = imp.transform(pd.DataFrame(y_test))


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest. Default = 100")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree. Default = None. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

    args = parser.parse_args()

    run.log("n_estimators:", np.float(args.n_estimators))
    run.log("max_depth:", np.int(args.max_depth))

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth).fit(x_train, y_train)

    # accuracy = model.score(x_test, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1], average="weighted")
    run.log("AUC_weighted", np.float(auc))
    
    #Missing part, need to serialize the model once it is trained, because azure only maintains logs and not the model.
#     if not os.path.isdir('./runs'):
#         os.mkdir('./runs')
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/run_'+str(auc)+"__"+str(args.n_estimators)+"_"+str(args.max_depth)+'.joblib')

if __name__ == '__main__':
    main()

# !ls