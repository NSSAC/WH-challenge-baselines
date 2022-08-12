import os

from argparse import ArgumentParser
from glob import glob

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

def make_eval_data(preds, eval_labels, pop):
    """
    May not have a prediction for all relevant individuals,
    so construct a dataframe with all persons in population
    """
    eval_df = pd.DataFrame({"pred" : None, "label" : None}, index=pop.index)
    eval_df.loc[eval_labels.index, "label"] = eval_labels["infected"]
    eval_df.loc[preds.index, "pred"] = preds
    
    eval_df.fillna(0.0, inplace=True)
    
    return eval_df

def dedup_eval_probs(eval_features, probs):
    """
    since for any person, we may have multiple overlapping predictions
    e.g. for the windows (t, t+D], (t+1, t+D+1],..., all of which may
    overlap with the evaluation time period, deduplicate by finding the 
    maximum probability of a positive classification
    """
    prob_df = pd.DataFrame({"pred" : probs, "pid" : eval_features["pid"]})
    return prob_df.groupby("pid").max()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--training_dir", help="directory with training instances")
    parser.add_argument("--eval_dir", help="directory with evaluation instances")
    parser.add_argument("--pop", help="population csv") 
    parser.add_argument("--eval-labels", help="file with evaluation labels")
    args = parser.parse_args()
 
    train_files = glob(os.path.join(args.training_dir, "train_*.csv")) 
    eval_files = glob(os.path.join(args.eval_dir, "eval_*.csv"))

    train_data = pd.concat((pd.read_csv(f, index_col=0) for f in train_files))
    eval_data = pd.concat((pd.read_csv(f, index_col=0) for f in eval_files))

    train_features = train_data.drop(["pid", "s_t+d"], axis=1)
    train_labels = train_data["s_t+d"]

    pop = pd.read_csv(args.pop)
    pop.set_index("pid", inplace=True)

    eval_labels = pd.read_csv(args.eval_labels)
    eval_labels.set_index("pid", inplace=True)

    X_samp, y = SMOTE().fit_resample(train_features, train_labels)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_samp)

    clf = LogisticRegression(max_iter=200).fit(X, y)

    eval_features = eval_data.drop(["pid", "s_t+d"], axis=1)
    X_eval = scaler.transform(eval_features)

    eval_probs = clf.predict_proba(X_eval)[:,1]
    eval_dedup = dedup_eval_probs(eval_data, eval_probs) 
    
    joined_preds = make_eval_data(eval_dedup["pred"], eval_labels, pop)
    clf_auprc = average_precision_score(joined_preds["label"], joined_preds["pred"])

    unif_prob = train_labels.mean()*np.ones(joined_preds["label"].shape)
    unif_auprc = average_precision_score(joined_preds["label"], unif_prob)

    rand_prob = np.random.uniform(size=joined_preds["label"].shape[0])
    rand_auprc = average_precision_score(joined_preds["label"], rand_prob)

    print(f"Logistic regression AUPRC {clf_auprc}")
    print(f"Uniform AUPRC {unif_auprc}, Random AUPRC {rand_auprc}")
