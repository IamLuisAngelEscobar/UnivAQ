from sklearn.ensemble import VotingClassifier
from auxiliary_functions import load_joblib


def Voting(X_train, y_train):
    path_to_read1 = "/Users/luisescobar/Documents/Thesis/Models/GPT_exp/hard_voting"

    # Define individual models with parameters tuned for imbalanced data
    xgb_sign = load_joblib(path_to_read1, "XGB")
    rf = load_joblib(path_to_read1, "RF")
    xgb_noise = load_joblib(path_to_read1, "LGBM")

    # Create the voting classifier with hard voting
    voting_clf = VotingClassifier(
        estimators=[("xgb", xgb_sign), ("rf", rf), ("lgbm", xgb_noise)],
        voting="hard",  # Majority voting based on predicted classes
    )

    # Train the voting classifier
    return voting_clf.fit(X_train, y_train)
