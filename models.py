import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from auxiliary_functions import load_joblib
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score


"""# Define a custom scoring function for F1 with a threshold
def custom_f1_scorer(y_true, y_pred_proba):
    # Apply custom threshold (e.g., 0.6) to predicted probabilities
    y_pred = (y_pred_proba[:, 1] > 0.6).astype(int)
    return f1_score(y_true, y_pred)

# Use make_scorer to create a custom scorer for RandomizedSearchCV
f1_scorer = make_scorer(custom_f1_scorer, needs_proba=True)"""


def custom_f2_scorer(y_true, y_pred_proba):
    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)  # Apply threshold if needed
    return fbeta_score(y_true, y_pred, beta=2)  # F2 score, prioritizing recall


f2_scorer = make_scorer(custom_f2_scorer, needs_proba=True)


def XGBmodel(X_train, y_train):
    # Initialize XGBoost classifier
    # scale_ratio = 300
    # print(f"Scale ratio {scale_ratio}")
    xgb_model = XGBClassifier(
        random_state=42,
        objective="binary:logistic",
        # scale_pos_weight=scale_ratio,
        eval_metric="aucpr",
    )
    """param_distributions = {
        "n_estimators": randint(50, 200),  # Number of boosting rounds
        "max_depth": randint(3, 10),  # Maximum depth of each tree
        "learning_rate": uniform(0.01, 0.3),  # Learning rate (eta)
        "subsample": uniform(0.5, 0.5),  # Subsample ratio of training instances
        "colsample_bytree": uniform(0.5, 0.5),  # Subsample ratio of features
        "gamma": uniform(
            0, 5
        ),  # Minimum loss reduction required to make a further partition
        "min_child_weight": randint(
            1, 10
        ),  # Minimum sum of instance weight (hessian) needed in a child
        # "scale_pos_weight": randint(
        #    1, 200
        # ),  # Weight for the positive class (around 265)
    }"""

    param_distributions = {
        "scale_pos_weight": randint(1, 200),  # Vary this based on the imbalance ratio
        "max_delta_step": [0, 1, 2, 5],  # Small values to stabilize updates
        "min_child_weight": randint(1, 20),  # Control for overfitting
        "max_depth": randint(3, 10),  # Keep depth lower to avoid overfitting
        "subsample": uniform(0.6, 0.4),  # Range 0.6 - 1.0
        "colsample_bytree": uniform(0.6, 0.4),  # Range 0.6 - 1.0
        "learning_rate": uniform(0.01, 0.3),  # Range 0.01 - 0.3
        "n_estimators": randint(100, 500),  # Number of boosting rounds
        "alpha": uniform(0, 1),  # L1 regularization
        "lambda": uniform(0, 1),  # L2 regularization
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings to sample
        cv=3,  # 5-fold cross-validation
        # scoring="accuracy",
        # scoring="neg_log_loss",
        scoring="f1",
        random_state=42,
        verbose=1,
        n_jobs=-1,  # Use all processors
    )

    # Train XGBoost model
    random_search.fit(X_train, y_train)
    # Best parameters found
    print(f"Best parameters found: {random_search.best_params_}")
    # Evaluate the best model on the test set
    best_xgb_model = random_search.best_estimator_

    return best_xgb_model


def RFmodel(X_train, y_train):
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
    # Define the hyperparameter grid to search through
    param_distributions = {
        "n_estimators": randint(50, 200),  # Number of trees in the forest
        "max_depth": randint(5, 20),  # Maximum depth of each tree
        "min_samples_split": randint(
            2, 10
        ),  # Minimum number of samples to split a node
        "min_samples_leaf": randint(
            1, 10
        ),  # Minimum number of samples required to be at a leaf node
        "bootstrap": [True, False],
        "class_weight": ["balanced", None],  # Whether bootstrap samples are used
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_distributions,
        n_iter=10,  # Number of parameter settings to sample
        cv=StratifiedKFold(
            n_splits=5
        ),  # Use StratifiedKFold to preserve class distribution
        scoring="f1",
        random_state=42,
        n_jobs=-1,  # Use all processors
        verbose=1,
    )

    random_search.fit(X_train, y_train)
    # Best parameters and score found
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best score found: {random_search.best_score_:.4f}")

    # Evaluate the best model on the test set
    best_rf_model = random_search.best_estimator_

    return best_rf_model


def LGBMmodel(X_train, y_train):
    # Initialize LightGBM classifier
    lgbm_model = LGBMClassifier(random_state=42, class_weight="balanced")

    param_distributions = {
        "n_estimators": randint(50, 200),  # Number of boosting rounds
        "max_depth": randint(3, 15),  # Maximum depth of each tree
        "learning_rate": uniform(0.01, 0.3),  # Learning rate
        "num_leaves": randint(20, 150),  # Maximum number of leaves in one tree
        "subsample": uniform(0.5, 0.5),  # Subsample ratio of the training data
        "colsample_bytree": uniform(0.5, 0.5),  # Subsample ratio of columns (features)
        "min_child_samples": randint(
            10, 100
        ),  # Minimum number of data points in a leaf
        "reg_alpha": uniform(0, 1),  # L1 regularization term
        "reg_lambda": uniform(0, 1),  # L2 regularization term
        "scale_pos_weight": randint(1, 200),
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings to sample
        cv=3,  # 5-fold cross-validation
        scoring="f1",
        random_state=42,
        n_jobs=-1,  # Use all processors
        verbose=0,
    )

    # Train LightGBM model
    random_search.fit(X_train, y_train)

    # Best parameters found
    print(f"Best parameters found: {random_search.best_params_}")

    # Evaluate the best model on the test set
    best_lgbm_model = random_search.best_estimator_

    return best_lgbm_model


def Voting(X_train, y_train):
    path_to_read1 = "/Users/luisescobar/Documents/Thesis/Models/GPT_exp/hard_voting"
    path_to_read2 = "/Users/luisescobar/Documents/Thesis/Models/original/lag_10min"

    # Define individual models with parameters tuned for imbalanced data
    xgb_sign = load_joblib(path_to_read1, "XGB/xgboost_model.pkl")
    xgb_noise = load_joblib(path_to_read2, "/xgboost_model.pkl")
    lgbm_sign = load_joblib(path_to_read1, "LGBM/lgbm_model.pkl")
    lgbm_noise = load_joblib(path_to_read2, "/lightgbm_model.pkl")
    rf_sign = load_joblib(path_to_read1, "RF/rf_model.pkl")

    # Create the voting classifier with soft voting
    voting_clf = VotingClassifier(
        estimators=[
            ("xgb_sign", xgb_sign),
            ("xgb_noise", xgb_noise),
            ("lgbm_sign", lgbm_sign),
            ("lgbm_noise", lgbm_noise),
            ("rf_sign", rf_sign),
        ],
        voting="soft",  # Majority voting based on predicted classes
    )

    # Train the voting classifier
    return voting_clf.fit(X_train, y_train)


def Meta(X_train, y_train):
    path_to_read1 = "/Users/luisescobar/Documents/Thesis/Models/GPT_exp/hard_voting"
    path_to_read2 = "/Users/luisescobar/Documents/Thesis/Models/original/lag_10min"

    # Define base models
    xgb_sign = load_joblib(path_to_read1, "XGB/xgboost_model.pkl")
    xgb_noise = load_joblib(path_to_read2, "/xgboost_model.pkl")
    lgbm_sign = load_joblib(path_to_read1, "LGBM/lgbm_model.pkl")
    lgbm_noise = load_joblib(path_to_read2, "/lightgbm_model.pkl")
    rf_sign = load_joblib(path_to_read1, "RF/rf_model.pkl")

    # Define the base models
    base_models = [
        ("xgb_sign", xgb_sign),
        ("xgb_noise", xgb_noise),
        ("lgbm_sign", lgbm_sign),
        ("lgbm_noise", lgbm_noise),
        ("rf_sign", rf_sign),
    ]
    # Define the meta-model
    meta_model = LogisticRegression()

    # Define the stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,  # Cross-validation for generating meta-features
        n_jobs=-1,
    )

    # Train the stacking classifier
    return stacking_clf.fit(X_train, y_train)
