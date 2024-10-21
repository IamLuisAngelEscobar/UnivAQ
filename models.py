from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def XGBmodel(X_train, y_train):
    # Initialize XGBoost classifier
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    param_distributions = {
        'n_estimators': randint(50, 200),           # Number of boosting rounds
        'max_depth': randint(3, 10),                # Maximum depth of each tree
        'learning_rate': uniform(0.01, 0.3),        # Learning rate (eta)
        'subsample': uniform(0.5, 0.5),             # Subsample ratio of training instances
        'colsample_bytree': uniform(0.5, 0.5),      # Subsample ratio of features
        'gamma': uniform(0, 5),                     # Minimum loss reduction required to make a further partition
        'min_child_weight': randint(1, 10),         # Minimum sum of instance weight (hessian) needed in a child
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=10,  # Number of parameter settings to sample
        cv=5,       # 5-fold cross-validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1   # Use all processors
    )

    # Train XGBoost model
    random_search.fit(X_train, y_train)
    # Best parameters found
    print(f'Best parameters found: {random_search.best_params_}')
    # Evaluate the best model on the test set
    best_xgb_model = random_search.best_estimator_

    return best_xgb_model


def RFmodel(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    # Define the hyperparameter grid to search through
    param_distributions = {
        'n_estimators': randint(50, 200),  # Number of trees in the forest
        'max_depth': randint(5, 20),       # Maximum depth of each tree
        'min_samples_split': randint(2, 10), # Minimum number of samples to split a node
        'min_samples_leaf': randint(1, 10),  # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False],        # Whether bootstrap samples are used
    }

        # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator = rf_model,
        param_distributions = param_distributions,
        n_iter = 10,  # Number of parameter settings to sample
        cv = 5,       # 5-fold cross-validation
        scoring = 'recall',
        random_state = 42,
        n_jobs = -1,   # Use all processors
        verbose = 1
    )
    #Drastically slower
    '''# Manually iterating through n_iter to display progress
    for _ in tqdm(range(random_search.n_iter)):
        random_search.fit(X_train_scaled, y_train)'''
    
    random_search.fit(X_train, y_train)
    # Best parameters and score found
    print(f'Best parameters found: {random_search.best_params_}')
    print(f'Best score found: {random_search.best_score_:.4f}')
    
    # Evaluate the best model on the test set
    best_rf_model = random_search.best_estimator_
    '''test_accuracy = best_rf_model.score(X_train, y_train)
    print(f'Test set accuracy: {test_accuracy:.4f}')'''
    
    '''# Feature importance
    importances = best_rf_model.feature_importances_
    feature_names = X_test.feature_names
    for name, importance in zip(feature_names, importances):
        print(f'Feature: {name}, Importance: {importance:.4f}')'''
    
    return best_rf_model


def LGBMmodel(X_train, y_train):
    # Initialize LightGBM classifier
    lgbm_model = LGBMClassifier(random_state=42)

    param_distributions = {
        'n_estimators': randint(50, 200),           # Number of boosting rounds
        'max_depth': randint(3, 15),                # Maximum depth of each tree
        'learning_rate': uniform(0.01, 0.3),        # Learning rate
        'num_leaves': randint(20, 150),             # Maximum number of leaves in one tree
        'subsample': uniform(0.5, 0.5),             # Subsample ratio of the training data
        'colsample_bytree': uniform(0.5, 0.5),      # Subsample ratio of columns (features)
        'min_child_samples': randint(10, 100),      # Minimum number of data points in a leaf
        'reg_alpha': uniform(0, 1),                 # L1 regularization term
        'reg_lambda': uniform(0, 1)                 # L2 regularization term
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=param_distributions,
        n_iter=10,  # Number of parameter settings to sample
        cv=5,       # 5-fold cross-validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,   # Use all processors
        verbose=-0
    )

    
    # Train LightGBM model
    random_search.fit(X_train, y_train)

    # Best parameters found
    print(f'Best parameters found: {random_search.best_params_}')
    
    # Evaluate the best model on the test set
    best_lgbm_model = random_search.best_estimator_
    
    return best_lgbm_model