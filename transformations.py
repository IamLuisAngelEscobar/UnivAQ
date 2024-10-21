import joblib
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.preprocessing import StandardScaler

def normaliztion(X_train, X_test, path_to_save):
    # Standardize features using standardscaler
    scaler = StandardScaler()
    # Fit the scaler on the training data (ONLY fit on training data)
    scaler.fit(X_train)
    # Transform (standardize) the training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_name = f'{path_to_save}/scaler.pkl'
    joblib.dump(scaler, scaler_name)    
    
    return X_train_scaled, X_test_scaled



def unbalanced(method, sampling):
    match method:
        case 'smote':
            return SMOTE(sampling_strategy=sampling, random_state=42)
        case 'border-line-smote':
            return BorderlineSMOTE(sampling_strategy=sampling, random_state=42)
        case 'svmsmote':
            return SVMSMOTE(sampling_strategy=sampling, random_state=42)
        case 'adasyn':
            return ADASYN(sampling_strategy=sampling, random_state=42)
        case _:
            return None  # Optional: handle unexpected cases
        