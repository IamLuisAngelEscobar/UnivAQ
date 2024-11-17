from evaluation_metrics import oscillation_red


def make_prediction(
    clot_dict, trt, lag, scaler_loaded, model_loaded, pred_values, scaler
):
    df = clot_dict[trt]

    # Create lagged features for each column except the target column
    for column in df.columns:
        if column != "Clotting_2":  # Skip the target column
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
            # df[f'{column}_lag_{lag}_lag_{lag}'] = df[column].shift(lag)

    # Remove rows with NaN values (due to shifting)
    df_lagged = df.dropna()
    df_lagged = df_lagged.reset_index(drop=True)

    # Prepare features (X) and target (y)
    # Drop original columns and only use lagged features
    lagged_columns = [col for col in df_lagged.columns if "lag_" in col]
    X = df_lagged[lagged_columns]
    X_scaler = scaler_loaded.transform(X)
    y_true = df_lagged["Clotting_2"]

    y_pred = model_loaded.predict(X_scaler)

    # Regularization on oscillations (on test)
    if pred_values[2] == 1:
        y_pred_reg = oscillation_red(y_pred, 0.1, 20)
    elif pred_values[2] == 0:
        y_pred_reg = y_pred

    # Now I need to create a single DataFrame including all the information
    if scaler == "None":
        X = scaler_loaded.inverse_transform(X)
    elif scaler == "Yes":
        X = X
    return y_true, y_pred, y_pred_reg, X
