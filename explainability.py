import shap
import random
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from pandas.plotting import table
from predictions import make_prediction


def time_series_plot(trt, clot_dict, feature, filter=None):
    """_summary_
    Plots the values of a specific feature for a given treatment

    Args:
        trt (string): Treatment to plot
        clot_dict (dict): Dictionary where the test treatments are stored
        feature (string): Corresponding to the feature we want to plot
        filter (int, optional): In case we want to consider just positive or negative cases
    """
    df_time = clot_dict[trt]
    if filter == 1:
        df_time = df_time[df_time["Clotting_2"] == 1]
    elif filter == 0:
        df_time = df_time[df_time["Clotting_2"] == 0]
    fig, ax = plt.subplots(figsize=(19, 6))
    df_time_tmp = df_time[feature].to_frame()
    df_time_tmp.plot(
        xlabel="time", ylabel="value", title=f"{trt} {feature} TimeSeries", ax=ax
    )


def generate_table(desc):
    """Helper function to create a table figure from a DataFrame."""
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    tabla = table(ax, desc, loc="upper right", colWidths=[0.17] * len(desc.columns))
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.2)


def time_series_desc(trt, clot_dict, export=0, cols=None, filter=None):
    """
    Computes statistical metrics over the features of a specific treatment.

    Args:
        trt (str): Treatment to describe.
        clot_dict (dict): Dictionary where the test treatments are stored.
        export (int): Determines output type: 0 returns DataFrame, 1 returns plot.
        cols (list, optional): Columns to consider for description. Defaults to None.
        filter (int, optional): In case we want to consider just positive or negative cases

    Returns:
        DataFrame or matplotlib figure: Descriptive stats DataFrame or a table plot.
    """
    df_time = clot_dict.get(trt)

    if filter == 1:
        df_time = df_time[df_time["Clotting_2"] == 1]
    elif filter == 0:
        df_time = df_time[df_time["Clotting_2"] == 0]

    if df_time is None:
        raise ValueError(f"Treatment '{trt}' not found in clot_dict.")

    # Select columns if specified, otherwise use all columns
    desc_df = df_time[cols] if cols else df_time
    desc = desc_df.describe().round(6)

    if export == 0:
        return desc
    elif export == 1:
        return generate_table(desc)
    else:
        raise ValueError("Invalid export option. Use 0 for DataFrame or 1 for plot.")


def box_plot(trt, clot_dict, feature, filter=None):
    """_summary_
    Plots the box plot of a gven feature for a specific treatment

    Args:
        trt (string): Treatment to plot
        clot_dict (dict): Dictionary where the test treatments are stored
        feature (string): Corresponding to the feature we want to plot
        filter (int, optional): In case we want to consider just positive or negative cases
    """
    df_time = clot_dict[trt]

    if filter == 1:
        df_time = df_time[df_time["Clotting_2"] == 1]
    elif filter == 0:
        df_time = df_time[df_time["Clotting_2"] == 0]

    fig, ax = plt.subplots(figsize=(15, 6))
    df_time[feature].plot(
        kind="box", vert=False, title=f"{trt} Distribution of {feature} Readings", ax=ax
    )


def shap_exp(params):
    trt = params["trt"]
    clot_dict = params["clot_dict"]
    lag = params["lag"]
    scaler_loaded = params["scaler_loaded"]
    model_loaded = params["model_loaded"]
    shap_method = params["shap_method"]
    feature = params.get("feature", None)

    df = clot_dict[trt]

    # Create lagged features for each column except the target column
    for column in df.columns:
        if column != "Clotting_2":  # Skip the target column
            lagged_column_name = f"{column}_lag_{lag}"
            if (
                lagged_column_name not in df.columns
            ):  # Check if lagged column already exists
                df[lagged_column_name] = df[column].shift(lag)

    # Remove rows with NaN values (due to shifting)
    df_lagged = df.dropna()
    df_lagged = df_lagged.reset_index(drop=True)

    # Prepare features (X) and target (y)
    # Drop original columns and only use lagged features
    lagged_columns = [col for col in df_lagged.columns if "lag_" in col]
    X = df_lagged[lagged_columns]
    X = scaler_loaded.transform(X)
    # y_true = df_lagged['Clotting_2']

    # Compute SHAP values
    # explainer = shap.TreeExplainer(model_loaded)
    explainer = shap.TreeExplainer(model_loaded, link="identity")
    shap_values = explainer(X)

    match shap_method:
        case "waterfall":
            plot = shap.plots.waterfall(shap_values[0])
        case "force":
            plot = shap.plots.force(shap_values[0])
        case "stack force":
            plot = shap.plots.force(shap_values[0:200])
        case "absolute mean":
            plot = shap.plots.bar(shap_values)
        case "beeswarm":
            plot = shap.plots.beeswarm(shap_values)
        case "dependence":
            plot = shap.plots.scatter(shap_values[:, feature])
    return plot


def shap_exp_overall(params):
    clot_dict = params["clot_dict"]
    no_clot_dict = params["no_clot_dict"]
    lag = params["lag"]
    scaler_loaded = params["scaler_loaded"]
    model_loaded = params["model_loaded"]
    shap_method = params["shap_method"]
    feature = params.get("feature", None)

    random.seed(42)
    if no_clot_dict == "None":
        loaded_dict = clot_dict
    else:
        loaded_dict = {**clot_dict, **no_clot_dict}

    items = list(loaded_dict.items())
    random.shuffle(items)
    loaded_dict = dict(items)

    dataframes_list = list(loaded_dict.values())
    combined_df = pd.concat(dataframes_list, ignore_index=False)
    df = combined_df.reset_index(drop=True)

    # Create lagged features for each column except the target column
    for column in df.columns:
        if column != "Clotting_2":  # Skip the target column
            lagged_column_name = f"{column}_lag_{lag}"
            if (
                lagged_column_name not in df.columns
            ):  # Check if lagged column already exists
                df[lagged_column_name] = df[column].shift(lag)

    # Remove rows with NaN values (due to shifting)
    df_lagged = df.dropna()
    df_lagged = df_lagged.reset_index(drop=True)

    # Prepare features (X) and target (y)
    # Drop original columns and only use lagged features
    lagged_columns = [col for col in df_lagged.columns if "lag_" in col]
    X = df_lagged[lagged_columns]
    X = scaler_loaded.transform(X)
    # y_true = df_lagged['Clotting_2']

    # Compute SHAP values
    # explainer = shap.TreeExplainer(model_loaded)
    explainer = shap.TreeExplainer(model_loaded, link="identity")
    shap_values = explainer(X)

    match shap_method:
        case "waterfall":
            plot = shap.plots.waterfall(shap_values[0])
        case "force":
            plot = shap.plots.force(shap_values[0])
        case "stack force":
            plot = shap.plots.force(shap_values[0:200])
        case "absolute mean":
            plot = shap.plots.bar(shap_values)
        case "beeswarm":
            plot = shap.plots.beeswarm(shap_values)
        case "dependence":
            plot = shap.plots.scatter(shap_values[:, feature])
    return plot


def corr_matrix(params):
    trt = params["trt"]
    clot_dict = params["clot_dict"]
    lag = params["lag"]
    scaler_loaded = params["scaler_loaded"]
    features = params.get("feature", None)
    df = clot_dict[trt]

    # Create lagged features for each column except the target column
    for column in df.columns:
        if column != "Clotting_2":  # Skip the target column
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    # Remove rows with NaN values (due to shifting)
    df_lagged = df.dropna()
    df_lagged = df_lagged.reset_index(drop=True)

    # Prepare features (X) and target (y)
    # Drop original columns and only use lagged features
    lagged_columns = [col for col in df_lagged.columns if "lag_" in col]
    X = df_lagged[lagged_columns]
    X = scaler_loaded.transform(X)
    # y_true = df_lagged["Clotting_2"]

    corr_matrix = pd.DataFrame(X, columns=features).corr()

    return sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".1g")


def interactive_plot(params):
    trt = params["trt"]
    clot_dict = params["clot_dict"]
    lag = params["lag"]
    scaler_loaded = params["scaler_loaded"]
    model_loaded = params["model_loaded"]
    columns = params["columns"]
    pred_values = params["pred_values"]
    scaler = params.get("scaler", None)

    y_true, y_pred, y_pred_reg, X = make_prediction(
        clot_dict, trt, lag, scaler_loaded, model_loaded, pred_values, scaler
    )
    df = pd.DataFrame(X, columns=columns)

    # Add the predictions based on pred_values and determine which prediction columns to include in `value_vars`
    value_vars = []
    if pred_values[0] == 1:
        df["y_true"] = y_true
        value_vars.append("y_true")
    if pred_values[1] == 1:
        df["y_pred"] = y_pred
        value_vars.append("y_pred")
    if pred_values[2] == 1:
        df["y_pred_reg"] = y_pred_reg
        value_vars.append("y_pred_reg")

    df = df.reset_index().rename(columns={"index": "Index"})

    # Melt only the prediction columns (y_true, y_pred) while preserving the original feature columns
    df_long = df.melt(
        id_vars=["Index"] + columns,
        value_vars=value_vars,
        var_name="Prediction Type",
        value_name="Value",
    )

    # Create the line plot with Plotly Express
    fig = px.line(
        df_long,
        x="Index",  # Use the index as the x-axis
        y="Value",
        color="Prediction Type",  # Differentiate y_true and y_pred
        title=f"True vs. Predicted Values {trt}",
        hover_data={
            **{col: True for col in columns},
            "Index": True,
            "Prediction Type": False,
        },  # Include all feature columns in hover data
    )

    # Show plot
    return fig.show()
