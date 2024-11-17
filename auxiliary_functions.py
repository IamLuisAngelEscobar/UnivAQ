import pickle
import joblib
import os


def load_open(filepath, filename):
    name_to_read = f"{filepath}/{filename}"
    with open(name_to_read, "rb") as file:
        pkl_file = pickle.load(file)
    return pkl_file


def load_joblib(filepath, filename=None):
    if filename:
        name_to_read = f"{filepath}/{filename}"
    else:
        name_to_read = filepath
    pkl_file = joblib.load(name_to_read)
    return pkl_file


def write_joblib(file, filepath, filename):
    save_name = f"{filepath}/{filename}"
    joblib.dump(file, save_name)
