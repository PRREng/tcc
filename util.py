import pandas as pd
import numpy as np
import glob


def reads_and_format_df(filename):
    """
    Function that reads the csv and drops unnecessary columns and converts
    necessary objects to numbers, and drops time too.
    """
    df = pd.read_csv(filename, sep=";")
    df = df.drop(
        [
            "Result Number",
            "Sweep Number",
            "Point Number",
            "AC Level (V)",
            "DC Level (V)",
            "Set Point ('C)",
            "Temperature ('C)",
            "Control ('C)",
            "Unnamed: 14",
            "Time",
        ],
        axis=1,
    )
    columns = df.columns
    for col in columns:
        df[col] = df[col].str.replace(",", ".")
        df[col] = pd.to_numeric(df[col])
    return df


def normalizer(df: pd.DataFrame):
    """
    Function that normalizes the range of the column values, in order to be
    easier for inference and training
    """
    for col in df.columns:
        mean = np.mean(df[col])
        std_dev = np.std(df[col])
        df[col] = (df[col] - mean) / std_dev
    return df


def load_data():
    csv_sfo1000_files = glob.glob("data/SFO-1000/*.csv")
    csv_sfo1200_files = glob.glob("data/SFO-1200/*.csv")
    y1000 = np.array([0.0 for _ in range(len(csv_sfo1000_files))], dtype=np.float32)
    y1200 = np.array([1.0 for _ in range(len(csv_sfo1200_files))], dtype=np.float32)

    y = np.concatenate((y1000, y1200), axis=0, dtype=np.float32)

    X = np.zeros(
        (len(csv_sfo1000_files) + len(csv_sfo1200_files), 71, 5), dtype=np.float32
    )
    i = 0
    csvs = csv_sfo1000_files + csv_sfo1200_files

    for csv_file in csvs:
        df = reads_and_format_df(csv_file)
        df = normalizer(df)
        np_array = df.to_numpy(dtype=np.float32)
        X[i] = np_array
        i += 1

    X[22, 70, 4] = X[22, 69, 4]

    return X, y

