def get_features_and_target(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y
