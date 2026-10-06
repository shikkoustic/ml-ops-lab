# To Do T1: feature list now comes from config, not from a hardcoded drop()
def get_features_and_target(df, config):
    X = df[config["data"]["features"]]
    y = df[config["data"]["target"]]
    return X, y
