from delivery import load_data, get_features_and_target, train_and_save_model

df = load_data()
X, y = get_features_and_target(df)
model = train_and_save_model(X, y)
print("Training done")
