# To Do T1: every setting (features, target, split, cv folds, scoring,
# where to save) is read from cancer/config.yaml instead of being typed here
from pathlib import Path

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train_and_save_model(df, config=None, model_path=None):
    config = config or load_config()
    seed = config["split"]["seed"]

    X = df[config["data"]["features"]]
    y = df[config["data"]["target"]]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed),
    }

    scores = {}
    for name, model in candidates.items():
        cv = cross_val_score(
            model, X_tr, y_tr,
            cv=config["training"]["cv_folds"],
            scoring=config["training"]["scoring"],
        )
        scores[name] = cv.mean()
        print(name, "CV", config["training"]["scoring"], round(cv.mean(), 3))

    best_name = max(scores, key=scores.get)
    print("Best model:", best_name)

    best_model = candidates[best_name].fit(X_tr, y_tr)
    holdout = accuracy_score(y_te, best_model.predict(X_te))
    print("held-out accuracy:", round(holdout, 3))

    joblib.dump(best_model, model_path or config["output"]["model_path"])
    return best_model, holdout
