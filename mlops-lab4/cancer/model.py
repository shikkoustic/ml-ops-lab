from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

# T1: compare models with cross-validation, save whichever scores best
def train_and_save_model(X, y):
    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    scores = {}
    for name, model in candidates.items():
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        scores[name] = cv.mean()
        print(name, "CV accuracy", round(cv.mean(), 3))

    best_name = max(scores, key=scores.get)
    print("Best model:", best_name)

    best_model = candidates[best_name]
    best_model.fit(X, y)
    joblib.dump(best_model, "cancer_model.joblib")
    return best_model
