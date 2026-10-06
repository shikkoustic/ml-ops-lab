# T4: report confusion matrix + classification report for the saved model
import joblib
from sklearn.metrics import confusion_matrix, classification_report

def report_performance(X, y):
    model = joblib.load("cancer_model.joblib")
    preds = model.predict(X)
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))
    print()
    print("Classification report:")
    print(classification_report(y, preds, target_names=["malignant", "benign"]))
