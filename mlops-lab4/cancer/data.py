from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_data():
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["target"] = bc.target        # 0 = malignant, 1 = benign
    return df
