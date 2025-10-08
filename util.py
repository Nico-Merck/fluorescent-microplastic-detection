import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, silhouette_score, silhouette_samples, confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, labels, plot=False):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro"),
        "Recall (macro)": recall_score(y_test, pred, average="macro"),
        "F1 (macro)": f1_score(y_test, pred, average="macro"),
        "ROC-AUC (ovr, micro)": roc_auc_score(y_test, probs, multi_class="ovr", average="micro")
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, pred, labels=labels, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    if plot:
        disp.plot(cmap="Blues", colorbar=False)

    return metrics_df, pred, probs

def min_max_norm(X, wavelengths, laser_end, spectral_dim=1):
    X = np.asarray(X)
    mask = wavelengths > laser_end
    X_min = X.min(axis=spectral_dim, keepdims=True)
    X_max = X[:, mask].max(axis=spectral_dim, keepdims=True)
    return (X - X_min) / (X_max - X_min)

def intensity_ratio_transform(X, I0, I1, I2, I3):
    X = np.asarray(X)
    R1 = X[:, I1] / X[:, I0]
    R2 = X[:, I2] / X[:, I3]
    return np.column_stack((R1, R2))

def df_from_dict(dict, normalize=True):
    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(dict, orient='index')
    # Set Multiindex with class and sample key
    df.index.name = 'sample_key'
    df.insert(0, 'class', df.index.map(lambda x: x.split('_')[0]))
    df = df.reset_index()
    df = df.set_index(['class', 'sample_key'])
    # Sort by class
    df = df.sort_index(level='class')
    if normalize:
        # Normalize spectra
        df = df.mul(df.shape[1]).div(df.sum(axis=1), axis=0)

    return df