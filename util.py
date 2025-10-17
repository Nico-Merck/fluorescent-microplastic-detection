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

from typing import Optional, Tuple, Dict

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

def intensity_ratio_transform(X, I0, I1):
    X = np.asarray(X)
    R1 = X[:, I1] / X[:, I0]
    return R1

def df_from_dict(dict, normalize=True, classes="all"):
    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(dict, orient='index')
    # Set Multiindex with class and sample key
    df.index.name = 'sample_key'
    df.insert(0, 'class', df.index.map(lambda x: x.split('_')[0]))
    df = df.reset_index()
    df = df.set_index(['class', 'sample_key'])
    if classes != "all":
        df = df[df.index.get_level_values('class').isin(classes)]
    # Sort by class
    df = df.sort_index(level='class')
    if normalize:
        # Normalize spectra
        df = df.mul(df.shape[1]).div(df.sum(axis=1), axis=0)

    return df

def mean_and_std(data: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the mean and sample standard deviation of the given data.

    Args:
    data: Input data as a numpy array.

    Returns:
    A tuple containing the mean and sample standard deviation of the input data.
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    return mean, std

def fwhm_and_bounds(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    max_index: np.ndarray
    ) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculates the full width at half maximum (FWHM) and the left and right half maximum wavelengths (with interpolation).

    Args:
    wavelengths: Wavelengths corresponding to the spectrum.
    spectrum: The spectrum as a numpy array.
    max_index: Index of the maximum value in the spectrum.

    Returns:
    A tuple containing the FWHM, lambda left, lambda right including all values and mean/std.
    """
    fwhm = np.empty(spectrum.shape[0])
    left_half_max = np.empty(spectrum.shape[0])
    right_half_max = np.empty(spectrum.shape[0])

    for i in range(len(spectrum)):
        idx_max = max_index[i][0]
        half_max = spectrum[i][idx_max] / 2

        # Search for the left crossing (from maximum to lower indices)
        left = np.where(spectrum[i][:idx_max] < half_max)[0]
        if left.size == 0:
            left_wl = wavelengths[0]
        else:
            l1 = left[-1]
            l2 = l1 + 1
            # Linear interpolation between l1 and l2 for subpixel accuracy
            x1, x2 = wavelengths[l1], wavelengths[l2]
            y1, y2 = spectrum[i][l1], spectrum[i][l2]
            left_wl = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

        # Search for the right crossing (from maximum to higher indices)
        right = np.where(spectrum[i][idx_max:] < half_max)[0]
        if right.size == 0:
            right_wl = wavelengths[-1]
        else:
            r1 = idx_max + right[0] - 1
            r2 = r1 + 1
            if r2 >= len(wavelengths):
                right_wl = wavelengths[-1]
            else:
                x1, x2 = wavelengths[r1], wavelengths[r2]
                y1, y2 = spectrum[i][r1], spectrum[i][r2]
                right_wl = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

        fwhm[i] = right_wl - left_wl
        left_half_max[i] = left_wl
        right_half_max[i] = right_wl

    fwhm_mean, fwhm_std = mean_and_std(fwhm)
    left_half_max_mean, left_half_max_std = mean_and_std(left_half_max)
    right_half_max_mean, right_half_max_std = mean_and_std(right_half_max)

    return fwhm, left_half_max, right_half_max, fwhm_mean, fwhm_std, left_half_max_mean, left_half_max_std, right_half_max_mean, right_half_max_std