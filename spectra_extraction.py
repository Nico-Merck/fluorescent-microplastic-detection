import numpy as np
from scipy.signal import find_peaks, peak_prominences

from pathlib import Path

top_n = 30              # Number of top peaks to find in spectra sums
prom_sum_peaks = 0.1    # Minimum prominence of peaks in spectra sums

data_path = Path.cwd() / "data/spectral_data"

for data_sel in range(1,4):
    data_dict = np.load(data_path / f"data_dict_{data_sel}.npy", allow_pickle=True).item()
    spectra_dict_path = data_path / f"spectra_peaks_dict_{data_sel}.npy"

    spectra_dict = {}  # Dictionary to hold spectra
    print("--------------------------------")
    print(f"Processing data set {data_sel}:")

    for sample, spectra in data_dict.items():
        # Get spectra and calculate sum of spectra
        spectra_sums = np.array([spectrum.sum() for spectrum in spectra])
        normed_spectra_sums = spectra_sums / np.max(spectra_sums)
        
        # Find peaks in summed spectra
        sum_peaks, properties = find_peaks(normed_spectra_sums, prominence = prom_sum_peaks)
        prominences = peak_prominences(normed_spectra_sums, sum_peaks)[0]

        # Sort peaks by prominence and select top_n peaks
        sorted_indices = np.argsort(prominences)[::-1]
        sorted_peaks = sum_peaks[sorted_indices]
        sorted_prominences = prominences[sorted_indices]

        # Select top_n peaks
        spectra_peaks = spectra[sorted_peaks[:top_n]]

        # Add spectra behind peaks to dictionary
        for j in range(len(spectra_peaks)):
            spectra_dict[f"{sample}_{str(j+1).zfill(3)}"] = spectra_peaks[j]

        print(f"Processed {sample}: {len(spectra)} spectra, {len(spectra_peaks)} peaks found")

    # Save the dictionary to a pickle file
    if spectra_dict_path.exists():
        # Load the existing dictionary
        spectra_dict_save = np.load(spectra_dict_path, allow_pickle=True).item()
        # Update the dictionary with new values
        spectra_dict_save.update(spectra_dict)
    else:
        # Initialize the dictionary
        spectra_dict_save = spectra_dict

    # Sort the dictionary by the last 3 characters of the keys
    spectra_dict_save = dict(sorted(spectra_dict_save.items(), key=lambda item: item[0][-3:]))

    # Save the dictionaries
    np.save(spectra_dict_path, spectra_dict_save, allow_pickle=True)
print("--------------------------------")