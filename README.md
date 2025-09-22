# A non-contact in-situ approach for detecting fluorescent microplastic particles in flowing water using fluorescence spectroscopy

This repository contains the code and data used for analysing fluorescence spectra of microplastic particles. The aim is to characterise different polymer samples and their fluorescent properties using spectroscopic methods.
This study demonstrates in-flow detection of polypropylene particles containing production-added fluorescent dyes: fluorescence spectroscopy enabled classification via intensity ratios and clustering, while interferometric particle imaging revealed particle size and type.

## Repository structure

```
.
├── data/                                            # Raw and processed spectral data
│   ├── *.csv                                        # Individual measurement series as CSV files
│   ├── spectra_dict.npy                             # NumPy dictionary containing all spectra
│   └── wavelengths.npy                              # NumPy array with wavelength bins
├── README.md                                        # Project description
└── spectroscopy_on_fluorescent_mp_particles.ipynb   # Jupyter Notebook with analysis workflow
```

## Contents

- **Data folder**  
  - `.csv` files: Spectra of all measurement series.  
  - `wavelengths.npy`: NumPy array containing the wavelength bins used for all spectra.  
  - `spectra_dict.npy`: Dictionary storing the spectra of all measurement series, accessible via NumPy.
 
- **Jupyter Notebook**  
  The main notebook (`spectroscopy_on_fluorescent_mp_particles.ipynb`) contains data loading, preprocessing, visualization, and analysis of the fluorescence spectra.  

## Requirements

To run the notebook, you need:

- Python 3.13.5+  
- Jupyter Notebook or JupyterLab  
- Packages:  
  ```bash
  numpy 2.2.6
  python 3.13.5
  pandas 1.5.3
  plotly 6.3.0
  scikit-learn 1.7.1
  scipy 1.16.1
  ```
  (see `environment.yml` for details)

Install dependencies with:
```bash
conda env create -f environment.yml
conda activate mp_venv
```

## Usage

1. Clone the repository:
   ```bash
   git clone [<repository-url>](https://github.com/Nico-Merck/fluorescent-microplastic-detection.git)
   cd [<repository-name>]
   ```

2. Install the environment as described above.  

3. Open the notebook:
   ```bash
   jupyter notebook spectroscopy_on_fluorescent_mp_particles.ipynb
   ```

4. Run all cells to reproduce the analysis and plots.  

## License

This project is licensed under the terms of the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for details.

