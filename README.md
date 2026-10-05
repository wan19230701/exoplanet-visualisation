# Visualising Exoplanet Diversity and Detection Biases

This project investigates the physical diversity of confirmed exoplanets and the observational selection effects that shape the NASA Exoplanet Archive catalogue.

## Research question

How do discovery methods shape the observed exoplanet population, and what patterns appear across planetary, orbital, and host-star properties?

## Data

The analysis uses a 6,287-row snapshot of the NASA Exoplanet Archive Planetary Systems Composite Parameters (`PSCompPars`) table. Each row represents one confirmed exoplanet. The included `sync.csv` contains discovery metadata, orbital and planetary properties, host-star properties, and Galactic coordinates.

Source: [NASA Exoplanet Archive: PSCompPars](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html).

## Methods

- Missing physical values are retained as missing rather than replaced with zero.
- Positive physical variables are visualised on logarithmic scales where appropriate.
- Discovery methods outside the seven most common categories are grouped as `Other`.
- Planet-radius classes are used only as visual categories, not compositional classifications.
- Pearson correlations and principal component analysis summarise multivariate structure.

## Results

The figures show that the confirmed-planet catalogue is strongly transit-dominated. Discovery method is associated with distinct orbital-period, radius, distance, and sky-coverage patterns, consistent with observational selection effects. The mass-radius-density, correlation, and PCA views also reveal broad physical and host-star variation in the observed sample.

These analyses describe the confirmed-planet catalogue; they do not estimate the true occurrence rate of planets or establish causal relationships.

## Repository contents

```text
exoplanet_visualisation_project.py  Analysis and figure-generation script
sync.csv                            NASA PSCompPars data snapshot
figures/                            13 generated static figures
```

## Reproduce the analysis

```bash
python -m pip install -r requirements.txt
python exoplanet_visualisation_project.py --data sync.csv --output exoplanet_figures
```

The script saves 13 PNG figures to `exoplanet_figures/`.

## Requirements

- Python 3.10+
- NumPy
- pandas
- Matplotlib

## Limitations

`PSCompPars` is a composite catalogue, so values for one planet can originate from different references. Missing data reduce the sample size for some figures. The repository includes confirmed planets only, and the visual size classes should not be interpreted as definitive physical compositions.
