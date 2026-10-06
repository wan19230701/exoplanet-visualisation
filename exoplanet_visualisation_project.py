"""Visualise exoplanet diversity and discovery bias from NASA PSCompPars data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


METHOD_COLOURS = {
    "Transit": "#0072B2", "Radial Velocity": "#D55E00", "Microlensing": "#009E73",
    "Imaging": "#CC79A7", "Transit Timing Variations": "#E69F00",
    "Pulsar Timing": "#56B4E9", "Astrometry": "#F0E442", "Other": "#666666",
}
SIZE_COLOURS = {
    "Terrestrial / rocky-size": "#8c564b", "Super-Earth / sub-Neptune": "#2ca02c",
    "Neptune-size": "#1f77b4", "Gas giant": "#ff7f0e", "Unknown radius": "#7f7f7f",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 360, "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 14, "axes.labelsize": 11.5, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.savefig(output_dir / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sample(data: pd.DataFrame, maximum: int = 6000) -> pd.DataFrame:
    return data if len(data) <= maximum else data.sample(maximum, random_state=3000)


def log_axes(ax: plt.Axes, x: bool = True, y: bool = True) -> None:
    if x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    if y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))


def load_data(path: Path) -> pd.DataFrame:
    required = [
        "pl_name", "hostname", "discoverymethod", "disc_year", "disc_facility", "pl_orbper",
        "pl_orbsmax", "pl_orbeccen", "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt", "pl_insol",
        "st_teff", "st_rad", "st_mass", "st_met", "st_lum", "sy_dist", "glat", "glon",
    ]
    data = pd.read_csv(path)
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    numeric = [column for column in required if column not in {"pl_name", "hostname", "discoverymethod", "disc_facility"}]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["pl_name", "discoverymethod", "disc_year"]).copy()
    data["disc_year"] = data["disc_year"].astype(int)

    positive = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt", "pl_insol", "st_teff", "st_rad", "st_mass", "sy_dist"]
    data.loc[:, positive] = data[positive].where(data[positive] > 0)
    top_methods = data["discoverymethod"].value_counts().head(7).index
    data["method"] = data["discoverymethod"].where(data["discoverymethod"].isin(top_methods), "Other")
    data["size_class"] = pd.cut(
        data["pl_rade"], [-np.inf, 1.6, 4, 8, np.inf],
        labels=["Terrestrial / rocky-size", "Super-Earth / sub-Neptune", "Neptune-size", "Gas giant"],
    ).astype(object).fillna("Unknown radius")
    for column in positive:
        data[f"log_{column}"] = np.log10(data[column])
    data["glon_rad"] = np.radians(-((data["glon"] + 180) % 360 - 180))
    data["glat_rad"] = np.radians(data["glat"])
    return data


def plot_completeness(data: pd.DataFrame, out: Path) -> None:
    columns = ["discoverymethod", "disc_year", "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt", "pl_insol", "st_teff", "st_rad", "st_mass", "st_met", "st_lum", "sy_dist", "glat", "glon"]
    values = data[columns].notna().mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 5.9))
    ax.barh(values.index, values, color="#4C78A8")
    ax.set(xlim=(0, 100), xlabel="Available values (%)", title="Data completeness of selected PSCompPars variables")
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=8.8)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig01_data_completeness.png")


def plot_discovery_timeline(data: pd.DataFrame, out: Path) -> None:
    yearly = data.groupby(["disc_year", "method"]).size().unstack(fill_value=0).sort_index()
    methods = yearly.sum().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.stackplot(yearly.index, *(yearly[method] for method in methods), labels=methods, colors=[METHOD_COLOURS.get(method, METHOD_COLOURS["Other"]) for method in methods], alpha=0.92)
    ax.set(xlabel="Discovery year", ylabel="Number of confirmed exoplanets", title="Annual confirmed exoplanet discoveries by method")
    ax.legend(loc="upper left", ncol=2)
    save(fig, out, "fig02_discovery_timeline_by_method.png")


def plot_method_counts(data: pd.DataFrame, out: Path) -> None:
    counts = data["method"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.barh(counts.index, counts, color=[METHOD_COLOURS.get(method, METHOD_COLOURS["Other"]) for method in counts.index])
    ax.set(xlabel="Number of confirmed planets", ylabel="Discovery method", title="Dominant methods in the observed exoplanet sample")
    for i, value in enumerate(counts):
        ax.text(value + counts.max() * 0.01, i, str(value), va="center", fontsize=8.8)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig03_discovery_method_counts.png")


def plot_facilities(data: pd.DataFrame, out: Path) -> None:
    counts = data["disc_facility"].fillna("Unknown").value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.barh(counts.index, counts, color="#4C78A8")
    ax.set(xlabel="Number of confirmed planets", ylabel="Discovery facility", title="Top discovery facilities in the dataset")
    for i, value in enumerate(counts):
        ax.text(value + counts.max() * 0.01, i, str(value), va="center", fontsize=8.8)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig04_top_discovery_facilities.png")


def plot_period_radius(data: pd.DataFrame, out: Path) -> None:
    plot = sample(data.dropna(subset=["pl_orbper", "pl_rade", "method"]))
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for method, group in plot.groupby("method"):
        ax.scatter(group["pl_orbper"], group["pl_rade"], s=18, alpha=0.55, linewidths=0, color=METHOD_COLOURS.get(method, METHOD_COLOURS["Other"]), label=method)
    log_axes(ax)
    ax.set(xlabel="Orbital period, days (log scale)", ylabel="Planet radius, Earth radii (log scale)", title="Orbital period-radius space reveals discovery-method bias")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    save(fig, out, "fig05_orbital_period_vs_radius_by_method.png")


def plot_mass_radius(data: pd.DataFrame, out: Path) -> None:
    plot = sample(data.dropna(subset=["pl_bmasse", "pl_rade", "pl_dens"]))
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    points = ax.scatter(plot["pl_bmasse"], plot["pl_rade"], c=np.log10(plot["pl_dens"]), cmap="viridis", s=22, alpha=0.62, linewidths=0)
    log_axes(ax)
    ax.set(xlabel="Planet mass, Earth masses (log scale)", ylabel="Planet radius, Earth radii (log scale)", title="Mass-radius relation coloured by planet density")
    for radius, label in [(1.6, "1.6 R⊕"), (4, "4 R⊕"), (8, "8 R⊕")]:
        ax.axhline(radius, color="black", lw=0.8, ls=":", alpha=0.55)
        ax.text(ax.get_xlim()[0] * 1.2, radius * 1.04, label, fontsize=8, alpha=0.7)
    fig.colorbar(points, ax=ax, pad=0.015, label="log10 density (g/cm³)")
    save(fig, out, "fig06_mass_radius_density.png")


def plot_radiation(data: pd.DataFrame, out: Path) -> None:
    plot = sample(data.dropna(subset=["pl_insol", "pl_eqt", "size_class"]))
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.axvspan(0.25, 4, color="#b5d8b5", alpha=0.23, label="Approx. temperate-insolation band")
    for size, group in plot.groupby("size_class", observed=False):
        ax.scatter(group["pl_insol"], group["pl_eqt"], s=18, alpha=0.58, linewidths=0, color=SIZE_COLOURS[size], label=size)
    ax.set_xscale("log")
    ax.set(xlabel="Insolation flux relative to Earth (log scale)", ylabel="Equilibrium temperature, K", title="Radiation environment of confirmed exoplanets")
    ax.legend(loc="upper left")
    save(fig, out, "fig07_insolation_vs_equilibrium_temperature.png")


def plot_hr(data: pd.DataFrame, out: Path) -> None:
    plot = sample(data.dropna(subset=["st_teff", "st_lum", "method"]).copy())
    plot["luminosity"] = 10 ** plot["st_lum"]
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for method, group in plot.groupby("method"):
        ax.scatter(group["st_teff"], group["luminosity"], s=18, alpha=0.52, linewidths=0, color=METHOD_COLOURS.get(method, METHOD_COLOURS["Other"]), label=method)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="Host-star effective temperature, K", ylabel="Host-star luminosity, L/Lsun (log scale)", title="Host-star distribution in an HR-diagram style view")
    ax.legend(loc="lower left", ncol=2)
    save(fig, out, "fig08_host_star_hr_diagram.png")


def plot_sky(data: pd.DataFrame, out: Path) -> None:
    plot = sample(data.dropna(subset=["glon_rad", "glat_rad", "method"]))
    fig = plt.figure(figsize=(8.9, 4.9))
    ax = fig.add_subplot(projection="mollweide")
    for method, group in plot.groupby("method"):
        ax.scatter(group["glon_rad"], group["glat_rad"], s=11, alpha=0.5, linewidths=0, color=METHOD_COLOURS.get(method, METHOD_COLOURS["Other"]), label=method)
    ax.grid(True, alpha=0.35)
    ax.set_title("Galactic sky distribution of known exoplanet host systems", pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    save(fig, out, "fig09_galactic_sky_map.png")


def plot_distance(data: pd.DataFrame, out: Path) -> None:
    plot = data.dropna(subset=["sy_dist", "method"])
    methods = plot["method"].value_counts().index
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    boxes = ax.boxplot([plot.loc[plot["method"] == method, "sy_dist"] for method in methods], tick_labels=methods, vert=False, patch_artist=True, showfliers=False)
    for box, method in zip(boxes["boxes"], methods):
        box.set_facecolor(METHOD_COLOURS.get(method, METHOD_COLOURS["Other"]))
        box.set_alpha(0.72)
    ax.set_xscale("log")
    ax.set(xlabel="System distance, parsecs (log scale)", ylabel="Discovery method", title="Distance selection effects differ across discovery methods")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig10_distance_distribution_by_method.png")


def plot_correlation(data: pd.DataFrame, out: Path) -> None:
    variables = {
        "log Orbital period": "log_pl_orbper", "log Semi-major axis": "log_pl_orbsmax", "Eccentricity": "pl_orbeccen", "log Radius": "log_pl_rade", "log Mass": "log_pl_bmasse", "log Density": "log_pl_dens", "log Temp.": "log_pl_eqt", "log Insolation": "log_pl_insol", "log Stellar Teff": "log_st_teff", "log Stellar radius": "log_st_rad", "log Stellar mass": "log_st_mass", "Stellar metallicity": "st_met", "Stellar log luminosity": "st_lum", "log Distance": "log_sy_dist",
    }
    corr = data[list(variables.values())].rename(columns={value: name for name, value in variables.items()}).corr(min_periods=150)
    fig, ax = plt.subplots(figsize=(8.5, 7.4))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(xticks=range(len(corr)), yticks=range(len(corr)), xticklabels=corr.columns, yticklabels=corr.index, title="Correlation structure of planetary, orbital and stellar variables")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.8, color="white" if abs(value) > 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035, label="Pearson correlation coefficient")
    save(fig, out, "fig11_correlation_heatmap.png")


def plot_pca(data: pd.DataFrame, out: Path) -> None:
    variables = {
        "log(P)": "log_pl_orbper", "log(a)": "log_pl_orbsmax", "e": "pl_orbeccen", "log(Rp)": "log_pl_rade", "log(Mp)": "log_pl_bmasse", "log(density)": "log_pl_dens", "log(Teq)": "log_pl_eqt", "log(S)": "log_pl_insol", "log(T*)": "log_st_teff", "log(R*)": "log_st_rad", "log(M*)": "log_st_mass", "[Fe/H]": "st_met", "log(L*)": "st_lum", "log(d)": "log_sy_dist",
    }
    pca = data[list(variables.values()) + ["size_class"]].dropna().copy()
    features = list(variables)
    pca = pca.rename(columns={value: name for name, value in variables.items()})
    matrix = pca[features].to_numpy(float)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    scores, loadings = left[:, :2] * singular[:2], right[:2].T
    explained = singular**2 / (singular**2).sum()

    score_data = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "size_class": pca["size_class"].to_numpy()})
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for size, group in sample(score_data, 4000).groupby("size_class"):
        ax.scatter(group["PC1"], group["PC2"], s=14, alpha=0.38, linewidths=0, color=SIZE_COLOURS[size], label=size)
    ax.axhline(0, color="black", lw=0.8, alpha=0.35); ax.axvline(0, color="black", lw=0.8, alpha=0.35)
    ax.set(xlabel=f"PC1 ({explained[0]:.1%} variance explained)", ylabel=f"PC2 ({explained[1]:.1%} variance explained)", title="PCA scores of exoplanet, orbit and host-star variables")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    save(fig, out, "fig12_pca_scores.png")

    strength = np.hypot(loadings[:, 0], loadings[:, 1])
    shown = np.argsort(strength)[-8:]
    fig, ax = plt.subplots(figsize=(6.7, 6.3))
    angle = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(angle), np.sin(angle), color="grey", lw=1, ls="--", alpha=0.45)
    label_offsets = {
        "log(P)": (22, 10), "log(a)": (-44, 14), "log(Teq)": (-34, -22),
        "log(S)": (34, -22), "log(L*)": (-52, 18), "log(T*)": (-56, 2),
        "log(R*)": (-52, -14), "log(Mp)": (-22, 18), "log(Rp)": (-44, -12),
        "log(density)": (18, -18), "log(d)": (22, 16), "e": (0, 18),
    }
    for index in shown:
        x, y = loadings[index]
        ax.arrow(0, 0, x, y, color="black", alpha=0.76, head_width=0.025, head_length=0.035, length_includes_head=True)
        label = features[index]
        offset = label_offsets.get(label, (12 if x >= 0 else -12, 12 if y >= 0 else -12))
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha="left" if offset[0] >= 0 else "right", va="bottom" if offset[1] >= 0 else "top", fontsize=8.8, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.86}, arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.6, "alpha": 0.65})
    ax.axhline(0, color="black", lw=0.8, alpha=0.35); ax.axvline(0, color="black", lw=0.8, alpha=0.35)
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal", xlabel=f"PC1 loading ({explained[0]:.1%} variance explained)", ylabel=f"PC2 loading ({explained[1]:.1%} variance explained)", title="PCA variable loadings")
    save(fig, out, "fig13_pca_loadings.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("sync.csv"), help="Path to sync.csv")
    parser.add_argument("--output", type=Path, default=Path("exoplanet_figures"), help="Directory for PNG figures")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    for plot in (plot_completeness, plot_discovery_timeline, plot_method_counts, plot_facilities, plot_period_radius, plot_mass_radius, plot_radiation, plot_hr, plot_sky, plot_distance, plot_correlation, plot_pca):
        plot(data, args.output)
    print(f"Saved 13 figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
