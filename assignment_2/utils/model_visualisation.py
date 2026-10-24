import os
import glob
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no screen in the Airflow container: draw to files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from utils.model_monitoring import PSI_THRESHOLDS, AUC_THRESHOLDS, GINI_THRESHOLDS


# colours: categorical series in fixed order (colour-blind safe), status colours only for thresholds
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
WARNING_COLOR = "#fab219"
CRITICAL_COLOR = "#d03b3b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


# ====================================================================== #
# Helpers
# ====================================================================== #


def _style_axis(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def _pad_x(ax, x):
    # keep the x-axis tight around the data months (also when there is only 1 month)
    ax.set_xlim(x.min() - pd.Timedelta(days=20), x.max() + pd.Timedelta(days=40))


def _threshold(ax, y, color, label):
    # dashed line carries the status colour; its label stays in text colour
    ax.axhline(y, color=color, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(label, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(4, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, color=TEXT_SECONDARY)


def _line(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)


def _label_last(ax, x, y, text_format="{:.2f}"):
    ax.annotate(text_format.format(y.iloc[-1]), xy=(x.iloc[-1], y.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)


def _not_available(ax, title):
    _style_axis(ax, title, "")
    ax.text(0.5, 0.5, "Not available yet", transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks([])
    ax.set_yticks([])


def _metric(df, metric_type, metric_name):
    """Time series of one metric, sorted by prediction month."""
    s = df[(df["metric_type"] == metric_type) & (df["metric_name"] == metric_name)]
    s = s.sort_values("prediction_month")
    return pd.to_datetime(s["prediction_month"]), s["metric_value"].reset_index(drop=True)


# ====================================================================== #
# Charts (each draws on one matplotlib axis)
# ====================================================================== #


def plot_psi(ax, df):
    title = "Score stability (PSI vs training)"
    x, y = _metric(df, "stability", "psi_score")
    if len(y) == 0:
        return _not_available(ax, title)
    _style_axis(ax, title, "PSI")
    _threshold(ax, PSI_THRESHOLDS[0], WARNING_COLOR, f"{PSI_THRESHOLDS[0]} moderate drift")
    _threshold(ax, PSI_THRESHOLDS[1], CRITICAL_COLOR, f"{PSI_THRESHOLDS[1]} significant drift")
    _line(ax, x.reset_index(drop=True), y, SERIES_COLORS[0])
    _label_last(ax, x.reset_index(drop=True), y)
    _pad_x(ax, x)
    ax.set_ylim(bottom=0)


def plot_csi(ax, df):
    title = "Feature stability (CSI, top 5 features)"
    csi = df[(df["metric_type"] == "stability") & (df["metric_name"].str.startswith("csi_"))]
    if len(csi) == 0:
        return _not_available(ax, title)
    _style_axis(ax, title, "CSI (log scale)")
    ax.set_yscale("log")
    _threshold(ax, PSI_THRESHOLDS[0], WARNING_COLOR, f"{PSI_THRESHOLDS[0]} moderate drift")
    _threshold(ax, PSI_THRESHOLDS[1], CRITICAL_COLOR, f"{PSI_THRESHOLDS[1]} significant drift")
    for i, metric_name in enumerate(sorted(csi["metric_name"].unique())):
        x, y = _metric(df, "stability", metric_name)
        _line(ax, x.reset_index(drop=True), y.clip(lower=1e-3), SERIES_COLORS[i % len(SERIES_COLORS)],
              label=metric_name.replace("csi_", ""))
    _pad_x(ax, pd.to_datetime(csi["prediction_month"]))
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=TEXT_PRIMARY)


def plot_auc(ax, df):
    title = "Performance: AUC (labels arrive 6 months later)"
    x, y = _metric(df, "performance", "auc")
    if len(y) == 0:
        return _not_available(ax, title)
    _style_axis(ax, title, "AUC")
    _threshold(ax, AUC_THRESHOLDS[1], WARNING_COLOR, f"{AUC_THRESHOLDS[1]} warning")
    _threshold(ax, AUC_THRESHOLDS[0], CRITICAL_COLOR, f"{AUC_THRESHOLDS[0]} refresh")
    _line(ax, x.reset_index(drop=True), y, SERIES_COLORS[0])
    _label_last(ax, x.reset_index(drop=True), y)
    _pad_x(ax, x)
    ax.set_ylim(0.5, 1.0)


def plot_gini(ax, df):
    title = "Performance: Gini"
    x, y = _metric(df, "performance", "gini")
    if len(y) == 0:
        return _not_available(ax, title)
    _style_axis(ax, title, "Gini")
    _threshold(ax, GINI_THRESHOLDS[1], WARNING_COLOR, f"{GINI_THRESHOLDS[1]} warning")
    _threshold(ax, GINI_THRESHOLDS[0], CRITICAL_COLOR, f"{GINI_THRESHOLDS[0]} refresh")
    _line(ax, x.reset_index(drop=True), y, SERIES_COLORS[0])
    _label_last(ax, x.reset_index(drop=True), y)
    _pad_x(ax, x)
    ax.set_ylim(0.0, 1.0)


def plot_bad_rate(ax, df):
    title = "Calibration: predicted vs actual default rate"
    x_a, y_a = _metric(df, "performance", "bad_rate_actual")
    x_p, y_p = _metric(df, "performance", "bad_rate_predicted")
    if len(y_a) == 0:
        return _not_available(ax, title)
    _style_axis(ax, title, "Default rate")
    _line(ax, x_a.reset_index(drop=True), y_a, SERIES_COLORS[0], label="actual")
    _line(ax, x_p.reset_index(drop=True), y_p, SERIES_COLORS[1], label="predicted (mean PD)")
    _label_last(ax, x_a.reset_index(drop=True), y_a)
    _label_last(ax, x_p.reset_index(drop=True), y_p)
    _pad_x(ax, x_a)
    ax.set_ylim(0, max(y_a.max(), y_p.max()) * 1.3)
    ax.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=TEXT_PRIMARY)


# ====================================================================== #
# Main function
# ====================================================================== #


def plot_monitoring_charts(snapshot_date_str, model_name, gold_monitoring_directory, charts_directory):
    """
    Visualise model monitoring results over time (lecture 7: automated reporting).
    Reads the monitoring gold table up to snapshot_date and saves PNG charts:
        psi.png, csi.png, performance_auc_gini.png, bad_rate.png, monitoring_dashboard.png

    Args:
        snapshot_date_str: date string 'YYYY-MM-DD' (charts show results known up to this date)
        model_name: model artefact file name, e.g. 'credit_model_2024_06_01.pkl'
        gold_monitoring_directory: path to gold model monitoring
        charts_directory: path to save charts

    Returns:
        list of saved chart file paths, or None if no monitoring results yet
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    model_version = model_name.replace(".pkl", "")

    # --- load monitoring gold table (all partitions up to snapshot_date) ---
    files_list = sorted(glob.glob(os.path.join(gold_monitoring_directory, model_version, "*.parquet")))
    if not files_list:
        print(f"{snapshot_date_str} | no monitoring results yet, skipping")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files_list])
    df = df[pd.to_datetime(df["monitoring_date"]).dt.date <= snapshot_date]
    if len(df) == 0:
        print(f"{snapshot_date_str} | no monitoring results yet, skipping")
        return None

    output_directory = os.path.join(charts_directory, model_version) + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    saved = []

    def save(fig, file_name):
        filepath = output_directory + file_name
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        saved.append(filepath)

    # --- individual charts (for the deck) ---
    for file_name, plot_function in [("psi.png", plot_psi), ("csi.png", plot_csi), ("bad_rate.png", plot_bad_rate)]:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
        plot_function(ax, df)
        save(fig, file_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=SURFACE)
    plot_auc(axes[0], df)
    plot_gini(axes[1], df)
    save(fig, "performance_auc_gini.png")

    # --- dashboard: all key charts on one page ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=SURFACE)
    plot_psi(axes[0, 0], df)
    plot_csi(axes[0, 1], df)
    plot_auc(axes[1, 0], df)
    plot_bad_rate(axes[1, 1], df)
    fig.suptitle(f"Model monitoring: {model_version} (results known as of {snapshot_date_str})",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout()
    save(fig, "monitoring_dashboard.png")

    print(f"{snapshot_date_str} | saved charts to: {output_directory}")
    return saved
