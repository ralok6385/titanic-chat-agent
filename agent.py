"""
agent.py — LangChain-style agent for Titanic dataset analysis.

This module provides the core intelligence of the chatbot.
It interprets natural-language questions, performs Pandas analysis
on the Titanic dataset, generates Matplotlib charts when needed,
and returns structured responses.

No OpenAI API key is required — the agent uses deterministic,
rule-based intent matching compatible with a LangChain tool-use
pattern.
"""

import os
import re
import uuid

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns

from langchain_core.tools import tool

# ──────────────────────────────────────────────
# 1.  LOAD DATASET (once, at module import)
# ──────────────────────────────────────────────
df: pd.DataFrame = sns.load_dataset("titanic")

# Directory where generated charts are saved
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 2.  MATPLOTLIB STYLE — premium dark theme
# ──────────────────────────────────────────────
PALETTE = {
    "primary":     "#6C63FF",
    "primary_l":   "#8B83FF",
    "secondary":   "#3A0CA3",
    "accent":      "#FF6B9D",
    "accent_l":    "#FF8FB6",
    "teal":        "#2EC4B6",
    "gold":        "#FFD166",
    "bg":          "#0F0F1A",
    "bg_light":    "#1A1A2E",
    "card":        "#22223A",
    "text":        "#EAEAFF",
    "text_sec":    "#9B9BB4",
    "grid":        "#FFFFFF",
    "grid_alpha":  0.06,
}

CHART_COLORS = ["#6C63FF", "#FF6B9D", "#2EC4B6", "#FFD166", "#8B83FF", "#FF8FB6"]


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes, title: str,
                       xlabel: str = "", ylabel: str = "") -> None:
    """Apply a consistent premium dark theme to any chart."""
    fig.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg_light"])

    ax.set_title(title, fontsize=16, fontweight="bold", color=PALETTE["text"],
                 pad=18, loc="left")
    ax.set_xlabel(xlabel, fontsize=11, color=PALETTE["text_sec"], labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, color=PALETTE["text_sec"], labelpad=10)
    ax.tick_params(colors=PALETTE["text_sec"], labelsize=9)

    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=PALETTE["grid_alpha"])
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=2.0)


def _save_chart(fig: plt.Figure, prefix: str) -> str:
    """Save a Matplotlib figure and return the file path."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filepath


# ──────────────────────────────────────────────
# 3.  LANGCHAIN-COMPATIBLE TOOLS
# ──────────────────────────────────────────────

@tool
def male_percentage() -> dict:
    """Return the percentage of male passengers on the Titanic."""
    total = len(df)
    males = len(df[df["sex"] == "male"])
    females = total - males
    pct_male = round((males / total) * 100, 2)
    pct_female = round((females / total) * 100, 2)

    # Donut chart
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [males, females]
    colors = [PALETTE["primary"], PALETTE["accent"]]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=["Male", "Female"], autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.75,
        textprops={"fontsize": 12, "color": PALETTE["text"]},
        wedgeprops={"width": 0.4, "edgecolor": PALETTE["bg"], "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(13)

    ax.text(0, 0, f"{pct_male}%\nMale", ha="center", va="center",
            fontsize=18, fontweight="bold", color=PALETTE["primary"])
    fig.set_facecolor(PALETTE["bg"])
    ax.set_title("Gender Breakdown", fontsize=16, fontweight="bold",
                 color=PALETTE["text"], pad=18)
    chart_path = _save_chart(fig, "male_pct")

    return {
        "answer": (
            f"🧑‍🤝‍🧑 **Gender Breakdown**\n\n"
            f"• **Male**: {males} passengers (**{pct_male}%**)\n"
            f"• **Female**: {females} passengers (**{pct_female}%**)\n\n"
            f"Nearly two-thirds of the passengers were male."
        ),
        "chart": chart_path,
    }


@tool
def average_fare() -> dict:
    """Return the average ticket fare on the Titanic with class breakdown."""
    avg = round(df["fare"].mean(), 2)
    median = round(df["fare"].median(), 2)
    max_fare = round(df["fare"].max(), 2)
    min_fare = round(df["fare"].min(), 2)
    by_class = df.groupby("pclass")["fare"].mean().round(2)

    # Horizontal bar chart by class
    fig, ax = plt.subplots(figsize=(8, 4.5))
    classes = [f"Class {c}" for c in by_class.index]
    values = by_class.values
    colors = CHART_COLORS[:3]

    bars = ax.barh(classes, values, color=colors, height=0.5,
                   edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                f"${val:.2f}", va="center", fontsize=11,
                fontweight="bold", color=PALETTE["text"])

    _apply_dark_theme(fig, ax, "Average Fare by Passenger Class", xlabel="Fare ($)")
    ax.invert_yaxis()
    chart_path = _save_chart(fig, "avg_fare")

    return {
        "answer": (
            f"💰 **Ticket Fare Analysis**\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Average Fare | **${avg}** |\n"
            f"| Median Fare | ${median} |\n"
            f"| Cheapest | ${min_fare} |\n"
            f"| Most Expensive | ${max_fare} |\n\n"
            f"**By Class:**\n"
            + "\n".join(f"• Class {c}: **${v:.2f}**" for c, v in by_class.items())
        ),
        "chart": chart_path,
    }


@tool
def passengers_per_port() -> dict:
    """Return passenger count per embarkation port with a donut chart."""
    port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    counts = df["embarked"].value_counts()

    lines = []
    labels, sizes = [], []
    for code, count in counts.items():
        name = port_map.get(code, code)
        pct = round(count / counts.sum() * 100, 1)
        lines.append(f"• **{name}** ({code}): {count} passengers ({pct}%)")
        labels.append(f"{name}\n({code})")
        sizes.append(count)

    # Donut chart
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = CHART_COLORS[:len(sizes)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 10, "color": PALETTE["text"]},
        pctdistance=0.78,
        wedgeprops={"width": 0.45, "edgecolor": PALETTE["bg"], "linewidth": 2.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(11)

    ax.text(0, 0, f"{counts.sum()}\nTotal", ha="center", va="center",
            fontsize=16, fontweight="bold", color=PALETTE["text_sec"])

    fig.set_facecolor(PALETTE["bg"])
    ax.set_title("Embarkation Ports", fontsize=16, fontweight="bold",
                 color=PALETTE["text"], pad=18)
    chart_path = _save_chart(fig, "embarkation_donut")

    return {
        "answer": "⚓ **Passengers by Embarkation Port**\n\n" + "\n".join(lines),
        "chart": chart_path,
    }


@tool
def age_histogram() -> dict:
    """Generate a histogram of passenger ages."""
    ages = df["age"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # KDE-filled histogram with gradient
    n, bins, patches = ax.hist(
        ages, bins=35, color=PALETTE["primary"], edgecolor=PALETTE["bg"],
        linewidth=0.8, alpha=0.85,
    )
    # Apply gradient
    cm = plt.cm.cool
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    for c, p in zip(norm, patches):
        plt.setp(p, "facecolor", cm(c))

    # Mean & median lines
    ax.axvline(ages.mean(), color=PALETTE["accent"], linestyle="--", linewidth=2,
               label=f"Mean: {ages.mean():.1f} yrs", alpha=0.9)
    ax.axvline(ages.median(), color=PALETTE["gold"], linestyle="--", linewidth=2,
               label=f"Median: {ages.median():.1f} yrs", alpha=0.9)

    _apply_dark_theme(fig, ax, "Distribution of Passenger Ages",
                      xlabel="Age (years)", ylabel="Number of Passengers")
    ax.legend(fontsize=10, frameon=False, labelcolor=PALETTE["text_sec"], loc="upper right")
    chart_path = _save_chart(fig, "age_histogram")

    return {
        "answer": (
            f"📊 **Age Distribution of Titanic Passengers**\n\n"
            f"| Stat | Value |\n"
            f"|------|-------|\n"
            f"| Mean Age | **{ages.mean():.1f}** years |\n"
            f"| Median Age | **{ages.median():.1f}** years |\n"
            f"| Youngest | {ages.min():.0f} years |\n"
            f"| Oldest | {ages.max():.0f} years |\n"
            f"| Std Dev | {ages.std():.1f} years |\n\n"
            f"_{len(df) - len(ages)} passengers had missing age data._"
        ),
        "chart": chart_path,
    }


@tool
def survival_rate() -> dict:
    """Analyse the overall survival rate and survival by gender/class."""
    overall = round(df["survived"].mean() * 100, 2)
    by_sex = df.groupby("sex")["survived"].mean().round(4) * 100
    by_class = df.groupby("pclass")["survived"].mean().round(4) * 100

    # Multi-panel chart: overall, by gender, by class
    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [1, 1.2, 1.5]})

    # Panel 1: Overall donut
    ax1 = axes[0]
    ax1.pie([overall, 100 - overall],
            colors=[PALETTE["teal"], PALETTE["card"]],
            startangle=90,
            wedgeprops={"width": 0.35, "edgecolor": PALETTE["bg"], "linewidth": 2})
    ax1.text(0, 0, f"{overall}%", ha="center", va="center",
             fontsize=22, fontweight="bold", color=PALETTE["teal"])
    ax1.set_title("Overall", fontsize=13, fontweight="bold",
                  color=PALETTE["text"], pad=12)

    # Panel 2: By gender
    ax2 = axes[1]
    genders = [s.capitalize() for s in by_sex.index]
    g_vals = by_sex.values
    g_colors = [PALETTE["accent"], PALETTE["primary"]]
    bars2 = ax2.bar(genders, g_vals, color=g_colors, width=0.5,
                    edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars2, g_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", fontsize=12,
                 fontweight="bold", color=PALETTE["text"])
    ax2.set_ylim(0, 105)
    ax2.set_title("By Gender", fontsize=13, fontweight="bold",
                  color=PALETTE["text"], pad=12)
    ax2.set_facecolor(PALETTE["bg_light"])
    ax2.tick_params(colors=PALETTE["text_sec"])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=PALETTE["grid_alpha"])

    # Panel 3: By class
    ax3 = axes[2]
    classes = [f"Class {c}" for c in by_class.index]
    c_vals = by_class.values
    c_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["teal"]]
    bars3 = ax3.bar(classes, c_vals, color=c_colors, width=0.5,
                    edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars3, c_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", fontsize=12,
                 fontweight="bold", color=PALETTE["text"])
    ax3.set_ylim(0, 105)
    ax3.set_title("By Class", fontsize=13, fontweight="bold",
                  color=PALETTE["text"], pad=12)
    ax3.set_facecolor(PALETTE["bg_light"])
    ax3.tick_params(colors=PALETTE["text_sec"])
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=PALETTE["grid_alpha"])

    fig.set_facecolor(PALETTE["bg"])
    fig.suptitle("Titanic Survival Rates", fontsize=18, fontweight="bold",
                 color=PALETTE["text"], y=1.04)
    fig.tight_layout(pad=2.0)
    chart_path = _save_chart(fig, "survival_rate")

    lines = [
        f"🛟 **Survival Rate Analysis**\n",
        f"**Overall:** {overall}% survived\n",
        "**By Gender:**",
    ]
    for sex, rate in by_sex.items():
        emoji = "👩" if sex == "female" else "👨"
        lines.append(f"  {emoji} {sex.capitalize()}: **{rate:.1f}%**")
    lines.append("\n**By Ticket Class:**")
    for cls, rate in by_class.items():
        lines.append(f"  🎫 Class {cls}: **{rate:.1f}%**")

    return {"answer": "\n".join(lines), "chart": chart_path}


@tool
def survival_by_class_chart() -> dict:
    """Generate a survival-rate-by-class bar chart."""
    by_class = df.groupby("pclass")["survived"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5.5))
    classes = [f"Class {c}" for c in by_class.index]
    colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["teal"]]
    bars = ax.bar(classes, by_class.values, color=colors, width=0.5,
                  edgecolor=PALETTE["bg"], linewidth=2)
    for bar, val in zip(bars, by_class.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=13,
                fontweight="bold", color=PALETTE["text"])

    _apply_dark_theme(fig, ax, "Survival Rate by Passenger Class",
                      ylabel="Survival Rate (%)")
    ax.set_ylim(0, 105)
    chart_path = _save_chart(fig, "survival_by_class")

    return {
        "answer": (
            "🎫 **Survival Rate by Class**\n\n"
            + "\n".join(f"• Class {c}: **{v:.1f}%**" for c, v in by_class.items())
            + "\n\n_First-class passengers had the highest survival rate._"
        ),
        "chart": chart_path,
    }


@tool
def gender_distribution() -> dict:
    """Show the gender distribution of passengers."""
    counts = df["sex"].value_counts()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [PALETTE["primary"], PALETTE["accent"]]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[s.capitalize() for s in counts.index],
        autopct="%1.1f%%", colors=colors, startangle=90,
        textprops={"fontsize": 12, "color": PALETTE["text"]},
        pctdistance=0.75,
        wedgeprops={"width": 0.4, "edgecolor": PALETTE["bg"], "linewidth": 2.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(13)

    ax.text(0, 0, f"{total}\nTotal", ha="center", va="center",
            fontsize=15, fontweight="bold", color=PALETTE["text_sec"])
    fig.set_facecolor(PALETTE["bg"])
    ax.set_title("Gender Distribution", fontsize=16, fontweight="bold",
                 color=PALETTE["text"], pad=18)
    chart_path = _save_chart(fig, "gender_dist")

    return {
        "answer": (
            "👫 **Gender Distribution**\n\n"
            + "\n".join(
                f"• {s.capitalize()}: **{c}** ({round(c/total*100,1)}%)"
                for s, c in counts.items()
            )
        ),
        "chart": chart_path,
    }


@tool
def fare_distribution() -> dict:
    """Show the distribution of ticket fares."""
    fares = df["fare"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(fares, bins=50, color=PALETTE["accent"], edgecolor=PALETTE["bg"],
            linewidth=0.6, alpha=0.85)

    ax.axvline(fares.mean(), color=PALETTE["primary"], linestyle="--",
               linewidth=2, label=f"Mean: ${fares.mean():.2f}", alpha=0.9)
    ax.axvline(fares.median(), color=PALETTE["gold"], linestyle="--",
               linewidth=2, label=f"Median: ${fares.median():.2f}", alpha=0.9)

    _apply_dark_theme(fig, ax, "Distribution of Ticket Fares",
                      xlabel="Fare ($)", ylabel="Number of Passengers")
    ax.legend(fontsize=10, frameon=False, labelcolor=PALETTE["text_sec"])
    chart_path = _save_chart(fig, "fare_distribution")

    return {
        "answer": (
            f"💵 **Fare Distribution**\n\n"
            f"| Stat | Value |\n"
            f"|------|-------|\n"
            f"| Mean | **${fares.mean():.2f}** |\n"
            f"| Median | ${fares.median():.2f} |\n"
            f"| Min | ${fares.min():.2f} |\n"
            f"| Max | ${fares.max():.2f} |\n"
            f"| Std Dev | ${fares.std():.2f} |"
        ),
        "chart": chart_path,
    }


@tool
def dataset_overview() -> dict:
    """Provide a high-level overview of the Titanic dataset with a summary chart."""
    total = len(df)
    survived = int(df["survived"].sum())
    perished = total - survived
    avg_age = round(df["age"].mean(), 1)
    avg_fare = round(df["fare"].mean(), 2)
    male_count = len(df[df["sex"] == "male"])
    female_count = len(df[df["sex"] == "female"])

    # Create an infographic-style multi-panel chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Survival donut
    ax1 = axes[0]
    ax1.pie([survived, perished],
            colors=[PALETTE["teal"], PALETTE["accent"]],
            startangle=90,
            wedgeprops={"width": 0.35, "edgecolor": PALETTE["bg"], "linewidth": 2})
    ax1.text(0, 0, f"{survived}\nSurvived", ha="center", va="center",
             fontsize=14, fontweight="bold", color=PALETTE["teal"])
    ax1.set_title("Survival", fontsize=13, fontweight="bold",
                  color=PALETTE["text"], pad=12)

    # Panel 2: Gender donut
    ax2 = axes[1]
    ax2.pie([male_count, female_count],
            colors=[PALETTE["primary"], PALETTE["accent"]],
            startangle=90,
            wedgeprops={"width": 0.35, "edgecolor": PALETTE["bg"], "linewidth": 2})
    ax2.text(0, 0, f"{total}\nTotal", ha="center", va="center",
             fontsize=14, fontweight="bold", color=PALETTE["text_sec"])
    ax2.set_title("Gender Split", fontsize=13, fontweight="bold",
                  color=PALETTE["text"], pad=12)

    # Panel 3: Class breakdown bar
    ax3 = axes[2]
    class_counts = df["pclass"].value_counts().sort_index()
    colors_bar = CHART_COLORS[:3]
    bars = ax3.bar([f"Class {c}" for c in class_counts.index],
                   class_counts.values, color=colors_bar, width=0.5,
                   edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars, class_counts.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", fontsize=12,
                 fontweight="bold", color=PALETTE["text"])
    ax3.set_title("By Class", fontsize=13, fontweight="bold",
                  color=PALETTE["text"], pad=12)
    ax3.set_facecolor(PALETTE["bg_light"])
    ax3.tick_params(colors=PALETTE["text_sec"])
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=PALETTE["grid_alpha"])

    fig.set_facecolor(PALETTE["bg"])
    fig.suptitle("Titanic Dataset — At a Glance", fontsize=17,
                 fontweight="bold", color=PALETTE["text"], y=1.03)
    fig.tight_layout(pad=2.0)
    chart_path = _save_chart(fig, "overview")

    return {
        "answer": (
            f"📋 **Titanic Dataset Overview**\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Passengers | **{total}** |\n"
            f"| Survived | {survived} ({round(survived/total*100,1)}%) |\n"
            f"| Perished | {perished} ({round(perished/total*100,1)}%) |\n"
            f"| Male | {male_count} ({round(male_count/total*100,1)}%) |\n"
            f"| Female | {female_count} ({round(female_count/total*100,1)}%) |\n"
            f"| Average Age | {avg_age} years |\n"
            f"| Average Fare | ${avg_fare} |"
        ),
        "chart": chart_path,
    }


@tool
def age_survival_analysis() -> dict:
    """Analyze survival rates across age groups."""
    data = df.dropna(subset=["age"]).copy()
    bins = [0, 12, 18, 30, 50, 80]
    labels = ["Child\n(0-12)", "Teen\n(13-17)", "Young Adult\n(18-30)",
              "Adult\n(31-50)", "Senior\n(51+)"]
    data["age_group"] = pd.cut(data["age"], bins=bins, labels=labels, right=False)
    rates = data.groupby("age_group", observed=True)["survived"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = CHART_COLORS[:len(rates)]
    bars = ax.bar(range(len(rates)), rates.values, color=colors, width=0.55,
                  edgecolor=PALETTE["bg"], linewidth=1.5)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, fontsize=10, color=PALETTE["text_sec"])

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=12,
                fontweight="bold", color=PALETTE["text"])

    _apply_dark_theme(fig, ax, "Survival Rate by Age Group",
                      ylabel="Survival Rate (%)")
    ax.set_ylim(0, 105)
    chart_path = _save_chart(fig, "age_survival")

    return {
        "answer": (
            "👶 **Survival by Age Group**\n\n"
            + "\n".join(f"• {lbl.replace(chr(10), ' ')}: **{val:.1f}%**"
                       for lbl, val in zip(labels, rates.values))
            + "\n\n_Children had the highest survival rate (\"women and children first\")._"
        ),
        "chart": chart_path,
    }


@tool
def family_size_analysis() -> dict:
    """Analyze how family size affected survival."""
    data = df.copy()
    data["family_size"] = data["sibsp"] + data["parch"] + 1
    bins = [1, 2, 4, 7, 12]
    labels = ["Alone", "Small (2-3)", "Medium (4-6)", "Large (7+)"]
    data["family_group"] = pd.cut(data["family_size"], bins=bins, labels=labels,
                                   right=False, include_lowest=True)
    rates = data.groupby("family_group", observed=True)["survived"].mean() * 100
    counts = data.groupby("family_group", observed=True).size()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = CHART_COLORS[:len(rates)]
    bars = ax.bar(range(len(rates)), rates.values, color=colors, width=0.55,
                  edgecolor=PALETTE["bg"], linewidth=1.5)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates.index, fontsize=10, color=PALETTE["text_sec"])

    for bar, val, cnt in zip(bars, rates.values, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=12,
                fontweight="bold", color=PALETTE["text"])
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                f"n={cnt}", ha="center", fontsize=9,
                color=PALETTE["text_sec"])

    _apply_dark_theme(fig, ax, "Survival Rate by Family Size",
                      ylabel="Survival Rate (%)")
    ax.set_ylim(0, 105)
    chart_path = _save_chart(fig, "family_survival")

    return {
        "answer": (
            "👨‍👩‍👧‍👦 **Family Size & Survival**\n\n"
            + "\n".join(
                f"• {lbl}: **{val:.1f}%** ({cnt} passengers)"
                for lbl, val, cnt in zip(labels, rates.values, counts.values)
            )
            + "\n\n_Small families (2-3 members) had the best survival odds._"
        ),
        "chart": chart_path,
    }


# ──────────────────────────────────────────────
# 4.  INTENT MATCHING (robust rule-based router)
# ──────────────────────────────────────────────
INTENT_PATTERNS: list[tuple[re.Pattern, callable]] = [
    # Male / female percentage
    (re.compile(r"(percentage|percent|ratio|proportion).*male", re.I), male_percentage),
    (re.compile(r"male.*(percentage|percent|ratio|proportion)", re.I), male_percentage),
    (re.compile(r"how many.*male", re.I), male_percentage),
    (re.compile(r"(men|man).*on.*(titanic|ship|board)", re.I), male_percentage),

    # Average fare
    (re.compile(r"(average|mean|avg).*fare", re.I), average_fare),
    (re.compile(r"fare.*(average|mean|avg)", re.I), average_fare),
    (re.compile(r"ticket.*(price|cost|fare)", re.I), average_fare),
    (re.compile(r"how much.*(fare|ticket|cost|pay)", re.I), average_fare),

    # Embarkation / port
    (re.compile(r"(embark|port|board)", re.I), passengers_per_port),
    (re.compile(r"passenger.*per.*(port|embark)", re.I), passengers_per_port),
    (re.compile(r"(southampton|cherbourg|queenstown)", re.I), passengers_per_port),

    # Age histogram
    (re.compile(r"(histogram|distribution|spread).*age", re.I), age_histogram),
    (re.compile(r"age.*(histogram|distribution|spread|chart|graph)", re.I), age_histogram),
    (re.compile(r"show.*age", re.I), age_histogram),
    (re.compile(r"how old", re.I), age_histogram),
    (re.compile(r"(young|old)est.*passenger", re.I), age_histogram),

    # Age + survival (specific combo)
    (re.compile(r"age.*surviv", re.I), age_survival_analysis),
    (re.compile(r"surviv.*age.*(group|range|bracket)", re.I), age_survival_analysis),
    (re.compile(r"(child|children|kid|teen).*surviv", re.I), age_survival_analysis),

    # Family
    (re.compile(r"famil", re.I), family_size_analysis),
    (re.compile(r"(sibsp|parch|sibling|parent|spouse)", re.I), family_size_analysis),
    (re.compile(r"(alone|solo).*travel", re.I), family_size_analysis),

    # Survival (general — must come after age-survival and family)
    (re.compile(r"surviv", re.I), survival_rate),
    (re.compile(r"(died|death|perish|mortality|live|alive)", re.I), survival_rate),
    (re.compile(r"(who|how many).*(made it|survive|die)", re.I), survival_rate),

    # Gender distribution
    (re.compile(r"gender.*(distribut|split|breakdown|chart|pie)", re.I), gender_distribution),
    (re.compile(r"(male|female).*(distribut|split|breakdown|chart)", re.I), gender_distribution),
    (re.compile(r"(men|women).*vs", re.I), gender_distribution),

    # Fare distribution
    (re.compile(r"fare.*(distribut|histogram|chart|spread)", re.I), fare_distribution),
    (re.compile(r"(expensive|cheap).*(ticket|fare)", re.I), fare_distribution),

    # Overview / summary (keep last — most general)
    (re.compile(r"(overview|summary|summarize|describe|info|dataset|data set|tell me about)", re.I), dataset_overview),
    (re.compile(r"(how many|total).*(passenger|people|record|row)", re.I), dataset_overview),
    (re.compile(r"^(hi|hello|hey|help)", re.I), dataset_overview),
]


# ──────────────────────────────────────────────
# 5.  PUBLIC API  —  answer_question()
# ──────────────────────────────────────────────

def answer_question(question: str) -> dict:
    """
    Interpret a natural-language question about the Titanic dataset,
    perform the relevant analysis, and return a structured response.

    Parameters
    ----------
    question : str
        The user's question in plain English.

    Returns
    -------
    dict
        {
            "answer": str,        # Markdown-formatted answer
            "chart": str | None   # Path to a chart image, or None
        }
    """
    question = question.strip()
    if not question:
        return {
            "answer": "Please ask a question about the Titanic dataset!",
            "chart": None,
        }

    # Try each pattern until one matches
    for pattern, tool_fn in INTENT_PATTERNS:
        if pattern.search(question):
            return tool_fn.invoke("")  # LangChain tool invocation

    # Fallback — no intent matched
    return {
        "answer": (
            "🤔 I'm not sure how to answer that. Here are topics I can help with:\n\n"
            "| Topic | Example Question |\n"
            "|-------|------------------|\n"
            "| 🧑‍🤝‍🧑 Gender | *What percentage of passengers were male?* |\n"
            "| 💰 Fares | *What was the average ticket fare?* |\n"
            "| ⚓ Ports | *How many passengers embarked from each port?* |\n"
            "| 📊 Ages | *Show me a histogram of passenger ages* |\n"
            "| 🛟 Survival | *What was the survival rate?* |\n"
            "| 👶 Age & Survival | *How did age affect survival?* |\n"
            "| 👨‍👩‍👧‍👦 Family | *Did family size affect survival?* |\n"
            "| 📋 Overview | *Give me an overview of the dataset* |"
        ),
        "chart": None,
    }
