"""
Generate visual representations of the Kalmyk Image DH analysis outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


LOGGER = logging.getLogger(__name__)
FIGURES_DIR = Path("output") / "figures"


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure: %s", path)


def plot_frequency(df: pd.DataFrame, output_path: Path) -> None:
    year_counts = (
        df.dropna(subset=["year"])
        .groupby("year")
        .size()
        .reset_index(name="mentions")
        .sort_values("year")
    )
    if year_counts.empty:
        LOGGER.warning("No valid year data for frequency plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=year_counts, x="year", y="mentions", marker="o", ax=ax)
    ax.set_title("Kalmyk ethnonym mentions by year")
    ax.set_ylabel("Number of contexts")
    ax.set_xlabel("Year")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_figure(fig, output_path)


def wordcloud_ethnonym(df: pd.DataFrame, ethnonym: str, output_path: Path) -> None:
    ethnonym = (ethnonym or "").lower()
    if ethnonym and "ethnonym_normalised" in df.columns:
        target_df = df[df["ethnonym_normalised"].fillna("").str.lower() == ethnonym]
    else:
        target_df = df
    if "summary_en" in target_df.columns:
        values = target_df["summary_en"].dropna().tolist()
    else:
        values = target_df["context"].dropna().tolist() if "context" in target_df.columns else []
    text_source = " ".join(filter(None, values))
    if not text_source and "context" in df.columns:
        text_source = " ".join(filter(None, df["context"].dropna().tolist()))

    if not text_source.strip():
        LOGGER.warning("No text available to generate word cloud.")
        return

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        collocations=False,
    ).generate(text_source)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Kalmyk context word cloud")
    _save_figure(fig, output_path)


def network_graph(df: pd.DataFrame, output_path: Path, min_weight: int = 2) -> None:
    if df.empty:
        LOGGER.warning("Empty contexts; skipping network graph.")
        return

    G = nx.Graph()
    edge_weights: dict[tuple[str, str], int] = {}
    authors: set[str] = set()
    ethnonyms: set[str] = set()
    places: set[str] = set()

    for row in df.to_dict(orient="records"):
        author = row.get("author") or "Unknown author"
        ethnonym = row.get("ethnonym_normalised") or row.get("ethnonym")
        if not ethnonym:
            continue

        authors.add(author)
        ethnonyms.add(ethnonym)
        edge_weights[(author, ethnonym)] = edge_weights.get((author, ethnonym), 0) + 1

        for place in row.get("toponyms", []) or []:
            places.add(place)
            edge_weights[(ethnonym, place)] = edge_weights.get((ethnonym, place), 0) + 1

    for author in authors:
        G.add_node(author, type="author")
    for ethnonym in ethnonyms:
        G.add_node(ethnonym, type="ethnonym")
    for place in places:
        G.add_node(place, type="place")

    for (source, target), weight in edge_weights.items():
        if weight < min_weight:
            continue
        G.add_edge(source, target, weight=weight)

    if G.number_of_edges() == 0:
        LOGGER.warning("Network graph has no edges; skipping visualisation.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.3)
    fig, ax = plt.subplots(figsize=(10, 8))

    node_colors = []
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type == "author":
            node_colors.append("#1f78b4")
        elif node_type == "ethnonym":
            node_colors.append("#33a02c")
        else:
            node_colors.append("#fb9a99")

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_size=900,
        font_size=9,
        node_color=node_colors,
        edge_color="#999999",
        width=[0.5 + w * 0.2 for w in weights],
    )
    ax.set_title("Author–Ethnonym–Place network")
    ax.axis("off")
    _save_figure(fig, output_path)


def plot_semantic_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if "semantic_label" not in df.columns:
        LOGGER.warning("semantic_label column missing; skipping semantic distribution.")
        return
    counts = df["semantic_label"].value_counts().sort_values(ascending=False)
    if counts.empty:
        LOGGER.warning("semantic_label column empty; skipping semantic distribution.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Distribution of Kalmyk Image Types")
    ax.set_xlabel("Semantic label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_sentiment_by_author(df: pd.DataFrame, output_path: Path) -> None:
    if {"author", "attitude"}.difference(df.columns):
        LOGGER.warning("Required columns missing for sentiment plot.")
        return
    crosstab = pd.crosstab(df["author"], df["attitude"])
    if crosstab.empty:
        LOGGER.warning("No sentiment data to visualise by author.")
        return
    ax = crosstab.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))
    ax.set_title("Sentiment by Author")
    ax.set_xlabel("Author")
    ax.set_ylabel("Contexts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_figure(ax.get_figure(), output_path)


def create_visuals(df: pd.DataFrame, figures_dir: Path | str = FIGURES_DIR) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_frequency(df, figures_dir / "mentions_by_year.png")
    wordcloud_ethnonym(df, "kalmyk", figures_dir / "wordcloud.png")
    network_graph(df, figures_dir / "network.png", min_weight=2)
    plot_semantic_distribution(df, figures_dir / "semantic_distribution.png")
    plot_sentiment_by_author(df, figures_dir / "sentiment_by_author.png")


__all__ = [
    "create_visuals",
    "plot_frequency",
    "wordcloud_ethnonym",
    "network_graph",
    "plot_semantic_distribution",
    "plot_sentiment_by_author",
]

