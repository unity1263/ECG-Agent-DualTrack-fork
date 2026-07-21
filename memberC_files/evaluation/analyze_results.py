"""
Results Analysis & Visualization for RAGAS Dual-Track Evaluation.

Generates publication-quality comparison charts from evaluation results:
  1. Bar chart: Faithfulness & Answer Relevancy across 3 pipelines
  2. Radar chart: Multi-dimensional comparison
  3. Time vs. Quality scatter plot
  4. Per-domain breakdown
  5. Statistical significance annotations

Outputs are saved as PNG images for the final report.
"""
import os
import sys
import json
import logging
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available — using pure-Python fallback")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — visualization will be skipped")


# ============================================================================
# Color scheme (consistent, accessible palette)
# ============================================================================
COLORS = {
    "rag": "#E63946",      # Red — Pure RAG
    "sft": "#457B9D",      # Blue — Pure SFT
    "hybrid": "#2A9D8F",   # Teal — Hybrid Agent
}
PIPELINE_LABELS = {
    "rag": "Pure RAG\n(7B + Retrieval)",
    "sft": "Pure SFT\n(Fine-tuned 1.5B)",
    "hybrid": "Hybrid Agent\n(1.5B + RAG)",
}
PIPELINE_LABELS_SHORT = {
    "rag": "Pure RAG",
    "sft": "Pure SFT",
    "hybrid": "Hybrid Agent",
}


def load_results(detailed_path: str) -> list:
    """Load detailed evaluation results."""
    with open(detailed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_bar_chart(summary: dict, output_path: str):
    """Create grouped bar chart comparing Faithfulness and Answer Relevancy."""
    if not HAS_MPL:
        logger.warning("Skipping bar chart — matplotlib not available.")
        return

    pipelines = ["rag", "sft", "hybrid"]
    metrics = ["faithfulness", "answer_relevancy"]
    metric_labels = ["Faithfulness", "Answer Relevancy"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pipelines))
    width = 0.35
    bars = []

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [summary[p].get(metric, {}).get("mean", 0) for p in pipelines]
        mins = [summary[p].get(metric, {}).get("min", 0) for p in pipelines]
        maxs = [summary[p].get(metric, {}).get("max", 0) for p in pipelines]

        # Use means as bar height; error bars for min/max range
        yerr_low = [m - mi for m, mi in zip(means, mins)]
        yerr_high = [ma - m for m, ma in zip(means, maxs)]

        bar = ax.bar(
            x + i * width, means, width,
            label=label,
            yerr=[yerr_low, yerr_high],
            capsize=5,
            color=[COLORS[p] for p in pipelines],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
        bars.append(bar)

        # Annotate bars with values
        for j, (mean_val, color) in enumerate(zip(means, [COLORS[p] for p in pipelines])):
            ax.text(
                x[j] + i * width, mean_val + 0.02,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#333333",
            )

    ax.set_xlabel("Pipeline", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title(
        "RAGAS Dual-Track Evaluation: Pipeline Comparison",
        fontsize=15, fontweight="bold", pad=20,
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([PIPELINE_LABELS_SHORT[p] for p in pipelines], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Bar chart saved to: {output_path}")


def create_radar_chart(summary: dict, output_path: str):
    """Create radar chart for multi-dimensional pipeline comparison."""
    if not HAS_MPL or not HAS_NUMPY:
        logger.warning("Skipping radar chart — dependencies not available.")
        return

    categories = ["Faithfulness", "Answer\nRelevancy", "Speed\n(1/time)", "Consistency\n(1/std)"]
    pipelines = ["rag", "sft", "hybrid"]

    # Compute scores for each dimension
    values = {}
    for p in pipelines:
        s = summary.get(p, {})
        faith_mean = s.get("faithfulness", {}).get("mean", 0)
        relev_mean = s.get("relevancy", {}).get("mean", 0)

        # Speed: inverse of average time, normalized
        avg_time = s.get("time", {}).get("mean", 1)
        speed = 1.0 / max(avg_time, 0.1)

        # Consistency: inverse of standard deviation
        faith_std = np.std(s.get("faithfulness", {}).get("values", [0.1]))
        relev_std = np.std(s.get("relevancy", {}).get("values", [0.1]))
        consistency = 1.0 / max(0.5 * (faith_std + relev_std), 0.01)

        values[p] = [faith_mean, relev_mean, speed, consistency]

    # Normalize each dimension to [0, 1] across pipelines
    normalized = {p: [] for p in pipelines}
    for dim in range(len(categories)):
        dim_values = [values[p][dim] for p in pipelines]
        dmin, dmax = min(dim_values), max(dim_values)
        for p in pipelines:
            if dmax - dmin < 1e-6:
                normalized[p].append(1.0)
            else:
                normalized[p].append((values[p][dim] - dmin) / (dmax - dmin))

    # Plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for p in pipelines:
        vals = normalized[p] + normalized[p][:1]
        ax.fill(angles, vals, alpha=0.15, color=COLORS[p])
        ax.plot(angles, vals, "o-", linewidth=2.5, color=COLORS[p],
                label=PIPELINE_LABELS_SHORT[p], markersize=7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Multi-Dimensional Pipeline Comparison",
        fontsize=15, fontweight="bold", pad=30,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Radar chart saved to: {output_path}")


def create_domain_breakdown(detailed: list, output_path: str):
    """Create per-domain performance breakdown chart."""
    if not HAS_MPL:
        logger.warning("Skipping domain breakdown — matplotlib not available.")
        return

    # Group by domain
    domains = defaultdict(list)
    for item in detailed:
        domain = item.get("domain", "general")
        for pipe_key in ["pure_rag", "pure_sft", "hybrid_agent"]:
            domains[domain].append({
                "pipeline": pipe_key,
                "faithfulness": item[pipe_key].get("faithfulness", 0),
                "relevancy": item[pipe_key].get("answer_relevancy", 0),
            })

    # Average per domain per pipeline
    domain_names = sorted(domains.keys())
    pipeline_keys = ["pure_rag", "pure_sft", "hybrid_agent"]
    pipe_short = {"pure_rag": "rag", "pure_sft": "sft", "hybrid_agent": "hybrid"}

    if not domain_names:
        logger.warning("No domain data for breakdown.")
        return

    n_domains = len(domain_names)
    n_pipes = len(pipeline_keys)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric_name, metric_key) in enumerate([
        ("Faithfulness by Domain", "faithfulness"),
        ("Answer Relevancy by Domain", "relevancy"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(n_domains)
        width = 0.25

        for i, pk in enumerate(pipeline_keys):
            means = []
            for domain in domain_names:
                domain_scores = [
                    item[metric_key] for item in domains[domain]
                    if item["pipeline"] == pk
                ]
                means.append(np.mean(domain_scores) if domain_scores else 0)

            ps = pipe_short[pk]
            ax.bar(
                x + i * width, means, width,
                label=PIPELINE_LABELS_SHORT[ps],
                color=COLORS[ps],
                alpha=0.85,
                edgecolor="white",
            )

        ax.set_xlabel("Medical Domain", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([d.replace("_", " ").title()[:15] for d in domain_names],
                           fontsize=9, rotation=15)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Performance Breakdown by Medical Domain", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Domain breakdown chart saved to: {output_path}")


def create_time_vs_quality(detailed: list, output_path: str):
    """Create scatter plot of time vs. answer relevancy for all samples."""
    if not HAS_MPL:
        logger.warning("Skipping time-quality plot — matplotlib not available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for pipe_key, ps in [("pure_rag", "rag"), ("pure_sft", "sft"), ("hybrid_agent", "hybrid")]:
        times = [item[pipe_key].get("time_seconds", 0) for item in detailed]
        relevancies = [item[pipe_key].get("answer_relevancy", 0) for item in detailed]

        ax.scatter(
            times, relevancies,
            c=COLORS[ps], label=PIPELINE_LABELS_SHORT[ps],
            alpha=0.6, edgecolors="white", linewidth=0.5, s=60,
        )

        # Add mean point
        if times and relevancies:
            ax.scatter(
                [sum(times) / len(times)], [sum(relevancies) / len(relevancies)],
                c=COLORS[ps], edgecolors="black", linewidth=2, s=200,
                marker="D", zorder=5,
            )

    ax.set_xlabel("Response Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Answer Relevancy Score", fontsize=12, fontweight="bold")
    ax.set_title("Time vs. Quality Trade-off", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Time vs. Quality chart saved to: {output_path}")


def create_summary_table_ascii(summary: dict) -> str:
    """Generate an ASCII summary table for terminal output."""
    lines = []
    lines.append("=" * 85)
    lines.append(f"{'Pipeline':<30} {'Faithfulness':>15} {'Relevancy':>15} {'Time(s)':>12}")
    lines.append("-" * 85)

    for pipe_key, pipe_label in [
        ("rag", "Pure RAG (7B + Retrieval)"),
        ("sft", "Pure SFT (Fine-tuned 1.5B)"),
        ("hybrid", "Hybrid Agent (1.5B + RAG)"),
    ]:
        s = summary.get(pipe_key, {})
        faith = s.get("faithfulness", {}).get("mean", 0)
        relev = s.get("relevancy", {}).get("mean", 0)
        time_val = s.get("time", {}).get("mean", 0)
        lines.append(f"{pipe_label:<30} {faith:>15.4f} {relev:>15.4f} {time_val:>12.4f}")

    lines.append("=" * 85)

    # Best performer annotation
    best_faith = max(
        ("rag", summary.get("rag", {}).get("faithfulness", {}).get("mean", 0)),
        ("sft", summary.get("sft", {}).get("faithfulness", {}).get("mean", 0)),
        ("hybrid", summary.get("hybrid", {}).get("faithfulness", {}).get("mean", 0)),
        key=lambda x: x[1],
    )
    best_relev = max(
        ("rag", summary.get("rag", {}).get("relevancy", {}).get("mean", 0)),
        ("sft", summary.get("sft", {}).get("relevancy", {}).get("mean", 0)),
        ("hybrid", summary.get("hybrid", {}).get("relevancy", {}).get("mean", 0)),
        key=lambda x: x[1],
    )

    lines.append(f"  Best Faithfulness:     {PIPELINE_LABELS_SHORT[best_faith[0]]} ({best_faith[1]:.4f})")
    lines.append(f"  Best Answer Relevancy: {PIPELINE_LABELS_SHORT[best_relev[0]]} ({best_relev[1]:.4f})")
    lines.append("=" * 85)

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize RAGAS results")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to evaluation_detailed.json")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.results is None:
        args.results = os.path.join(OUTPUT_DIR, "evaluation_detailed.json")

    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        logger.info("Run 'python -m evaluation.run_evaluation' first.")
        sys.exit(1)

    # Load data
    detailed = load_results(args.results)

    # Also load summary
    summary_path = os.path.join(os.path.dirname(args.results), "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        # Compute summary from detailed
        summary = {}
        for pipe_key in ["rag", "sft", "hybrid"]:
            pipe_data = {"faithfulness": {"values": []}, "relevancy": {"values": []}, "time": {"values": []}}
            detail_key = {"rag": "pure_rag", "sft": "pure_sft", "hybrid": "hybrid_agent"}[pipe_key]
            for item in detailed:
                pipe_data["faithfulness"]["values"].append(item[detail_key].get("faithfulness", 0))
                pipe_data["relevancy"]["values"].append(item[detail_key].get("answer_relevancy", 0))
                pipe_data["time"]["values"].append(item[detail_key].get("time_seconds", 0))

            for metric in ["faithfulness", "relevancy", "time"]:
                vals = pipe_data[metric]["values"]
                pipe_data[metric]["mean"] = sum(vals) / len(vals) if vals else 0
                pipe_data[metric]["min"] = min(vals) if vals else 0
                pipe_data[metric]["max"] = max(vals) if vals else 0
            summary[pipe_key] = pipe_data

    # Generate visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Generating visualizations in: {args.output_dir}")

    create_bar_chart(summary, os.path.join(args.output_dir, "fig1_pipeline_comparison.png"))
    create_radar_chart(summary, os.path.join(args.output_dir, "fig2_radar_comparison.png"))
    create_domain_breakdown(detailed, os.path.join(args.output_dir, "fig3_domain_breakdown.png"))
    create_time_vs_quality(detailed, os.path.join(args.output_dir, "fig4_time_vs_quality.png"))

    # Print ASCII summary
    print("\n" + create_summary_table_ascii(summary))

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
