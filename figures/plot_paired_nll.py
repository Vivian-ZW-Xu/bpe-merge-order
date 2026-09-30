"""Validate locked per-paper NLL data and render English SVG/PNG figures.

This script uses only the standard library plus the already installed
`rsvg-convert` command for PNG export. It never loads a model or writes inputs.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PAIRED = ROOT / "experiments/sentence_data_revision_v2_paired_analysis"
V2 = ROOT / "experiments/sentence_data_revision_v2"
CSV = PAIRED / "per_paper.csv"
SUMMARY = PAIRED / "summary.json"
ANALYSIS = V2 / "analysis.json"
RESULTS = {
    "holdout_2021": V2 / "results_holdout_2021.json",
    "pilot_2020": V2 / "results_pilot_2020.json",
}
BATCHES = (("holdout_2021", "2021 holdout", 112),
           ("pilot_2020", "2020 pilot", 20))
CONDITIONS = {
    "A": "A_original", "B": "B_original_shuffle_seed42",
    "D": "D_complete_shuffle_seed42", "E": "E_original_rules_D_relative_order",
}
COMPARISONS = (("D_minus_E", "D - E", "E"),
               ("D_minus_B", "D - B", "B"),
               ("D_minus_A", "D - A", "A"))
OUTPUTS = {
    "paper_differences.svg": None,
    "paper_differences.png": None,
    "length_vs_nll.svg": None,
    "length_vs_nll.png": None,
}
BLUE = "#176baf"
ORANGE = "#c96024"
GRID = "#e1e7ed"
INK = "#172a3a"
MUTED = "#526477"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def validate() -> tuple[dict[str, list[dict]], dict, dict[str, str]]:
    require(all(not (HERE / name).exists() for name in OUTPUTS),
            "Figure output exists; refusing to overwrite")
    paths = (CSV, SUMMARY, ANALYSIS, *RESULTS.values())
    before = {str(path): sha(path) for path in paths}
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    prior = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    require(before[str(CSV)] == summary["per_paper_csv_sha256"],
            "CSV differs from paired-analysis summary")
    with CSV.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    require(len(raw) == 132, "Expected 132 per-paper CSV rows")
    batches = {}
    for batch, _, expected in BATCHES:
        source = json.loads(RESULTS[batch].read_text(encoding="utf-8"))
        require(source["status"] == "complete", f"{batch}: incomplete scoring")
        require(before[str(RESULTS[batch])] == summary["source_files"][RESULTS[batch].name]["sha256"],
                f"{batch}: results hash differs from paired-analysis source")
        rows = [row for row in raw if row["batch"] == batch]
        require(len(rows) == expected == len(source["rows"]), f"{batch}: PMCID count")
        require(len({row["pmcid"] for row in rows}) == expected,
                f"{batch}: duplicate PMCID")
        require([row["pmcid"] for row in rows] == [row["pmcid"] for row in source["rows"]],
                f"{batch}: PMCID order differs")
        converted = []
        for row, scored in zip(rows, source["rows"]):
            pmcid = row["pmcid"]
            label = f"{batch}/{pmcid}"
            ids = scored["target_token_ids"]
            require(int(row["target_token_count"]) == len(ids) == scored["target_token_count"],
                    f"{label}: target token count")
            numbers = {key: float(value) for key, value in row.items()
                       if key not in {"batch", "pmcid"}}
            for short, full in CONDITIONS.items():
                condition = scored["conditions"][full]
                require(close(numbers[f"{short}_mean_nll"],
                              condition["mean_nll_per_target_token"]),
                        f"{label}/{short}: NLL differs")
                require(int(numbers[f"{short}_prompt_tokens"]) ==
                        len(condition["prompt_token_ids"]) == condition["prompt_token_count"],
                        f"{label}/{short}: prompt length differs")
            for field, _, other in COMPARISONS:
                require(close(numbers[field],
                              numbers["D_mean_nll"] - numbers[f"{other}_mean_nll"]),
                        f"{label}: paired difference {field}")
            for other in ("E", "B"):
                require(close(numbers[f"D_over_{other}_length_ratio"],
                              numbers["D_prompt_tokens"] / numbers[f"{other}_prompt_tokens"]),
                        f"{label}: prompt length ratio D/{other}")
            converted.append({"pmcid": pmcid, **numbers})
        for field, _, _ in COMPARISONS:
            values = [row[field] for row in converted]
            saved = summary["batches"][batch]["comparisons"][field]
            old = prior["batches"][batch]["new"]["comparisons"][field]
            require(close(statistics.fmean(values), saved["distribution"]["mean"])
                    and close(statistics.fmean(values), old["mean_difference"]),
                    f"{batch}/{field}: mean mismatch")
            wins = sum(value < 0 for value in values)
            require(wins == saved["D_lower"] == old["D_lower_count"],
                    f"{batch}/{field}: win count mismatch")
        for other in ("E", "B"):
            label = f"D_vs_{other}"
            ratios = [row[f"D_over_{other}_length_ratio"] for row in converted]
            saved = summary["batches"][batch]["length_diagnostics"][label]
            require(close(statistics.fmean(ratios), saved["D_over_other_ratio"]["mean"]),
                    f"{batch}/{label}: mean length ratio mismatch")
            require(sum(0.95 <= ratio <= 1.05 for ratio in ratios) ==
                    saved["within_5_percent_length_count"],
                    f"{batch}/{label}: near-length count mismatch")
        batches[batch] = converted
    require(not (set(row["pmcid"] for row in batches["holdout_2021"]) &
                 set(row["pmcid"] for row in batches["pilot_2020"])),
            "Batches contain overlapping PMCIDs")
    require({str(path): sha(path) for path in paths} == before,
            "An input changed during validation")
    return batches, summary, before


class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
                      f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<rect width="100%" height="100%" fill="#ffffff"/>']

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             stroke: str = "none", opacity: float = 1) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
                          f'height="{h:.2f}" fill="{fill}" stroke="{stroke}" '
                          f'opacity="{opacity}"/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
             width: float = 1, dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
                          f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{d}/>')

    def circle(self, x: float, y: float, radius: float, fill: str,
               opacity: float = 1) -> None:
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                          f'fill="{fill}" opacity="{opacity}"/>')

    def text(self, x: float, y: float, value: str, size: int = 16,
             color: str = INK, anchor: str = "start", weight: int = 400) -> None:
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" '
                          f'font-family="Helvetica,Arial,sans-serif" font-size="{size}" '
                          f'font-weight="{weight}" text-anchor="{anchor}">'
                          f'{html.escape(value)}</text>')

    def save(self, path: Path) -> None:
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")


def headline(canvas: SVG, title: str, subtitle: str) -> None:
    canvas.text(64, 53, title, size=28, weight=700)
    canvas.text(64, 82, subtitle, size=17, color=MUTED)
    canvas.circle(69, 113, 6, BLUE)
    canvas.text(84, 119, "D lower NLL", size=15)
    canvas.circle(242, 113, 6, ORANGE)
    canvas.text(257, 119, "D higher NLL", size=15)


def paper_difference_figure(batches: dict[str, list[dict]]) -> None:
    canvas = SVG(1440, 1180)
    headline(canvas, "Paper-level next-sentence NLL differences",
             "One dot per paper; negative D - comparator means lower NLL for D.")
    xmin, xmax = -0.8, 0.55
    x_ticks = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4]
    panels = [("holdout_2021", 190, 545), ("pilot_2020", 790, 255)]
    x_panels = [75, 535, 995]
    width = 370
    for batch, top, height in panels:
        rows = batches[batch]
        title = "2021 holdout" if batch == "holdout_2021" else "2020 pilot"
        canvas.text(75, top - 25, f"{title}  |  n = {len(rows)}", 20, weight=700)
        for column, (field, short, _) in enumerate(COMPARISONS):
            left = x_panels[column]
            plot_top, plot_bottom = top + 40, top + height - 39
            canvas.rect(left, top, width, height, "#ffffff", GRID)
            canvas.text(left + 12, top + 26, short, 19, weight=700)
            values = sorted(row[field] for row in rows)
            canvas.text(left + width - 12, top + 26,
                        f"{sum(v < 0 for v in values)}/{len(values)} lower", 14,
                        color=MUTED, anchor="end")
            project = lambda value: left + (value - xmin) / (xmax - xmin) * width
            for tick in x_ticks:
                xx = project(tick)
                canvas.line(xx, plot_top, xx, plot_bottom,
                            "#9aa9b8" if tick == 0 else GRID,
                            2 if tick == 0 else 1)
                canvas.text(xx, top + height - 15, f"{tick:+.1f}" if tick else "0",
                            13, MUTED, anchor="middle")
            for i, value in enumerate(values):
                yy = plot_top + (i + 0.5) * (plot_bottom - plot_top) / len(values)
                canvas.circle(project(value), yy, 2.5 if len(values) > 50 else 4.2,
                              BLUE if value < 0 else ORANGE, 0.85)
    canvas.text(720, 1093, "Mean target-token NLL difference (D - comparator)",
                17, anchor="middle", weight=600)
    canvas.text(75, 1124,
                "Papers are sorted separately in each panel, most negative at top. Shared official A target IDs; teacher forcing.",
                15, MUTED)
    canvas.text(75, 1150,
                "Not MAUVE. These observations do not identify the causal effect of added rules or prompt length.",
                15, MUTED)
    canvas.save(HERE / "paper_differences.svg")


def length_scatter_figure(batches: dict[str, list[dict]]) -> None:
    canvas = SVG(1400, 1190)
    headline(canvas, "Prompt-length ratios and next-sentence NLL",
             "Actual paper-level observations; lower NLL is better, and negative D - comparator favors D.")
    xmin, xmax = 0.4, 1.06
    ymin, ymax = -0.8, 0.55
    x_ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_ticks = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4]
    panels = [("holdout_2021", 195), ("pilot_2020", 650)]
    cols = [("E", "D_minus_E", 75), ("B", "D_minus_B", 740)]
    panel_width, panel_height = 585, 380
    for batch, top in panels:
        rows = batches[batch]
        title = "2021 holdout" if batch == "holdout_2021" else "2020 pilot"
        canvas.text(75, top - 25, f"{title}  |  n = {len(rows)}", 20, weight=700)
        for other, field, left in cols:
            canvas.rect(left, top, panel_width, panel_height, "#ffffff", GRID)
            canvas.text(left + 15, top + 29, f"D / {other} length vs. D - {other} NLL", 18,
                        weight=700)
            px1, px2 = left + 56, left + panel_width - 24
            py1, py2 = top + 54, top + panel_height - 47
            project_x = lambda value: px1 + (value - xmin) / (xmax - xmin) * (px2 - px1)
            project_y = lambda value: py2 - (value - ymin) / (ymax - ymin) * (py2 - py1)
            canvas.rect(project_x(0.95), py1, project_x(1.05) - project_x(0.95),
                        py2 - py1, "#eff2f4")
            for tick in x_ticks:
                xx = project_x(tick)
                canvas.line(xx, py1, xx, py2, GRID)
                canvas.text(xx, py2 + 20, f"{tick:.1f}", 13, MUTED, anchor="middle")
            for tick in y_ticks:
                yy = project_y(tick)
                canvas.line(px1, yy, px2, yy, GRID)
                canvas.text(px1 - 8, yy + 5, f"{tick:+.1f}" if tick else "0",
                            13, MUTED, anchor="end")
            canvas.line(project_x(1), py1, project_x(1), py2, "#778796", 2, "6 4")
            canvas.line(px1, project_y(0), px2, project_y(0), "#8393a3", 2)
            for row in rows:
                delta = row[field]
                ratio = row[f"D_over_{other}_length_ratio"]
                canvas.circle(project_x(ratio), project_y(delta),
                              4.1 if len(rows) > 50 else 5.2,
                              BLUE if delta < 0 else ORANGE, 0.68)
            canvas.text(project_x(1), py1 + 20, "ratio = 1", 13, MUTED,
                        anchor="end")
            canvas.text(project_x(0.95) - 5, py1 + 39,
                        "No near-length cases", 13, MUTED, anchor="end")
            canvas.text((px1 + px2) / 2, top + panel_height - 7,
                        f"Full-chat prompt tokens: D / {other}", 14,
                        anchor="middle", weight=600)
    canvas.text(75, 1088,
                "Gray band: length ratio 0.95-1.05; no observed paper falls in it. Dashed line: equal prompt length.",
                15, MUTED)
    canvas.text(75, 1116,
                "Shared official A target IDs under teacher forcing; not MAUVE or an open-ended generation score.",
                15, MUTED)
    canvas.text(75, 1144,
                "Length and added merge rules change together here; the scatter does not establish either causal effect.",
                15, MUTED)
    canvas.save(HERE / "length_vs_nll.svg")


def render_png(stem: str) -> None:
    subprocess.run(["rsvg-convert", "--format=png", "--dpi-x=170", "--dpi-y=170",
                    f"--output={HERE / (stem + '.png')}", str(HERE / (stem + ".svg"))],
                   check=True)


def main() -> None:
    batches, _, original_hashes = validate()
    paper_difference_figure(batches)
    length_scatter_figure(batches)
    render_png("paper_differences")
    render_png("length_vs_nll")
    require({name: sha(Path(name)) for name in original_hashes} == original_hashes,
            "Input changed during figure export")
    print(json.dumps({"validated": {batch: len(rows) for batch, rows in batches.items()},
                      "point_counts": {"paper_differences": 3 * sum(map(len, batches.values())),
                                       "length_vs_nll": 2 * sum(map(len, batches.values()))},
                      "source_files_unchanged": True,
                      "outputs": {name: {"bytes": (HERE / name).stat().st_size,
                                         "sha256": sha(HERE / name)} for name in OUTPUTS}},
                     indent=2))


if __name__ == "__main__":
    main()
