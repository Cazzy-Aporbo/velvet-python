"""
Velvet Python CLI — project tooling and interactive demos.

Commands:
    velvet info       Show project structure and environment
    velvet test       Run the pytest suite
    velvet lint       Run ruff on src/ and tests/
    velvet classify   Interactive text classification demo
    velvet tree       Display project as a Rich tree
    velvet recommend  Recommend model families from dataset shape
    velvet clean      Remove __pycache__, .pytest_cache, etc.
    velvet version    Print version and Python info
"""

from __future__ import annotations

import shutil
from argparse import ArgumentParser
import subprocess
import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
except ModuleNotFoundError:
    from typing import Any

    class _FallbackExit(SystemExit):
        """Replacement for `typer.Exit` when Typer is unavailable."""

    class _FallbackText(str):
        def append(self, _text: str, *_args: Any, **_kwargs: Any) -> None:
            return None

        @classmethod
        def assemble(cls, *items: object) -> str:
            return "".join(str(item) for item in items)

    class _FallbackConsole:
        def print(self, *items: object, **_kwargs: object) -> None:
            print(*items)

    class _FallbackPanel:
        def __init__(self, content: object, **_kwargs: object):
            self.content = content

        @classmethod
        def fit(cls, content: object, **_kwargs: object) -> "_FallbackPanel":
            return cls(content, **_kwargs)

        def __str__(self) -> str:
            return str(self.content)

    class _FallbackTable:
        def __init__(self, *_, **__):
            self.rows: list[list[object]] = []

        def add_column(self, *_args: object, **_kwargs: object) -> None:
            return None

        def add_row(self, *values: object) -> None:
            self.rows.append(list(values))

        def __str__(self) -> str:
            if not self.rows:
                return ""
            return "\n".join(", ".join(str(value) for value in row) for row in self.rows)

    class _FallbackTree:
        def __init__(self, label: str, **_kwargs: object):
            self.label = label
            self.children: list[object] = []

        def add(self, text: str) -> "_FallbackTree":
            child = _FallbackTree(text)
            self.children.append(child)
            return child

        def _render(self, level: int = 0) -> str:
            lines = ["  " * level + str(self.label)]
            for child in self.children:
                if isinstance(child, _FallbackTree):
                    lines.append(child._render(level + 1))
                else:
                    lines.append("  " * (level + 1) + str(child))
            return "\n".join(lines)

        def __str__(self) -> str:
            return self._render()

    def _to_int_positive(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise ValueError("value must be positive")
        return parsed

    def _build_fallback_app() -> Any:
        class _FallbackTyper:
            def __init__(self) -> None:
                self.commands: dict[str, object] = {}

            def command(self):
                def decorate(fn: object) -> object:
                    self.commands[fn.__name__] = fn
                    return fn

                return decorate

            def __call__(self) -> None:
                parser = ArgumentParser(prog="velvet")
                subs = parser.add_subparsers(dest="command", required=True)

                sub_info = subs.add_parser("info")
                sub_info.set_defaults(_target=info)

                sub_test = subs.add_parser("test")
                sub_test.add_argument("--verbose", "-v", action="store_true")
                sub_test.add_argument("--cov", "-c", action="store_true")
                sub_test.set_defaults(_target=test)

                sub_lint = subs.add_parser("lint")
                sub_lint.set_defaults(_target=lint)

                sub_classify = subs.add_parser("classify")
                sub_classify.add_argument("text")
                sub_classify.add_argument("--strategy", "-s", default="all")
                sub_classify.set_defaults(_target=classify)

                sub_tree = subs.add_parser("tree")
                sub_tree.add_argument("--depth", "-d", type=int, default=3)
                sub_tree.set_defaults(_target=tree)

                sub_clean = subs.add_parser("clean")
                sub_clean.add_argument("--all", "-a", action="store_true", dest="all_")
                sub_clean.set_defaults(_target=clean)

                sub_pipeline = subs.add_parser("pipeline")
                sub_pipeline.add_argument("--epochs", "-e", type=_to_int_positive, default=3)
                sub_pipeline.add_argument("--output", "-o", default="artifacts")
                sub_pipeline.add_argument("--seed", "-s", type=int, default=42)
                sub_pipeline.add_argument("--ledger", default="")
                sub_pipeline.set_defaults(_target=pipeline)

                sub_recommend = subs.add_parser("recommend")
                sub_recommend.add_argument("--rows", "-n", type=_to_int_positive, default=200, dest="row_count")
                sub_recommend.add_argument("--classes", "-c", type=int, default=3)
                sub_recommend.add_argument("--probabilities", action="store_true", dest="need_probabilities")
                sub_recommend.add_argument("--no-explainability", action="store_false", dest="prioritize_explainability", default=True)
                sub_recommend.add_argument("--p95-latency-ms", type=int, default=None)
                sub_recommend.set_defaults(_target=recommend)

                sub_version = subs.add_parser("version")
                sub_version.set_defaults(_target=version)

                parsed = parser.parse_args()
                target = parsed._target
                kwargs = vars(parsed)
                kwargs.pop("command")
                kwargs.pop("_target")
                target(**kwargs)

        return _FallbackTyper()

    typer = None
    Exit = _FallbackExit
    Option = lambda *args, **kwargs: kwargs.get("default", None)
    Argument = lambda *args, **kwargs: kwargs.get("default", None)
    Console = _FallbackConsole
    Panel = _FallbackPanel
    Table = _FallbackTable
    Text = _FallbackText
    Tree = _FallbackTree
    app = _build_fallback_app()

else:
    Option = typer.Option
    Argument = typer.Argument
    Exit = typer.Exit
    Console = Console
    Panel = Panel
    Table = Table
    Text = Text
    Tree = Tree
    app = typer.Typer(
        name="velvet",
        help="Velvet Python CLI",
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )

from src import __author__, __version__
from src.ai import CosineSimilarityClassifier, NaiveBayesClassifier
from src.algorithm_guide import recommend_text_algorithms
from src.evidence_ledger import build_evidence_ledger, write_evidence_ledger
from src.model_registry import model_builders
from src.pipeline import dump_run_manifests, run_epochs, summarize_run_series, summarize_runs

PROJECT_ROOT = Path(__file__).parent
console = Console()


def _gradient(text: str) -> Text:
    """Render text with a pastel gradient — lavender to rose."""
    colors = [
        "#EDE5FF", "#E8DFF9", "#E3D9F3", "#DED3ED",
        "#D9CDE7", "#DCC7E1", "#DFC1DB", "#E2BBD5",
        "#E5B5CF", "#FFE4E1",
    ]
    out = Text()
    for i, ch in enumerate(text):
        idx = min(int(i / max(len(text), 1) * len(colors)), len(colors) - 1)
        out.append(ch, style=colors[idx])
    return out


def _header() -> None:
    console.print(Panel(
        Text.assemble(
            _gradient("VELVET PYTHON"), "\n",
            Text(f"v{__version__}", style="#8B7D8B"), " · ",
            Text(__author__, style="#706B70"),
            justify="center",
        ),
        border_style="#DDA0DD",
        padding=(1, 2),
    ))


@app.command()
def info() -> None:
    """Show project structure and environment."""
    _header()

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Key", style="#8B7D8B")
    tbl.add_column("Value", style="#706B70")

    tbl.add_row("Python", sys.version.split()[0])
    tbl.add_row("Platform", sys.platform)
    tbl.add_row("Venv", "active" if sys.prefix != sys.base_prefix else "none")
    tbl.add_row("src/ modules", str(len(list((PROJECT_ROOT / "src").glob("*.py")))))
    tbl.add_row("tests/", str(len(list((PROJECT_ROOT / "tests").glob("test_*.py")))))
    tbl.add_row("scripts/", str(len(list((PROJECT_ROOT / "scripts").glob("*.py")))))
    tbl.add_row("pyfiles/", str(len(list((PROJECT_ROOT / "pyfiles").glob("*.py")))))

    console.print(Panel(tbl, title="[#8B7D8B]Environment[/#8B7D8B]", border_style="#E6E6FA", padding=(1, 2)))


@app.command()
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    coverage: bool = typer.Option(False, "--cov", "-c"),
) -> None:
    """Run the pytest suite."""
    _header()
    cmd = [sys.executable, "-m", "pytest", "tests/", "--tb=short"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    raise typer.Exit(result.returncode)


@app.command()
def lint() -> None:
    """Lint src/ and tests/ with ruff."""
    _header()
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        console.print("[green]No issues found.[/green]")
    raise typer.Exit(result.returncode)


@app.command()
def classify(
    text: str = typer.Argument(..., help="Text to classify"),
    strategy: str = typer.Option("all", "--strategy", "-s", help="rule | bayes | cosine | all"),
) -> None:
    """Classify text using strategies from src/ai.py."""
    from src.ai import classify_text
    from src.data_utils import load_dataset

    _header()
    data = load_dataset()
    texts = [text for text, _ in data]
    labels = [label for _, label in data]

    tbl = Table(title=f"[#8B7D8B]Classifying:[/#8B7D8B] {text}", border_style="#DDA0DD")
    tbl.add_column("Strategy", style="#FFE4E1")
    tbl.add_column("Prediction", style="#EDE5FF")

    if strategy in ("rule", "all"):
        tbl.add_row("Rule-based", classify_text(text))

    if strategy in ("bayes", "all"):
        nb = NaiveBayesClassifier()
        nb.train(texts, labels)
        pred = nb.predict(text)
        proba = nb.predict_proba(text)
        top_p = max(proba.values())
        tbl.add_row("Naive Bayes", f"{pred} ({top_p:.0%})")

    if strategy in ("cosine", "all"):
        cs = CosineSimilarityClassifier()
        cs.train(texts, labels)
        tbl.add_row("Cosine Similarity", cs.predict(text))

    console.print(tbl)


@app.command()
def tree(
    depth: int = typer.Option(3, "--depth", "-d"),
) -> None:
    """Display project structure as a Rich tree."""
    _header()
    root_tree = Tree("[#8B7D8B]velvet-python[/#8B7D8B]", style="#DDA0DD")

    skip = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

    def _walk(node, path: Path, level: int = 0) -> None:
        if level >= depth:
            return
        try:
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return
        for item in items:
            if item.name in skip or item.name.startswith("."):
                continue
            if item.is_dir():
                branch = node.add(f"[#FFE4E1]{item.name}/[/#FFE4E1]")
                _walk(branch, item, level + 1)
            else:
                ext_style = {
                    ".py": "#EDE5FF", ".md": "#F0E6FF",
                    ".yml": "#FFF0F5", ".yaml": "#FFF0F5",
                    ".toml": "#FFF0F5",
                }.get(item.suffix, "#FFEFD5")
                node.add(f"[{ext_style}]{item.name}[/{ext_style}]")

    _walk(root_tree, PROJECT_ROOT)
    console.print(root_tree)


@app.command()
def clean(
    all_: bool = typer.Option(False, "--all", "-a", help="Remove build artifacts too"),
) -> None:
    """Remove __pycache__, .pytest_cache, and other transient files."""
    _header()
    patterns = [
        "**/__pycache__", "**/*.pyc", "**/.pytest_cache",
        "**/.mypy_cache", "**/.ruff_cache", "**/.coverage",
    ]
    if all_:
        patterns.extend(["build", "dist", "**/*.egg-info"])

    removed = 0
    for pattern in patterns:
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1

    console.print(f"[green]Cleaned {removed} items.[/green]")


@app.command()
def pipeline(
    epochs: int = typer.Option(3, "--epochs", "-e", min=1),
    output: str = typer.Option("artifacts", "--output", "-o", help="Directory for run manifests"),
    seed: int = typer.Option(42, "--seed", "-s", help="Base seed for deterministic splits"),
    ledger: str = typer.Option("", "--ledger", help="Optional path for evidence ledger output"),
) -> None:
    """Run reproducible baseline pipelines and save JSON manifests."""
    _header()

    from src.data_utils import load_dataset

    dataset = load_dataset()
    builders = model_builders()

    runs = []
    for name, builder in builders.items():
        runs.extend(
            run_epochs(
                model_name=name,
                model_builder=builder,
                data=dataset,
                epochs=epochs,
                seed=seed,
            ),
        )

    paths = dump_run_manifests(runs, PROJECT_ROOT / output)
    summary = summarize_runs(runs)
    series_summary = summarize_run_series(runs)

    ledger_payload = None
    ledger_path = None
    if ledger:
        ledger_payload = build_evidence_ledger(runs, accuracy_spread_threshold=0.05)
        ledger_path = write_evidence_ledger(ledger_payload, PROJECT_ROOT / ledger)

    console.print(Panel.fit("Pipeline run complete", border_style="#DDA0DD"))
    table = Table(title="Summary", border_style="#DDA0DD")
    table.add_column("Model")
    table.add_column("Seed")
    table.add_column("Accuracy")
    table.add_column("Train")
    table.add_column("Test")

    for row in summary:
        table.add_row(row["model_name"], str(row["seed"]), f"{row['accuracy']:.4f}",
                      str(row["train_size"]), str(row["test_size"]))

    console.print(table)
    series_table = Table(title="Model families", border_style="#DDA0DD")
    series_table.add_column("Model")
    series_table.add_column("Runs")
    series_table.add_column("Accuracy mean")
    series_table.add_column("Spread")
    series_table.add_column("Dataset hash stable")

    for row in series_summary:
        series_table.add_row(
            row["model_name"],
            str(row["run_count"]),
            f"{row['accuracy']['mean']:.4f}",
            f"{row['accuracy']['spread']:.4f}",
            "yes" if row["dataset_hash_stable"] else "no",
        )

    console.print(series_table)
    console.print("[#8B7D8B]Manifests:[/#8B7D8B]")
    for path in paths:
        console.print(f" - {path}")
    if ledger_path is not None and ledger_payload is not None:
        console.print(f"[#8B7D8B]Ledger:[/#8B7D8B] {ledger_path}")
        console.print(
            "[#6f6680]Drift alerts:[/#6f6680] "
            f"{len(ledger_payload['drift_alerts'])} "
            f"(models: {ledger_payload['model_count']})"
        )


@app.command()
def recommend(
    row_count: int = typer.Option(
        200,
        "--rows",
        "-n",
        min=1,
        help="Estimated labeled row count in your first dataset",
    ),
    class_count: int = typer.Option(
        3,
        "--classes",
        "-c",
        min=2,
        help="Estimated number of classes you need to predict",
    ),
    need_probabilities: bool = typer.Option(
        False,
        "--probabilities",
        help="Prefer models that expose probability outputs",
    ),
    prioritize_explainability: bool = typer.Option(
        True,
        "--explainability/--no-explainability",
        help="Trade some complexity for interpretability",
    ),
    p95_latency_ms: int | None = typer.Option(
        None,
        "--p95-latency-ms",
        min=1,
        help="Target p95 latency budget in milliseconds",
    ),
) -> None:
    """Recommend model families based on constraints."""
    _header()

    recommendations = recommend_text_algorithms(
        row_count=row_count,
        class_count=class_count,
        needs_probabilities=need_probabilities,
        needs_explainability=prioritize_explainability,
        p95_latency_ms=p95_latency_ms,
    )

    table = Table(
        title=f"[#8B7D8B]Recommendations for {row_count} rows x {class_count} classes[/#8B7D8B]",
        border_style="#DDA0DD",
    )
    table.add_column("Rank", justify="right", style="#FFE4E1")
    table.add_column("Model", style="#EDE5FF")
    table.add_column("Best for", style="#F0E6FF")
    table.add_column("Tradeoffs", style="#E6D9F0")

    for idx, profile in enumerate(recommendations, start=1):
        table.add_row(
            str(idx),
            profile.name.replace("_", " ").title(),
            ", ".join(profile.best_for),
            "; ".join(profile.avoid_if),
        )

    console.print(Panel.fit(
        "This is an engineering starting point, not a final architecture choice. "
        "Start with recommendation #1, run one manifest sweep, then decide what "
        "to add after the observed drift.",
        border_style="#EED8FA",
    ))
    console.print(table)


@app.command()
def version() -> None:
    """Print version and Python info."""
    console.print(_gradient(f"Velvet Python v{__version__}"))
    console.print(f"[#706B70]{__author__}[/#706B70]")
    console.print(f"[#706B70]Python {sys.version.split()[0]}[/#706B70]")


if __name__ == "__main__":
    app()
