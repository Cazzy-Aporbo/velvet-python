"""Build the GitHub Pages catalog for velvet-python.

The output is a single JSON document consumed by docs/app.js.
It is intentionally dependency-free so the catalog can be rebuilt
on any standard Python installation.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "catalog.json"
REPO_URL = "https://github.com/Cazzy-Aporbo/velvet-python/blob/main"

SKIP_PARTS = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}

CATEGORY_DEFS: tuple[dict[str, str], ...] = (
    {
        "key": "core",
        "label": "Core Systems",
        "match": "src/",
        "description": "The part of the repository where inputs become behavior, and behavior becomes evidence.",
    },
    {
        "key": "cli",
        "label": "Command Surface",
        "match": "CLI.py",
        "description": "Interactive entrypoints that turn the codebase into something you can run, inspect, and explain.",
    },
    {
        "key": "scripts",
        "label": "Workflow Scripts",
        "match": "scripts/",
        "description": "Runners, audits, and portfolio-style utilities that package repeatable work into useful commands.",
    },
    {
        "key": "tests",
        "label": "Evidence & Tests",
        "match": "tests/",
        "description": "The proof layer: contracts, failure cases, and the checks that keep claims inspectable.",
    },
    {
        "key": "environments",
        "label": "Environment Design",
        "match": "environments/",
        "description": "Setup design, environment automation, and the practical work of making Python usable across machines.",
    },
    {
        "key": "foundations",
        "label": "Concept Studies",
        "match": "pyfiles/",
        "description": "Standalone explorations that teach language fundamentals, mathematical thinking, and implementation style.",
    },
    {
        "key": "games",
        "label": "Interactive Worlds",
        "match": "pyfiles/games/",
        "description": "Game-like programs and simulations that teach by motion, constraint, and experimentation.",
    },
    {
        "key": "applied",
        "label": "Applied Analysis",
        "match": "pyfiles/more/",
        "description": "Scenario-heavy work that shows how Python travels into pricing, dashboards, analysis, and decision support.",
    },
    {
        "key": "labs",
        "label": "Lab Fragments",
        "match": "pyfiles/test_files/",
        "description": "Smaller, sharper exercises used to try, break, repair, and understand an idea in isolation.",
    },
    {
        "key": "legacy-env",
        "label": "Environment Labs",
        "match": "01-environments/",
        "description": "Earlier environment and dependency experiments that still show the shape of the thinking.",
    },
)

TRACKS = (
    {
        "id": "foundations",
        "title": "Start With Fluency",
        "description": "Read small files that expose basic syntax, control flow, and mathematical thinking without hiding the mechanics.",
        "categories": ["foundations", "labs"],
    },
    {
        "id": "systems",
        "title": "Move Into Systems",
        "description": "Step into the core modules where validation, modeling, manifests, and evidence become one continuous workflow.",
        "categories": ["core", "cli", "scripts"],
    },
    {
        "id": "proof",
        "title": "Learn Through Proof",
        "description": "Use the tests as a study guide for what the code promises, what it rejects, and where it is meant to fail fast.",
        "categories": ["tests"],
    },
    {
        "id": "worlds",
        "title": "Play Through Ideas",
        "description": "Use simulations, games, and interactive experiments to see algorithms as motion instead of only text.",
        "categories": ["games", "applied"],
    },
)

FEATURED_PATHS = {
    "CLI.py",
    "src/ai.py",
    "src/data_utils.py",
    "src/pipeline.py",
    "src/evidence_ledger.py",
    "src/model_registry.py",
    "scripts/run_experiments.py",
    "scripts/dataset_audit.py",
    "pyfiles/python_lessons_suite.py",
    "pyfiles/recursion_masterclass.py",
    "pyfiles/statistics_module_walkthrough.py",
    "tests/test_pipeline_contracts.py",
}

TAG_KEYWORDS = {
    "bayes": "Bayes",
    "cosine": "Cosine Similarity",
    "tfidf": "TF-IDF",
    "regression": "Regression",
    "matrix": "Matrix Work",
    "recursion": "Recursion",
    "statistics": "Statistics",
    "probability": "Probability",
    "quantum": "Simulation",
    "game": "Interactive Learning",
    "dashboard": "Dashboarding",
    "sankey": "Flow Analysis",
    "audit": "Auditing",
    "pipeline": "Pipelines",
    "dataset": "Data Quality",
    "environment": "Environment Setup",
    "config": "Configuration",
    "visual": "Visualization",
    "neural": "Neural Systems",
    "pricing": "Pricing Logic",
    "api": "API Work",
}


@dataclass(frozen=True)
class FileFacts:
    path: str
    name: str
    title: str
    category_key: str
    category_label: str
    summary: str
    headline: str
    why_it_matters: str
    learning_moment: str
    best_for: str
    difficulty: str
    depth_index: int
    tags: list[str]
    anchors: list[str]
    stats: dict[str, int]
    github_url: str
    featured: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def title_from_path(path: Path) -> str:
    stem = path.stem.replace("-", " ").replace("_", " ")
    title = re.sub(r"\s+", " ", stem).strip()
    if not title:
        return path.name
    return " ".join(word.upper() if word.isupper() else word.capitalize() for word in title.split())


def pick_category(relative_path: str) -> tuple[str, str]:
    matched = CATEGORY_DEFS[0]
    for category in CATEGORY_DEFS:
        if relative_path == category["match"] or relative_path.startswith(category["match"]):
            matched = category
    return matched["key"], matched["label"]


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def first_comment_block(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        if line.startswith("#"):
            cleaned = line.lstrip("#").strip()
            if cleaned:
                lines.append(cleaned)
            continue
        break
    return " ".join(lines)


def first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    match = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    return match[0]


def module_summary(path: Path, text: str) -> str:
    try:
        tree = ast.parse(text)
        doc = ast.get_docstring(tree)
    except SyntaxError:
        doc = None
    if doc:
        sentence = first_sentence(doc)
        if sentence:
            return sentence
    comment_block = first_comment_block(text)
    if comment_block:
        sentence = first_sentence(comment_block)
        if sentence:
            return sentence
    return f"{title_from_path(path)} is part of the velvet-python learning catalog."


def ast_counts(text: str) -> tuple[int, int, int, list[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0, 0, 0, []

    function_count = 0
    class_count = 0
    import_count = 0
    anchors: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function_count += 1
            if len(anchors) < 6:
                anchors.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_count += 1
            if len(anchors) < 6:
                anchors.append(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            import_count += 1

    return function_count, class_count, import_count, anchors


def line_counts(text: str) -> tuple[int, int, int]:
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    comment_lines = [line for line in lines if line.strip().startswith("#")]
    return len(lines), len(nonempty), len(comment_lines)


def infer_tags(path: str, summary: str, anchors: list[str]) -> list[str]:
    haystack = " ".join([path.lower(), summary.lower(), " ".join(name.lower() for name in anchors)])
    tags = [tag for keyword, tag in TAG_KEYWORDS.items() if keyword in haystack]
    if path.startswith("tests/"):
        tags.append("Testing")
    if path.startswith("src/"):
        tags.append("Core Engineering")
    if path.startswith("scripts/"):
        tags.append("Automation")
    if path.startswith("pyfiles/games/"):
        tags.append("Simulation")
    if path.startswith("pyfiles/more/"):
        tags.append("Applied Work")
    if "cli" in haystack:
        tags.append("Command Line")
    return sorted(dict.fromkeys(tags))


def difficulty_band(nonempty_lines: int, function_count: int, class_count: int) -> str:
    score = nonempty_lines + function_count * 10 + class_count * 18
    if score < 120:
        return "gentle"
    if score < 280:
        return "steady"
    if score < 520:
        return "deep"
    return "intensive"


def depth_index(nonempty_lines: int, function_count: int, class_count: int, tags: list[str]) -> int:
    score = nonempty_lines * 0.18 + function_count * 4.2 + class_count * 6.1 + len(tags) * 2.0
    return max(12, min(98, int(round(score))))


def choose_variant(key: str, options: list[str]) -> str:
    index = int(sha1(key.encode("utf-8")).hexdigest(), 16) % len(options)
    return options[index]


def narrative(category_key: str, title: str, summary: str, tags: list[str]) -> tuple[str, str, str]:
    tag_text = ", ".join(tags[:3]).lower() if tags else "practical Python"

    why_templates = {
        "core": [
            f"{title} is one of the places where the repository becomes accountable. It turns abstract learning into behavior you can rerun and inspect.",
            f"{title} matters because it shows how careful Python work carries its assumptions in the open instead of hiding them behind convenience.",
        ],
        "tests": [
            f"{title} matters because it teaches the repo by pressure. It shows what should hold steady when the code changes.",
            f"{title} is useful when you want to see how trust is earned: not by claims, but by repeatable checks and readable failures.",
        ],
        "games": [
            f"{title} keeps the learning physical. It lets an idea move, collide, and reveal itself instead of staying trapped in a static explanation.",
            f"{title} matters because some concepts become easier when they are felt as motion, timing, and interaction instead of only prose.",
        ],
        "foundations": [
            f"{title} is a good place to slow down and make a concept tangible. It keeps the language close to the math and the mechanics visible.",
            f"{title} matters because it teaches from the inside out: implementation first, then reflection, then comparison.",
        ],
        "scripts": [
            f"{title} shows how a useful Python tool grows beyond a notebook moment and becomes a repeatable workflow.",
            f"{title} matters because it turns {tag_text} into something you can invoke, compare, and hand to another person.",
        ],
    }

    learning_templates = [
        f"Start here when you want to see how {tag_text} is written plainly enough to study without losing the engineering edge.",
        f"Read this when you want a smaller room to think in: one file, one shape of problem, and enough detail to trace the choices.",
        f"This file rewards slow reading. The implementation is the lesson as much as the result.",
    ]

    best_for_templates = {
        "core": "best for readers moving from tutorials into reproducible systems work",
        "cli": "best for readers who learn by running commands and following outputs",
        "scripts": "best for readers turning one-off ideas into repeatable workflows",
        "tests": "best for readers who want to understand behavior through proof",
        "environments": "best for readers who need Python to behave across real machines",
        "games": "best for readers who understand faster when the idea becomes interactive",
        "applied": "best for readers exploring how Python travels into business and analysis",
        "labs": "best for readers trying to isolate one bug, one pattern, or one concept",
        "legacy-env": "best for readers studying environment tradeoffs and earlier design paths",
        "foundations": "best for readers building fluency from the language upward",
    }

    why = choose_variant(f"{title}:why", why_templates.get(category_key, why_templates["foundations"]))
    learning = choose_variant(f"{title}:learning", learning_templates)
    best_for = best_for_templates.get(category_key, "best for readers learning by tracing real code")

    return why, learning, best_for


def headline_from(summary: str, title: str) -> str:
    sentence = first_sentence(summary)
    if sentence and sentence != title:
        return sentence
    return f"{title} turns one slice of Python into something concrete enough to study."


def file_facts(path: Path) -> FileFacts:
    relative = path.relative_to(ROOT).as_posix()
    category_key, category_label = pick_category(relative)
    text = safe_read(path)
    summary = module_summary(path, text)
    function_count, class_count, import_count, anchors = ast_counts(text)
    total_lines, nonempty_lines, comment_lines = line_counts(text)
    tags = infer_tags(relative, summary, anchors)
    difficulty = difficulty_band(nonempty_lines, function_count, class_count)
    depth = depth_index(nonempty_lines, function_count, class_count, tags)
    why, learning, best_for = narrative(category_key, title_from_path(path), summary, tags)

    return FileFacts(
        path=relative,
        name=path.name,
        title=title_from_path(path),
        category_key=category_key,
        category_label=category_label,
        summary=summary,
        headline=headline_from(summary, title_from_path(path)),
        why_it_matters=why,
        learning_moment=learning,
        best_for=best_for,
        difficulty=difficulty,
        depth_index=depth,
        tags=tags,
        anchors=anchors,
        stats={
            "total_lines": total_lines,
            "nonempty_lines": nonempty_lines,
            "comment_lines": comment_lines,
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
        },
        github_url=f"{REPO_URL}/{relative}",
        featured=relative in FEATURED_PATHS,
    )


def discover_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if path.is_file() and not should_skip(path.relative_to(ROOT))
    )


def category_descriptions() -> dict[str, str]:
    return {category["key"]: category["description"] for category in CATEGORY_DEFS}


def track_payload(files: list[FileFacts]) -> list[dict[str, Any]]:
    category_counts: dict[str, int] = {}
    for file in files:
        category_counts[file.category_key] = category_counts.get(file.category_key, 0) + 1

    payload = []
    for track in TRACKS:
        count = sum(category_counts.get(category, 0) for category in track["categories"])
        payload.append(
            {
                "id": track["id"],
                "title": track["title"],
                "description": track["description"],
                "file_count": count,
                "categories": track["categories"],
            }
        )
    return payload


def featured_payload(files: list[FileFacts]) -> list[dict[str, Any]]:
    featured = [file for file in files if file.featured]
    if len(featured) < 10:
        supplemental = sorted(files, key=lambda item: item.depth_index, reverse=True)
        for file in supplemental:
            if file not in featured:
                featured.append(file)
            if len(featured) >= 12:
                break

    return [serialize_file(file) for file in featured[:12]]


def serialize_file(file: FileFacts) -> dict[str, Any]:
    return {
        "id": file.path.replace("/", "--").replace(".", "-"),
        "path": file.path,
        "name": file.name,
        "title": file.title,
        "category_key": file.category_key,
        "category_label": file.category_label,
        "summary": file.summary,
        "headline": file.headline,
        "why_it_matters": file.why_it_matters,
        "learning_moment": file.learning_moment,
        "best_for": file.best_for,
        "difficulty": file.difficulty,
        "depth_index": file.depth_index,
        "tags": file.tags,
        "anchors": file.anchors,
        "stats": file.stats,
        "github_url": file.github_url,
        "featured": file.featured,
    }


def build_payload() -> dict[str, Any]:
    files = [file_facts(path) for path in discover_files()]
    category_lookup = category_descriptions()

    total_lines = sum(file.stats["total_lines"] for file in files)
    total_functions = sum(file.stats["function_count"] for file in files)
    total_classes = sum(file.stats["class_count"] for file in files)
    category_counts: dict[str, int] = {}
    for file in files:
        category_counts[file.category_key] = category_counts.get(file.category_key, 0) + 1

    top_depth = sorted(files, key=lambda item: (item.depth_index, item.stats["nonempty_lines"]), reverse=True)

    return {
        "generated_at": utc_now(),
        "repository": {
            "name": "velvet-python",
            "owner": "Cazzy-Aporbo",
            "url": "https://github.com/Cazzy-Aporbo/velvet-python",
            "pages_hint": "Enable GitHub Pages from the docs/ folder on main.",
        },
        "stats": {
            "python_file_count": len(files),
            "total_lines": total_lines,
            "function_count": total_functions,
            "class_count": total_classes,
            "category_counts": category_counts,
        },
        "categories": [
            {
                "key": category["key"],
                "label": category["label"],
                "description": category_lookup[category["key"]],
                "file_count": category_counts.get(category["key"], 0),
            }
            for category in CATEGORY_DEFS
        ],
        "tracks": track_payload(files),
        "featured": featured_payload(files),
        "depth_leaders": [serialize_file(file) for file in top_depth[:8]],
        "files": [serialize_file(file) for file in files],
    }


def main() -> None:
    payload = build_payload()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {payload['stats']['python_file_count']} Python files.")


if __name__ == "__main__":
    main()
