from __future__ import annotations

from pathlib import Path

from scripts.build_pages_catalog import OUTPUT, ROOT, build_payload, discover_files, file_facts


def test_discover_files_finds_python_catalog_inputs() -> None:
    files = discover_files()
    relative_paths = {path.relative_to(ROOT).as_posix() for path in files}

    assert "src/pipeline.py" in relative_paths
    assert "scripts/run_experiments.py" in relative_paths
    assert "tests/test_pipeline_contracts.py" in relative_paths
    assert "docs/app.js" not in relative_paths


def test_file_facts_exposes_learning_metadata_for_core_file() -> None:
    facts = file_facts(ROOT / "src" / "pipeline.py")

    assert facts.category_key == "core"
    assert facts.category_label == "Core Systems"
    assert facts.depth_index >= 12
    assert facts.github_url.endswith("src/pipeline.py")
    assert facts.stats["function_count"] >= 1
    assert facts.summary
    assert facts.why_it_matters
    assert facts.learning_moment


def test_build_payload_is_consistent_with_generated_catalog_output() -> None:
    payload = build_payload()

    assert payload["repository"]["name"] == "velvet-python"
    assert payload["stats"]["python_file_count"] == len(payload["files"])
    assert payload["tracks"]
    assert payload["featured"]
    assert any(file["path"] == "src/pipeline.py" for file in payload["files"])
    assert any(category["key"] == "core" for category in payload["categories"])


def test_catalog_output_file_exists_and_is_inside_docs() -> None:
    assert OUTPUT == ROOT / "docs" / "catalog.json"
    assert Path(OUTPUT).parent.name == "docs"
