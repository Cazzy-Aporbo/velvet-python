"""Streamlit dashboard for managing environments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

try:
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
except ModuleNotFoundError as exc:
    streamlit_error = exc
else:
    streamlit_error = None

if streamlit_error:
    # This module can be imported for static analysis without Streamlit installed.
    # Keep it lightweight for learners and CI checks.
    st = None
    pd = None
    go = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.environment_manager import EnvironmentManager, VirtualEnvironment
except ModuleNotFoundError:
    EnvironmentManager = None
    VirtualEnvironment = None


def _safe_json(obj):
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def _init_style():
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top, #f6f6ff, #f9f5ff 45%, #f0f6ff);
            }
            .block-container { padding-top: 2rem; }
            h1 { color: #2f2a45; }
            h2, h3 { color: #3a3552; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(manager: EnvironmentManager) -> None:
    st.markdown("## Environment Overview")

    if not manager.environments:
        st.info("No environments yet. Open **Create Environment** to get started.")
        return

    env_rows = []
    for env in manager.environments.values():
        env_rows.append(
            {
                "name": env.name,
                "python_version": env.python_version,
                "path": str(env.path),
                "created_at": env.created_at.isoformat(),
                "packages": len(env.packages),
                "active": "yes" if env.is_active else "no",
            }
        )
    st.dataframe(pd.DataFrame(env_rows), use_container_width=True)

    versions = pd.Series([row["python_version"] for row in env_rows]).value_counts()
    if not versions.empty:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=versions.index,
                    values=versions.values,
                    hole=0.35,
                )
            ]
        )
        fig.update_layout(title="Python Version Distribution")
        st.plotly_chart(fig, use_container_width=True)


def _render_create(manager: EnvironmentManager) -> None:
    st.markdown("## Create Environment")
    with st.form("create_env_form"):
        env_name = st.text_input("Environment name", value="project-env")
        project_dir = st.text_input(
            "Project directory",
            value=str(Path.cwd()),
        )
        requirements_file = st.text_input("requirements.txt (optional)", value="")
        dev_requirements = st.text_input("requirements-dev.txt (optional)", value="")
        submit = st.form_submit_button("Create")

    if submit:
        req_path = Path(requirements_file) if requirements_file else None
        dev_req_path = Path(dev_requirements) if dev_requirements else None
        try:
            env = manager.create_environment(
                name=env_name.strip(),
                project_dir=Path(project_dir),
                requirements=req_path,
                dev_requirements=dev_req_path,
            )
            st.success(f"Created {env.name} at {env.path}")
            st.json(_safe_json(env))
            st.rerun()
        except Exception as exc:  # pragma: no cover - CLI style integration message
            st.error(f"Create failed: {exc}")


def _render_compare(manager: EnvironmentManager) -> None:
    st.markdown("## Compare Environments")
    env_names = sorted(manager.environments.keys())
    if len(env_names) < 2:
        st.warning("Create at least two environments before comparing.")
        return

    a = st.selectbox("First environment", env_names, key="cmp_a")
    b = st.selectbox("Second environment", env_names, key="cmp_b")

    if st.button("Compare"):
        if a == b:
            st.error("Choose two different environments.")
            return
        manager.compare_environments(a, b)


def _render_tools() -> None:
    st.markdown("## Tooling Notes")
    st.write(
        """
        Practical recommendations:
        - Prefer `requirements.txt` in repositories.
        - Keep `.venv` in `.gitignore`.
        - Pin versions to stabilize reproducibility.
        """
    )


def run():
    if st is None:
        print("Streamlit is not installed. Install with `pip install streamlit`.")
        return

    if EnvironmentManager is None:
        st.error("Unable to import EnvironmentManager from src.environment_manager.")
        return

    st.set_page_config(page_title="Velvet Python", page_icon="🧠", layout="wide")
    _init_style()

    if "manager" not in st.session_state:
        st.session_state["manager"] = EnvironmentManager()

    manager: EnvironmentManager = st.session_state["manager"]
    st.title("Velvet Python Environment Lab")
    st.caption("A practical environment manager for deterministic local tooling.")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Create Environment", "Compare Environments", "Tooling Notes"],
    )

    if page == "Overview":
        _render_overview(manager)
    elif page == "Create Environment":
        _render_create(manager)
    elif page == "Compare Environments":
        _render_compare(manager)
    else:
        _render_tools()


if __name__ == "__main__":
    if st is None:
        # Avoid crashing imports in non-UI test environments.
        print("Run with: streamlit run 01-environments/app.py")
    else:
        run()
