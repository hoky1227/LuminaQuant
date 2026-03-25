from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "workflow_jobs_view.py"
SPEC = importlib.util.spec_from_file_location("dashboard_workflow_jobs_view_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load workflow jobs view module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
render_workflow_jobs_section = MODULE.render_workflow_jobs_section


class _FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def subheader(self, value: str) -> None:
        self.calls.append(("subheader", value))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def dataframe(self, value, use_container_width: bool = False) -> None:
        self.calls.append(("dataframe", list(value.columns)))

    def selectbox(self, label: str, options: list[str], key: str):
        self.calls.append(("selectbox", (label, list(options), key)))
        return options[0]

    def caption(self, value: str) -> None:
        self.calls.append(("caption", value))

    def text_area(self, label: str, value: str, height: int, key: str) -> None:
        self.calls.append(("text_area", (label, value, height, key)))


def test_render_workflow_jobs_section_reports_empty_state() -> None:
    fake_st = _FakeStreamlit()

    render_workflow_jobs_section(
        streamlit=fake_st,
        db_path="postgres://lumina",
        refresh_counter=1,
        load_workflow_jobs=lambda db_path, refresh_counter=0: pd.DataFrame(),
        render_active_workflow_job_controls=lambda **kwargs: None,
        tail_text_file=lambda path, max_chars=0: "",
    )

    assert ("subheader", "Workflow Jobs") in fake_st.calls
    assert ("info", "No workflow jobs recorded yet.") in fake_st.calls


def test_render_workflow_jobs_section_renders_table_controls_and_log_viewer() -> None:
    fake_st = _FakeStreamlit()
    calls: list[tuple[str, object]] = []
    workflow_jobs = pd.DataFrame(
        [
            {
                "job_id": "job-1",
                "started_at": "2026-03-25T00:00:00Z",
                "workflow": "backtest",
                "status": "RUNNING",
                "requested_mode": "paper",
                "strategy": "Alpha",
                "pid": 123,
                "run_id": "run-1",
                "exit_code": None,
                "command_json": '["uv", "run"]',
                "log_path": "/tmp/job.log",
            }
        ]
    )

    render_workflow_jobs_section(
        streamlit=fake_st,
        db_path="postgres://lumina",
        refresh_counter=7,
        load_workflow_jobs=lambda db_path, refresh_counter=0: workflow_jobs,
        render_active_workflow_job_controls=lambda **kwargs: calls.append(("controls", kwargs["db_path"])),
        tail_text_file=lambda path, max_chars=0: f"tail:{path}:{max_chars}",
    )

    assert ("controls", "postgres://lumina") in calls
    assert any(call[0] == "dataframe" for call in fake_st.calls)
    assert ("caption", "Log path: /tmp/job.log") in fake_st.calls
    assert any(call[0] == "text_area" and call[1][1] == "tail:/tmp/job.log:25000" for call in fake_st.calls)
