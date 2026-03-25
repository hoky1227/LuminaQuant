"""Workflow-jobs rendering helpers extracted from the Streamlit dashboard app."""

from __future__ import annotations

from typing import Any


def render_workflow_job_log_viewer(
    *,
    streamlit: Any,
    workflow_jobs,
    tail_text_file,
) -> None:
    log_job_id = streamlit.selectbox(
        "Job Log Viewer",
        workflow_jobs["job_id"].astype(str).tolist(),
        key="workflow_log_viewer_job",
    )
    log_row = workflow_jobs[workflow_jobs["job_id"].astype(str) == str(log_job_id)].iloc[0]
    streamlit.caption(f"Log path: {log_row.get('log_path')}")
    streamlit.text_area(
        "Job Log Tail",
        value=tail_text_file(str(log_row.get("log_path") or ""), max_chars=25000),
        height=260,
        key="workflow_log_tail_view",
    )


def render_workflow_jobs_section(
    *,
    streamlit: Any,
    db_path,
    refresh_counter,
    load_workflow_jobs,
    render_active_workflow_job_controls,
    tail_text_file,
) -> None:
    streamlit.subheader("Workflow Jobs")
    workflow_jobs = load_workflow_jobs(db_path, refresh_counter=refresh_counter)
    if workflow_jobs.empty:
        streamlit.info("No workflow jobs recorded yet.")
        return

    jobs_view = workflow_jobs.copy()
    jobs_view["command"] = jobs_view["command_json"].fillna("").astype(str).str.slice(0, 120)
    streamlit.dataframe(
        jobs_view[
            [
                "started_at",
                "workflow",
                "status",
                "requested_mode",
                "strategy",
                "pid",
                "run_id",
                "exit_code",
                "command",
            ]
        ],
        use_container_width=True,
    )

    active_jobs = workflow_jobs[workflow_jobs["status"].isin(["RUNNING", "STOP_REQUESTED"])].copy()
    if not active_jobs.empty:
        render_active_workflow_job_controls(db_path=db_path, active_jobs=active_jobs)

    render_workflow_job_log_viewer(
        streamlit=streamlit,
        workflow_jobs=workflow_jobs,
        tail_text_file=tail_text_file,
    )
