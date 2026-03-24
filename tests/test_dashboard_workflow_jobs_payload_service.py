from __future__ import annotations

from lumina_quant.dashboard import workflow_jobs_service


def test_load_recent_workflow_jobs_payload_short_circuits_without_dsn() -> None:
    payload = workflow_jobs_service.load_recent_workflow_jobs_payload(dsn="", limit=5)

    assert payload["status"] == "missing_dsn"
    assert payload["jobs"] == []
