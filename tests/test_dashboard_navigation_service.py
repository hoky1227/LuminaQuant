from __future__ import annotations

from apps.dashboard.services.dashboard_navigation import select_dashboard_view


class _FakeSidebar:
    def __init__(self, *, selected_view: str) -> None:
        self.selected_view = selected_view
        self.calls: list[tuple[str, list[str], int]] = []

    def radio(self, label: str, options: list[str], index: int = 0) -> str:
        self.calls.append((label, list(options), index))
        return self.selected_view


class _FakeStreamlit:
    def __init__(self, *, selected_view: str) -> None:
        self.sidebar = _FakeSidebar(selected_view=selected_view)


def test_select_dashboard_view_queries_sidebar_radio() -> None:
    streamlit = _FakeStreamlit(selected_view="Exact-Window Suite")

    selected = select_dashboard_view(
        streamlit=streamlit,
        label="Dashboard View",
        view_options=("Main Dashboard", "Exact-Window Suite"),
        index=0,
    )

    assert selected == "Exact-Window Suite"
    assert streamlit.sidebar.calls == [
        ("Dashboard View", ["Main Dashboard", "Exact-Window Suite"], 0)
    ]
