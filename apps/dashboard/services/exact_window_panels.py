"""Shared Streamlit render helpers for exact-window dashboard panels."""

from __future__ import annotations

import streamlit as st


def render_exact_window_visual_cockpit(
    context,
    selection,
    *,
    oos_scatter_figure,
    metric_heatmap_figure,
    rss_bar_figure,
    window_timeline_figure,
    coverage_timeline_figure,
    st_module=st,
) -> None:
    command_top_left, command_top_right = st_module.columns((1.3, 1.1))
    with command_top_left:
        st_module.subheader('Performance Map')
        scatter_fig = oos_scatter_figure(selection.summary_frame)
        if scatter_fig is None:
            st_module.info('No OOS scatter view available.')
        else:
            st_module.plotly_chart(scatter_fig, use_container_width=True)
    with command_top_right:
        st_module.subheader('Metric Heatmap')
        heatmap_fig = metric_heatmap_figure(selection.metric_matrix)
        if heatmap_fig is None:
            st_module.info('No metric heatmap available.')
        else:
            st_module.plotly_chart(heatmap_fig, use_container_width=True)

    command_bottom_left, command_bottom_mid, command_bottom_right = st_module.columns((1.0, 1.0, 1.15))
    with command_bottom_left:
        st_module.subheader('Memory by Timeframe')
        rss_fig = rss_bar_figure(selection.summary_frame)
        if rss_fig is None:
            st_module.info('No memory chart available.')
        else:
            st_module.plotly_chart(rss_fig, use_container_width=True)
    with command_bottom_mid:
        st_module.subheader('Window Timeline')
        window_fig = window_timeline_figure(context.summary)
        if window_fig is None:
            st_module.info('No window timeline available.')
        else:
            st_module.plotly_chart(window_fig, use_container_width=True)
    with command_bottom_right:
        st_module.subheader('Coverage Timeline')
        coverage_fig = coverage_timeline_figure(context.coverage_status)
        if coverage_fig is None:
            st_module.info('No coverage timeline available.')
        else:
            st_module.plotly_chart(coverage_fig, use_container_width=True)


def render_exact_window_control_strip(
    context,
    selection,
    *,
    panel_html,
    status_chip,
    st_module=st,
) -> None:
    control_left, control_mid, control_right = st_module.columns((1.2, 1.4, 1.4))
    with control_left:
        st_module.markdown(
            panel_html(
                'Execution profile',
                'Windowing, requested symbols/timeframes, and whether this bundle was custom-windowed.',
                [
                    status_chip('Custom windows', 'yes' if selection.execution_profile.get('custom_windows') else 'no'),
                    status_chip('Allow metals', 'yes' if selection.execution_profile.get('allow_metals') else 'no'),
                    status_chip('Requested TF', ', '.join(selection.execution_profile.get('requested_timeframes') or [])),
                ],
            ),
            unsafe_allow_html=True,
        )
    with control_mid:
        metals_blocker = context.bundle.get('followup_status', {}).get('metals_blocker_latest') or {}
        st_module.markdown(
            panel_html(
                'Metal / mixed-asset status',
                str(metals_blocker.get('reason') or 'Metals allowed when requested; dashboard now surfaces blocker state explicitly.'),
                [
                    status_chip('Requested', len(metals_blocker.get('requested_symbols') or selection.execution_profile.get('requested_symbols') or [])),
                    status_chip('Eligible', len(metals_blocker.get('eligible_symbols') or context.summary.get('eligible_symbols') or [])),
                    status_chip('Blocked metals', len(metals_blocker.get('blocked_metals') or [])),
                ],
            ),
            unsafe_allow_html=True,
        )
    with control_right:
        st_module.markdown(
            panel_html(
                'Memory discipline',
                'Heavy runs remain serialized and the page exposes run-level RSS evidence so the 8GB ceiling stays auditable.',
                [
                    status_chip('Decision peak', f"{float(context.decision.get('max_peak_rss_mib') or 0.0):.1f} MiB"),
                    status_chip('Root memory status', (context.memory_evidence or {}).get('status') or 'n/a'),
                    status_chip('Queue items', len(selection.queue_rows)),
                ],
            ),
            unsafe_allow_html=True,
        )


__all__ = [
    'render_exact_window_control_strip',
    'render_exact_window_visual_cockpit',
]
