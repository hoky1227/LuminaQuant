"""Shared Streamlit render helpers for exact-window dashboard panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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


def render_exact_window_timeframe_overview(
    context,
    selection,
    *,
    format_frame,
    timeframe_card_html,
    timeframe_sort_key,
    st_module=st,
) -> None:
    st_module.subheader('All Timeframes At a Glance')
    st_module.markdown(
        '<div class="exact-window-section-caption">Every timeframe, best candidate, risk/return profile, and rejection reason on one screen.</div>',
        unsafe_allow_html=True,
    )
    st_module.markdown(
        '<div class="exact-window-tf-grid">'
        + ''.join(
            timeframe_card_html(row)
            for row in sorted(
                context.timeframe_rows,
                key=lambda item: timeframe_sort_key(str(item.get('timeframe') or '')),
            )
        )
        + '</div>',
        unsafe_allow_html=True,
    )
    st_module.dataframe(format_frame(selection.summary_frame), use_container_width=True, hide_index=True)

    matrix_left, matrix_right = st_module.columns((1.4, 1.6))
    with matrix_left:
        st_module.subheader('Metric Matrix')
        if selection.metric_matrix.empty:
            st_module.info('No timeframe metric matrix available.')
        else:
            st_module.dataframe(
                format_frame(selection.metric_matrix.reset_index()),
                use_container_width=True,
                hide_index=True,
            )
    with matrix_right:
        st_module.subheader('Universe Coverage / Metals')
        if context.coverage_status.empty:
            st_module.info('No coverage table available.')
        else:
            st_module.dataframe(context.coverage_status, use_container_width=True, hide_index=True)


def render_exact_window_candidate_analysis(
    context,
    selection,
    *,
    candidate_pool_frame,
    strict_pass_frame,
    format_frame,
    top_candidates,
    family_mix_frame,
    fail_reason_summary,
    st_module=st,
) -> None:
    overview_left, overview_right = st_module.columns((3, 2))
    with overview_left:
        st_module.subheader('Selection / Promotion Overview')
        pool_frame = candidate_pool_frame(context.decision)
        if pool_frame.empty:
            st_module.info('No candidate-pool rows are currently saved.')
        else:
            st_module.dataframe(format_frame(pool_frame), use_container_width=True, hide_index=True)
    with overview_right:
        st_module.subheader('Strict Pass Anchor')
        strict_frame = strict_pass_frame(context.decision)
        if strict_frame.empty:
            st_module.info('No strict-pass strategy saved.')
        else:
            st_module.dataframe(format_frame(strict_frame), use_container_width=True, hide_index=True)

    st_module.subheader('Candidate Leaderboards')
    board_left, board_mid, board_right = st_module.columns(3)
    with board_left:
        st_module.caption('Top by OOS quality')
        frame = (
            selection.candidate_scope.sort_values(['oos_sharpe', 'oos_return'], ascending=[False, False])
            if not selection.candidate_scope.empty
            else pd.DataFrame()
        )
        st_module.dataframe(
            format_frame(
                top_candidates(
                    frame,
                    columns=[
                        'timeframe',
                        'asset_mix',
                        'strategy',
                        'name',
                        'symbols',
                        'oos_return',
                        'oos_sharpe',
                        'oos_sortino',
                        'oos_calmar',
                        'oos_mdd',
                        'oos_pbo',
                    ],
                )
            ),
            use_container_width=True,
            hide_index=True,
        )
    with board_mid:
        st_module.caption('Top by validation')
        frame = (
            selection.candidate_scope.sort_values(['val_sharpe', 'val_return'], ascending=[False, False])
            if not selection.candidate_scope.empty
            else pd.DataFrame()
        )
        st_module.dataframe(
            format_frame(
                top_candidates(
                    frame,
                    columns=[
                        'timeframe',
                        'asset_mix',
                        'strategy',
                        'name',
                        'symbols',
                        'val_return',
                        'val_sharpe',
                        'val_pbo',
                        'oos_return',
                        'oos_sharpe',
                    ],
                )
            ),
            use_container_width=True,
            hide_index=True,
        )
    with board_right:
        st_module.caption('Top by execution realism')
        frame = (
            selection.candidate_scope.sort_values(
                ['oos_trades', 'oos_win_rate', 'oos_avg_trade'],
                ascending=[False, False, False],
            )
            if not selection.candidate_scope.empty
            else pd.DataFrame()
        )
        st_module.dataframe(
            format_frame(
                top_candidates(
                    frame,
                    columns=[
                        'timeframe',
                        'asset_mix',
                        'strategy',
                        'name',
                        'symbols',
                        'oos_trades',
                        'oos_turnover',
                        'oos_win_rate',
                        'oos_avg_trade',
                        'rejects',
                    ],
                )
            ),
            use_container_width=True,
            hide_index=True,
        )

    family_frame, mix_frame = family_mix_frame(context.details_frame)
    mix_left, mix_right = st_module.columns(2)
    with mix_left:
        st_module.subheader('Family / Timeframe Distribution')
        if family_frame.empty:
            st_module.info('No family distribution available.')
        else:
            st_module.dataframe(family_frame, use_container_width=True, hide_index=True)
    with mix_right:
        st_module.subheader('Asset-Mix Distribution')
        if mix_frame.empty:
            st_module.info('No asset-mix distribution available.')
        else:
            st_module.dataframe(mix_frame, use_container_width=True, hide_index=True)

    fail_by_reason, fail_by_timeframe, fail_proposals = fail_reason_summary(context.bundle)
    fail_left, fail_mid, fail_right = st_module.columns((1.2, 1.5, 2.1))
    with fail_left:
        st_module.subheader('Reject Reasons')
        if fail_by_reason.empty:
            st_module.info('No fail analysis available.')
        else:
            st_module.dataframe(fail_by_reason, use_container_width=True, hide_index=True)
    with fail_mid:
        st_module.subheader('Rejects by Timeframe')
        if fail_by_timeframe.empty:
            st_module.info('No timeframe breakdown available.')
        else:
            st_module.dataframe(fail_by_timeframe, use_container_width=True, hide_index=True)
    with fail_right:
        st_module.subheader('Next-Step Proposals')
        if fail_proposals.empty:
            st_module.info('No proposals generated.')
        else:
            st_module.dataframe(fail_proposals, use_container_width=True, hide_index=True)

    if context.next_iteration:
        with st_module.expander('Next Iteration Triage', expanded=True):
            st_module.json(context.next_iteration)


def render_exact_window_selected_timeframe_summary(
    selection,
    *,
    card_grid_renderer,
    format_value,
    split_cockpit_html,
    st_module=st,
) -> None:
    st_module.subheader(f'Selected Timeframe Deep Dive — {selection.selected_timeframe}')
    selected_metrics = [
        ('Evaluated', format_value(selection.selected_row.get('evaluated_count'), 'int')),
        ('Candidate Pool', format_value(selection.selected_row.get('candidate_pool_strategy_count'), 'int')),
        ('BTC Beating', format_value(selection.selected_row.get('btc_beating_strategy_count'), 'int')),
        (
            'Peak RSS',
            f"{float((selection.selected_row.get('memory_evidence') or {}).get('peak_rss_mib') or 0.0):.1f} MiB",
        ),
        ('Validation Score', format_value(selection.selected_best.get('validation_score'), 'float3')),
        (
            'Timeframe Selection',
            format_value(selection.selected_best.get('timeframe_selection_score'), 'float3'),
        ),
        ('OOS Trades', format_value(selection.selected_oos.get('trade_count'), 'int')),
        ('OOS PBO', format_value(selection.selected_oos.get('pbo'), 'float3')),
    ]
    card_grid_renderer(
        [
            (
                label,
                value,
                selection.selected_best.get('strategy_class') or selection.selected_best.get('name') or '—',
            )
            for label, value in selected_metrics
        ]
    )
    st_module.caption('Train / validation / OOS cockpit')
    st_module.markdown(split_cockpit_html(selection.selected_best), unsafe_allow_html=True)


def render_exact_window_overview_tab(
    selection,
    *,
    best_row_snapshot,
    format_frame,
    st_module=st,
) -> None:
    snapshot = best_row_snapshot(selection.selected_best)
    if snapshot.empty:
        st_module.info('No best row available for this timeframe.')
        return

    st_module.dataframe(format_frame(snapshot), use_container_width=True, hide_index=True)
    left, right = st_module.columns((3, 2))
    with left:
        st_module.write('Symbols')
        st_module.write(selection.selected_best.get('symbols') or [])
        st_module.write('Rejection reasons')
        st_module.write(selection.selected_best.get('rejection_reasons') or [])
        st_module.write('Hard reject reasons')
        st_module.json(selection.selected_best.get('hard_reject_reasons') or {})
    with right:
        st_module.write('Parameters')
        st_module.json(selection.selected_best.get('params') or {})
        st_module.write('Metadata')
        st_module.json(selection.selected_best.get('metadata') or {})


def render_exact_window_deployment_tab(
    context,
    selection,
    *,
    format_frame,
    st_module=st,
) -> None:
    if selection.deployment_frame.empty:
        st_module.info('No deployment panel available.')
        return

    if selection.deployment_artifact:
        st_module.caption(
            f"Primary deployment artifact: {selection.deployment_artifact.get('scenario_id')} · "
            f"{selection.deployment_artifact.get('label')} · generated {selection.deployment_artifact.get('generated_at')}"
        )
    st_module.dataframe(format_frame(selection.deployment_frame), use_container_width=True, hide_index=True)
    dep_tab_left, dep_tab_right = st_module.columns((1.0, 1.6))
    with dep_tab_left:
        st_module.caption('Blend metrics by split')
        st_module.dataframe(
            format_frame(selection.deployment_split_metrics_frame),
            use_container_width=True,
            hide_index=True,
        )
    with dep_tab_right:
        if selection.deployment_oos_curve.empty:
            st_module.info('No OOS deployment blend curve available.')
        else:
            st_module.caption('Equal-weight OOS blend curve')
            st_module.line_chart(selection.deployment_oos_curve, use_container_width=True)
    if not selection.deployment_scenario_frame.empty:
        scenario_left, scenario_right = st_module.columns((1.2, 1.6))
        with scenario_left:
            st_module.caption('Deployment scenario matrix')
            st_module.dataframe(
                format_frame(selection.deployment_scenario_frame),
                use_container_width=True,
                hide_index=True,
            )
        with scenario_right:
            if selection.deployment_scenario_curve.empty:
                st_module.info('No deployment scenario comparison curve available.')
            else:
                st_module.caption('Scenario comparison — OOS cumulative return')
                st_module.line_chart(selection.deployment_scenario_curve, use_container_width=True)
    if selection.deployment_artifact or selection.deployment_scenarios_artifact:
        with st_module.expander('Deployment artifact paths', expanded=False):
            followup_root = context.bundle.get('followup_status_root')
            root = Path(str(followup_root)) if followup_root else None
            st_module.json(
                {
                    'deployment_combo_json': root and str(root / 'deployment_combo_latest.json'),
                    'deployment_combo_md': root and str(root / 'deployment_combo_latest.md'),
                    'deployment_scenarios_json': root and str(root / 'deployment_scenarios_latest.json'),
                    'deployment_scenarios_md': root and str(root / 'deployment_scenarios_latest.md'),
                }
            )


def render_exact_window_leaderboards_tab(
    selection,
    *,
    format_frame,
    st_module=st,
) -> None:
    if selection.candidate_scope.empty:
        st_module.info('No leaderboard rows available for this filter.')
        return

    st_module.dataframe(format_frame(selection.candidate_scope), use_container_width=True, hide_index=True)


def render_exact_window_time_series_tab(
    selection,
    *,
    chart_frame,
    split_order,
    stream_frame,
    st_module=st,
) -> None:
    cumulative = chart_frame(selection.selected_best, 'cumulative_return')
    raw_returns = chart_frame(selection.selected_best, 'return')
    drawdown = chart_frame(selection.selected_best, 'drawdown')
    if cumulative.empty:
        st_module.info('Return streams not available for this timeframe.')
        return

    top, bottom = st_module.columns(2)
    with top:
        st_module.caption('Cumulative return by split')
        st_module.line_chart(cumulative, use_container_width=True)
    with bottom:
        st_module.caption('Drawdown by split')
        st_module.line_chart(drawdown, use_container_width=True)
    st_module.caption('Raw periodic return by split')
    st_module.line_chart(raw_returns, use_container_width=True)
    with st_module.expander('Raw stream preview', expanded=False):
        preview = pd.concat(
            [
                stream_frame((selection.selected_best.get('return_streams') or {}).get(split) or [], split)
                for split in split_order
            ],
            ignore_index=True,
        )
        st_module.dataframe(preview.tail(100), use_container_width=True, hide_index=True)


def render_exact_window_split_metrics_tab(
    selection,
    *,
    split_metrics_frame,
    st_module=st,
) -> None:
    st_module.dataframe(split_metrics_frame(selection.selected_best), use_container_width=True, hide_index=True)


def render_exact_window_portfolio_tab(
    summary,
    *,
    format_frame,
    portfolio_chart_frame,
    portfolio_metrics_frame,
    portfolio_weights_frame,
    st_module=st,
) -> None:
    portfolio_metrics = portfolio_metrics_frame(summary)
    portfolio_weights = portfolio_weights_frame(summary)
    port_left, port_right = st_module.columns((2, 3))
    with port_left:
        if portfolio_metrics.empty:
            st_module.info('No portfolio metrics available.')
        else:
            st_module.dataframe(portfolio_metrics, use_container_width=True, hide_index=True)
    with port_right:
        if portfolio_weights.empty:
            st_module.info('No portfolio weights available.')
        else:
            st_module.dataframe(format_frame(portfolio_weights), use_container_width=True, hide_index=True)
    portfolio_curve = portfolio_chart_frame(summary, 'cumulative_return')
    portfolio_dd = portfolio_chart_frame(summary, 'drawdown')
    if not portfolio_curve.empty:
        port_curve_col, port_dd_col = st_module.columns(2)
        with port_curve_col:
            st_module.caption('Portfolio cumulative return by split')
            st_module.line_chart(portfolio_curve, use_container_width=True)
        with port_dd_col:
            st_module.caption('Portfolio drawdown by split')
            st_module.line_chart(portfolio_dd, use_container_width=True)


def render_exact_window_monthly_hurdles_tab(
    selection,
    *,
    format_frame,
    monthly_hurdle_frame,
    st_module=st,
) -> None:
    hurdle_frame = monthly_hurdle_frame(selection.selected_best)
    if hurdle_frame.empty:
        st_module.info('No monthly hurdle data available.')
        return

    st_module.dataframe(format_frame(hurdle_frame), use_container_width=True, hide_index=True)
    hurdle_chart = hurdle_frame[['month', 'split', 'strategy_return', 'threshold', 'btc_buy_hold_return']].copy()
    hurdle_chart['series'] = hurdle_chart['split'] + ' strategy'
    strategy_pivot = hurdle_chart.pivot_table(index='month', columns='series', values='strategy_return', aggfunc='last')
    threshold_pivot = hurdle_chart.pivot_table(index='month', columns='split', values='threshold', aggfunc='last')
    threshold_pivot.columns = [f'{column} threshold' for column in threshold_pivot.columns]
    btc_pivot = hurdle_chart.pivot_table(index='month', columns='split', values='btc_buy_hold_return', aggfunc='last')
    btc_pivot.columns = [f'{column} btc' for column in btc_pivot.columns]
    plot_frame = pd.concat([strategy_pivot, threshold_pivot, btc_pivot], axis=1).sort_index()
    st_module.caption('Monthly hurdle comparison')
    st_module.line_chart(plot_frame, use_container_width=True)


def render_exact_window_universe_tab(
    context,
    *,
    format_frame,
    top_candidates,
    st_module=st,
) -> None:
    st_module.write('Coverage table')
    if context.coverage_status.empty:
        st_module.info('No coverage table available.')
    else:
        st_module.dataframe(context.coverage_status, use_container_width=True, hide_index=True)
    mixed_scope = (
        context.details_frame[context.details_frame['asset_mix'] == 'crypto-metal mix']
        if not context.details_frame.empty
        else pd.DataFrame()
    )
    metal_scope = (
        context.details_frame[context.details_frame['asset_mix'].isin(['pure metal', 'crypto-metal mix'])]
        if not context.details_frame.empty
        else pd.DataFrame()
    )
    uni_left, uni_right = st_module.columns(2)
    with uni_left:
        st_module.caption('Mixed-asset candidates')
        if mixed_scope.empty:
            st_module.info('No saved crypto-metal candidates in current details bundle.')
        else:
            st_module.dataframe(
                format_frame(
                    top_candidates(
                        mixed_scope.sort_values(['oos_sharpe', 'oos_return'], ascending=[False, False]),
                        columns=[
                            'timeframe',
                            'strategy',
                            'name',
                            'symbols',
                            'val_return',
                            'oos_return',
                            'oos_sharpe',
                            'oos_pbo',
                            'oos_trades',
                        ],
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )
    with uni_right:
        st_module.caption('Metals blocker / notes')
        st_module.json(context.bundle.get('followup_status', {}).get('metals_blocker_latest') or {})
        if not metal_scope.empty:
            st_module.caption('Metal-linked candidates')
            st_module.dataframe(
                format_frame(
                    top_candidates(
                        metal_scope.sort_values(['oos_sharpe', 'oos_return'], ascending=[False, False]),
                        columns=[
                            'timeframe',
                            'asset_mix',
                            'strategy',
                            'name',
                            'symbols',
                            'oos_return',
                            'oos_sharpe',
                            'oos_pbo',
                            'rejects',
                        ],
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_exact_window_followup_tab(
    context,
    *,
    format_frame,
    st_module=st,
) -> None:
    if context.followup_frame.empty:
        st_module.info('No follow-up runs saved.')
        return

    st_module.dataframe(format_frame(context.followup_frame), use_container_width=True, hide_index=True)
    status_chart = (
        context.followup_frame.assign(stage_name=context.followup_frame['stage'])
        .set_index('stage_name')[['peak_rss_mib']]
        .dropna()
    )
    if not status_chart.empty:
        st_module.caption('Follow-up peak RSS (MiB)')
        st_module.bar_chart(status_chart, use_container_width=True)


def render_exact_window_registry_tab(
    context,
    *,
    format_frame,
    st_module=st,
) -> None:
    archive_payload = context.bundle.get('followup_status', {}).get('backtest_log_archive_latest') or {}
    if context.registry_frame.empty:
        st_module.info('No run registry entries saved.')
    else:
        st_module.dataframe(format_frame(context.registry_frame), use_container_width=True, hide_index=True)
    with st_module.expander('Archived log ledger', expanded=False):
        st_module.json(archive_payload)


def render_exact_window_rejects_tab(
    context,
    selection,
    *,
    reject_reason_frame,
    st_module=st,
) -> None:
    frame = reject_reason_frame(selection.selected_row)
    if frame.empty:
        st_module.info('No reject-reason counts available.')
    else:
        st_module.dataframe(frame, use_container_width=True, hide_index=True)
    with st_module.expander('Root fail analysis', expanded=False):
        st_module.json(context.bundle.get('fail_analysis') or {})


def render_exact_window_diagnostics_tab(
    context,
    selection,
    *,
    coverage_frame,
    st_module=st,
) -> None:
    path_frame = pd.DataFrame(
        [
            {
                'summary_path': selection.selected_row.get('summary_path'),
                'details_path': selection.selected_row.get('details_path'),
                'fail_analysis_path': selection.selected_row.get('fail_analysis_path'),
                'source_summary_path': selection.selected_best.get('source_summary_path'),
                'source_details_path': selection.selected_best.get('source_details_path'),
            }
        ]
    )
    diag_left, diag_right = st_module.columns((2, 3))
    with diag_left:
        st_module.write('Artifact Paths')
        st_module.dataframe(path_frame, use_container_width=True, hide_index=True)
        st_module.write('Selected timeframe memory evidence')
        st_module.json(selection.selected_row.get('memory_evidence') or {})
        if context.memory_evidence:
            with st_module.expander('Root latest memory evidence', expanded=False):
                st_module.json(context.memory_evidence)
    with diag_right:
        st_module.write('Coverage')
        coverage_status = coverage_frame(context.summary)
        if coverage_status.empty:
            st_module.info('No coverage table available.')
        else:
            st_module.dataframe(coverage_status, use_container_width=True, hide_index=True)
        with st_module.expander('Execution Profile', expanded=False):
            st_module.json(context.summary.get('execution_profile') or {})
        with st_module.expander('Windows', expanded=False):
            st_module.json(context.summary.get('windows') or {})
        with st_module.expander('Bundle Paths', expanded=False):
            st_module.json(
                {
                    'summary_generated_at': context.summary.get('generated_at'),
                    'latest_pointer': context.bundle.get('latest_pointer'),
                    'run_root': context.bundle.get('run_root'),
                    'root_paths': context.bundle.get('paths'),
                    'followup_status_root': context.bundle.get('followup_status_root'),
                }
            )


__all__ = [
    'render_exact_window_candidate_analysis',
    'render_exact_window_control_strip',
    'render_exact_window_deployment_tab',
    'render_exact_window_diagnostics_tab',
    'render_exact_window_followup_tab',
    'render_exact_window_leaderboards_tab',
    'render_exact_window_monthly_hurdles_tab',
    'render_exact_window_overview_tab',
    'render_exact_window_portfolio_tab',
    'render_exact_window_registry_tab',
    'render_exact_window_rejects_tab',
    'render_exact_window_selected_timeframe_summary',
    'render_exact_window_split_metrics_tab',
    'render_exact_window_time_series_tab',
    'render_exact_window_timeframe_overview',
    'render_exact_window_universe_tab',
    'render_exact_window_visual_cockpit',
]
