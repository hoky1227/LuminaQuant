"""Shared render helpers for the exact-window dashboard."""

from __future__ import annotations

import html

import streamlit as st


def _card_html(label: str, value: str, sub: str) -> str:
    return (
        '<div class="exact-window-card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}</div>'
        "</div>"
    )


def render_exact_window_card_grid(cards: list[tuple[str, str, str]]) -> None:
    st.markdown(
        '<div class="exact-window-card-grid">' + "".join(_card_html(*card) for card in cards) + "</div>",
        unsafe_allow_html=True,
    )


__all__ = ["render_exact_window_card_grid"]
