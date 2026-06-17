import streamlit as st
import plotly.express as px


COLORS = {
    "primary": "#cc785c",
    "primary_active": "#a9583e",
    "ink": "#141413",
    "body": "#3d3d3a",
    "muted": "#6c6a64",
    "muted_soft": "#8e8b82",
    "hairline": "#e6dfd8",
    "hairline_soft": "#ebe6df",
    "canvas": "#faf9f5",
    "surface_soft": "#f5f0e8",
    "surface_card": "#efe9de",
    "surface_dark": "#181715",
    "surface_dark_elevated": "#252320",
    "on_dark": "#faf9f5",
    "on_dark_soft": "#a09d96",
    "accent_teal": "#5db8a6",
    "accent_amber": "#e8a55a",
    "success": "#5db872",
    "warning": "#d4a017",
    "error": "#c64545",
}

TYPE_COLORS = {
    1: "#0057ff",
    2: "#00b050",
    3: "#ff9900",
    4: "#ff1f1f",
    5: "#a100ff",
    6: "#00a6d6",
    7: "#ff4fa3",
    8: "#111111",
}

PLOTLY_COLORWAY = list(TYPE_COLORS.values())


def apply_theme():
    px.defaults.color_discrete_sequence = PLOTLY_COLORWAY
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --rp-primary: {COLORS["primary"]};
            --rp-primary-active: {COLORS["primary_active"]};
            --rp-ink: {COLORS["ink"]};
            --rp-body: {COLORS["body"]};
            --rp-muted: {COLORS["muted"]};
            --rp-hairline: {COLORS["hairline"]};
            --rp-canvas: {COLORS["canvas"]};
            --rp-surface-soft: {COLORS["surface_soft"]};
            --rp-surface-card: {COLORS["surface_card"]};
            --rp-surface-dark: {COLORS["surface_dark"]};
            --rp-on-dark: {COLORS["on_dark"]};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background: var(--rp-canvas);
            color: var(--rp-body);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        [data-testid="stHeader"] {{
            background: rgba(250, 249, 245, 0.92);
            border-bottom: 1px solid var(--rp-hairline);
        }}

        [data-testid="stSidebar"] {{
            background: var(--rp-surface-dark);
            border-right: 1px solid #252320;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--rp-on-dark);
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 8px;
            padding: 4px 8px;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
            background: var(--rp-surface-dark-elevated) !important;
            color: var(--rp-primary) !important;
            border: 1px solid #3a3731 !important;
            border-radius: 8px;
            font-weight: 600;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {{
            border-color: var(--rp-primary) !important;
            box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.22) !important;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
            background: var(--rp-surface-dark-elevated) !important;
            color: var(--rp-primary) !important;
            border: 1px solid #3a3731 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {{
            background: var(--rp-primary) !important;
            border-color: var(--rp-primary) !important;
            color: white !important;
        }}

        [data-testid="stSidebar"] [data-testid="stNumberInput"] svg {{
            color: var(--rp-primary) !important;
            fill: var(--rp-primary) !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: var(--rp-surface-dark-elevated) !important;
            border: 1px solid #3a3731 !important;
            border-radius: 8px;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: var(--rp-primary) !important;
            font-weight: 600;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] svg {{
            color: var(--rp-primary) !important;
            fill: var(--rp-primary) !important;
        }}

        h1, h2, h3 {{
            font-family: "Cormorant Garamond", "Times New Roman", serif;
            font-weight: 500;
            color: var(--rp-ink);
            letter-spacing: 0;
        }}

        h1 {{
            font-size: 52px;
            line-height: 1.05;
        }}

        h2 {{
            font-size: 38px;
            line-height: 1.12;
        }}

        h3 {{
            font-size: 28px;
            line-height: 1.2;
        }}

        p, li, div, label {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        hr {{
            border-color: var(--rp-hairline);
        }}

        [data-testid="stMarkdownContainer"] code,
        code, pre {{
            font-family: "JetBrains Mono", ui-monospace, monospace;
            background: {COLORS["surface_soft"]};
            color: var(--rp-ink);
            border-radius: 6px;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid var(--rp-hairline);
            background: var(--rp-surface-card);
            color: var(--rp-ink);
        }}

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFileUploader"] button {{
            background: var(--rp-primary);
            color: white;
            border: 1px solid var(--rp-primary);
            border-radius: 8px;
            min-height: 40px;
            padding: 10px 18px;
            font-weight: 500;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFileUploader"] button:hover {{
            background: var(--rp-primary-active);
            border-color: var(--rp-primary-active);
            color: white;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid var(--rp-hairline);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px 8px 0 0;
            color: var(--rp-muted);
            padding: 10px 14px;
            font-weight: 500;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--rp-surface-card);
            color: var(--rp-ink);
        }}

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            border: 1px solid var(--rp-hairline);
            border-radius: 12px;
            overflow: hidden;
        }}

        [data-testid="stMetric"],
        div[data-testid="stExpander"] {{
            background: var(--rp-surface-card);
            border: 1px solid var(--rp-hairline);
            border-radius: 12px;
            padding: 12px;
        }}

        .rp-hero {{
            background: var(--rp-surface-card);
            border: 1px solid var(--rp-hairline);
            border-radius: 16px;
            padding: 36px 40px;
            margin: 8px 0 28px 0;
        }}

        .rp-kicker {{
            color: var(--rp-primary);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .rp-hero h2 {{
            margin: 0 0 10px 0;
            font-size: 44px;
            line-height: 1.08;
        }}

        .rp-hero p {{
            color: var(--rp-body);
            font-size: 16px;
            line-height: 1.6;
            max-width: 850px;
            margin: 0;
        }}

        .rp-card {{
            background: var(--rp-surface-card);
            border: 1px solid var(--rp-hairline);
            border-radius: 12px;
            padding: 22px 24px;
            min-height: 150px;
        }}

        .rp-card-dark {{
            background: var(--rp-surface-dark);
            border: 1px solid #252320;
            border-radius: 12px;
            color: var(--rp-on-dark);
            padding: 22px 24px;
            min-height: 150px;
        }}

        .rp-card h3,
        .rp-card-dark h3 {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 18px;
            font-weight: 600;
            margin: 0 0 8px 0;
        }}

        .rp-card-dark h3 {{
            color: var(--rp-on-dark);
        }}

        .rp-card p {{
            color: var(--rp-body);
            margin: 0;
            line-height: 1.55;
        }}

        .rp-card-dark p {{
            color: #c7c2ba;
            margin: 0;
            line-height: 1.55;
        }}

        .rp-chip {{
            display: inline-block;
            background: var(--rp-primary);
            color: white;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 12px;
        }}

        .rp-status {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--rp-body);
            font-weight: 600;
        }}

        .rp-status-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--rp-primary);
            display: inline-block;
        }}

        .rp-status-dot-muted {{
            background: #8e8b82;
        }}

        .rp-step {{
            background: var(--rp-canvas);
            border: 1px solid var(--rp-hairline);
            border-radius: 12px;
            padding: 16px 18px;
            min-height: 112px;
        }}

        .rp-step strong {{
            color: var(--rp-ink);
            display: block;
            margin-bottom: 6px;
        }}

        .rp-step span {{
            color: var(--rp-muted);
            font-size: 14px;
            line-height: 1.5;
        }}

        input, textarea, [data-baseweb="select"] > div {{
            background: var(--rp-canvas);
            border-color: var(--rp-hairline);
            border-radius: 8px;
        }}

        input:focus, textarea:focus {{
            border-color: var(--rp-primary) !important;
            box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.15) !important;
        }}

        .block-container {{
            padding-top: 48px;
            padding-bottom: 64px;
            max-width: 1280px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig, legend_title=None):
    fig.update_layout(
        paper_bgcolor=COLORS["canvas"],
        plot_bgcolor=COLORS["canvas"],
        colorway=PLOTLY_COLORWAY,
        font=dict(family="Inter, sans-serif", color=COLORS["body"], size=13),
        title=dict(font=dict(family="Cormorant Garamond, serif", color=COLORS["ink"], size=28)),
        legend=dict(
            title=legend_title,
            bgcolor="rgba(250,249,245,0.86)",
            bordercolor=COLORS["hairline"],
            borderwidth=1,
            font=dict(color=COLORS["body"]),
        ),
        margin=dict(l=72, r=32, t=72, b=64),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["hairline"],
        zerolinecolor=COLORS["hairline"],
        linecolor=COLORS["ink"],
        tickfont=dict(color=COLORS["body"]),
        title_font=dict(color=COLORS["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["hairline"],
        zerolinecolor=COLORS["hairline"],
        linecolor=COLORS["ink"],
        tickfont=dict(color=COLORS["body"]),
        title_font=dict(color=COLORS["muted"]),
    )
    return fig
