from __future__ import annotations

import base64
import html
import os
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from ai_analysis import (
    DEFAULT_OPENAI_IMAGE_MODEL,
    DEFAULT_OPENAI_MODEL,
    ai_score_average,
    analyze_creative_with_openai,
    build_ai_score_rows,
    consolidate_performance_with_openai,
    generate_improved_image_with_openai,
    get_openai_client,
)
from image_metrics import (
    average_score,
    build_basic_metric_rows,
    interpret_score,
    measure_picture_metrics_from_bytes,
    score_picture_metrics,
)


BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "image" / "elctrolux_logo.png"

TARGET_RATIOS = {
    "Square 1:1": 1.0,
    "Instagram 4:5": 0.8,
    "TikTok / Reels 9:16": 0.5625,
    "Landscape 16:9": 1.7778,
}

DEFAULT_PLATFORM_GOAL = "General ecommerce media"

OPENAI_MODEL_OPTIONS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

OPENAI_IMAGE_MODEL_OPTIONS = [
    "gpt-image-1.5",
    "gpt-image-2",
    "gpt-image-1",
    "gpt-image-1-mini",
    "chatgpt-image-latest",
]


def infer_target_format(aspect_ratio: float) -> tuple[str, float]:
    target_format = min(
        TARGET_RATIOS,
        key=lambda label: abs(TARGET_RATIOS[label] - aspect_ratio),
    )
    return target_format, TARGET_RATIOS[target_format]


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")

    st.set_page_config(
        page_title="Electrolux Creative Performance Portal",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    render_header()

    input_columns = st.columns([1.45, 1], vertical_alignment="top")
    with input_columns[0]:
        uploaded_file = render_upload_control()
    with input_columns[1]:
        settings = render_analysis_controls()

    if uploaded_file is None:
        render_empty_state()
        return

    image_bytes = uploaded_file.getvalue()

    try:
        metrics = measure_picture_metrics_from_bytes(image_bytes)
        target_format, target_ratio = infer_target_format(metrics["aspect_ratio"])
        settings = {
            **settings,
            "target_format": target_format,
            "target_ratio": target_ratio,
            "platform_goal": DEFAULT_PLATFORM_GOAL,
        }
        basic_scores = score_picture_metrics(metrics, settings["target_ratio"])
    except Exception as error:
        st.error(f"Could not analyze this image: {error}")
        return

    render_uploaded_image(image_bytes, uploaded_file.name, metrics, settings)
    render_basic_scoring(metrics, basic_scores)

    analysis_key = f"{uploaded_file.name}-{len(image_bytes)}-{settings['target_format']}-{settings['platform_goal']}-{settings['model']}"

    if st.button("Run AI scoring", type="primary", width="stretch"):
        run_openai_analysis(
            image_bytes=image_bytes,
            metrics=metrics,
            basic_scores=basic_scores,
            settings=settings,
            analysis_key=analysis_key,
        )

    result = st.session_state.get("analysis_result")
    if result and result.get("analysis_key") == analysis_key:
        render_ai_scoring(result["ai_result"])
        render_final_review(result["performance_review"])
        render_improved_image_stage(
            image_bytes=image_bytes,
            metrics=metrics,
            basic_scores=basic_scores,
            settings=settings,
            result=result,
            analysis_key=analysis_key,
        )


def render_header() -> None:
    logo_html = "<span class='brand-fallback'>Electrolux</span>"
    logo_uri = image_data_uri(LOGO_PATH)
    if logo_uri:
        logo_html = f"<img src='{logo_uri}' alt='Electrolux' />"

    st.markdown(
        f"""
        <header class="top-bar">
            <div class="brand-lockup">
                {logo_html}
                <span>Creative Performance Portal</span>
            </div>
            <div class="top-status">Internal media evaluation</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_controls() -> dict[str, Any]:
    env_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    model_options = OPENAI_MODEL_OPTIONS.copy()
    if env_model not in model_options:
        model_options.insert(0, env_model)

    env_image_model = os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_OPENAI_IMAGE_MODEL)
    image_model_options = OPENAI_IMAGE_MODEL_OPTIONS.copy()
    if env_image_model not in image_model_options:
        image_model_options.insert(0, env_image_model)

    st.markdown(
        """
        <div class="side-panel-heading">
            <div class="section-kicker">Advanced</div>
            <div class="section-title">AI models</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = st.selectbox(
        "Scoring model",
        model_options,
        index=model_options.index(env_model),
    )
    image_model = st.selectbox(
        "Image generation model",
        image_model_options,
        index=image_model_options.index(env_image_model),
    )

    credential_status = "Configured" if os.getenv("OPENAI_API_KEY") else "Missing .env key"
    st.markdown(
        f"""
        <div class="credential-row flat">
            <span>Credential</span>
            <strong>{credential_status}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "model": model,
        "image_model": image_model,
    }


def render_upload_control() -> Any:
    st.markdown(
        """
        <div class="upload-heading-clean">
            <div>
                <div class="section-kicker">Media input</div>
                <div class="section-title">Upload creative image</div>
            </div>
            <div class="section-note">JPG, PNG, or WEBP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.file_uploader(
        "Upload creative image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state clean">
            Upload an image to calculate Python metrics, AI scoring, and improvement suggestions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_uploaded_image(
    image_bytes: bytes,
    file_name: str,
    metrics: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Uploaded media</div>
                <div class="section-title">Creative preview</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_column, summary_column = st.columns([1.15, 1], vertical_alignment="top")

    with image_column:
        st.image(image_bytes, caption=file_name, width="stretch")

    with summary_column:
        st.markdown("#### Media details")
        render_media_detail_tiles(
            [
                ("Width", f"{metrics['width']} px"),
                ("Height", f"{metrics['height']} px"),
                ("Ratio", f"{metrics['aspect_ratio']:.3f}"),
                ("Format", settings["target_format"]),
            ]
        )


def render_basic_scoring(
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
) -> None:
    render_section_heading("Basic scoring", "")

    score = average_score(basic_scores)
    status = interpret_score(score)
    render_score_summary_cards(
        [
            {"label": "Image score", "value": f"{score}/10", "status": status},
            {"label": "Status", "value": status, "status": status},
        ]
    )

    rows = build_basic_metric_rows(metrics, basic_scores)
    render_basic_metric_list(rows)


def render_media_detail_tiles(items: list[tuple[str, str]]) -> None:
    tile_html = "".join(
        f"<div class='detail-tile'><span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value in items
    )
    st.markdown(f"<div class='detail-grid'>{tile_html}</div>", unsafe_allow_html=True)


def render_basic_metric_list(rows: list[dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="metric-list-header">
            <span>Metric</span>
            <span>Value</span>
            <span>Score</span>
            <span>Status</span>
            <span>Details</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for row in rows:
        columns = st.columns([2.2, 1.2, 1.05, 1.1, 0.55], vertical_alignment="center")
        columns[0].markdown(f"<div class='metric-name'>{html.escape(row['Metric'])}</div>", unsafe_allow_html=True)
        columns[1].markdown(f"<div class='metric-value'>{html.escape(row['Value'])}</div>", unsafe_allow_html=True)
        columns[2].markdown(f"<div class='metric-score'>{html.escape(row['Score'])}</div>", unsafe_allow_html=True)
        columns[3].markdown(
            f"<div class='metric-status'>{render_status_badge(row['Status'])}</div>",
            unsafe_allow_html=True,
        )

        with columns[4].popover("?", width="content"):
            st.markdown(f"**{row['Metric']}**")
            st.write(row["Meaning"])
            st.markdown("**Scoring criteria**")
            st.write(row["Criteria"])


def render_score_summary_cards(cards: list[dict[str, str]]) -> None:
    card_html = "".join(
        f"<div class='score-card {status_class(card.get('status', ''))}'>"
        f"<span>{html.escape(card['label'])}</span>"
        f"{render_score_card_value(card)}"
        f"</div>"
        for card in cards
    )
    st.markdown(f"<div class='score-card-grid'>{card_html}</div>", unsafe_allow_html=True)


def render_score_card_value(card: dict[str, str]) -> str:
    if card["label"].lower() in {"status", "level"}:
        return render_status_badge(card["value"])

    return (
        f"<strong>{html.escape(card['value'])}</strong>"
        f"{render_status_badge(card['status']) if card.get('status') else ''}"
    )


def render_status_badge(status: str) -> str:
    return f"<span class='status-badge {status_class(status)}'>{html.escape(status)}</span>"


def status_class(status: str) -> str:
    normalized_status = status.strip().lower()

    if normalized_status in {"excellent", "good"}:
        return "status-green"
    if normalized_status == "okay":
        return "status-yellow"
    if normalized_status == "weak":
        return "status-orange"
    if normalized_status == "poor":
        return "status-red"

    return "status-neutral"


def run_openai_analysis(
    image_bytes: bytes,
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
    settings: dict[str, Any],
    analysis_key: str,
) -> None:
    try:
        client = get_openai_client()

        with st.spinner("Running AI creative scoring..."):
            ai_result = analyze_creative_with_openai(
                client=client,
                image_bytes=image_bytes,
                model=settings["model"],
                target_format=settings["target_format"],
                platform_goal=settings["platform_goal"],
            )

        with st.spinner("Building final performance review..."):
            performance_review = consolidate_performance_with_openai(
                client=client,
                model=settings["model"],
                metrics=metrics,
                basic_scores=basic_scores,
                ai_scores=ai_result,
                target_format=settings["target_format"],
                platform_goal=settings["platform_goal"],
            )

        st.session_state["analysis_result"] = {
            "analysis_key": analysis_key,
            "ai_result": ai_result,
            "performance_review": performance_review,
        }
    except Exception as error:
        st.error(str(error))


def render_ai_scoring(ai_result: dict[str, Any]) -> None:
    render_section_heading("AI scoring", "Creative effectiveness scored from visible image evidence")

    average = ai_score_average(ai_result)
    status = interpret_score(average)
    render_score_summary_cards(
        [
            {"label": "AI score", "value": f"{average}/10", "status": status},
            {"label": "Status", "value": status, "status": status},
        ]
    )

    quick_read = ai_result.get("quick_read")
    if quick_read:
        st.info(quick_read)

    st.dataframe(build_ai_score_rows(ai_result), hide_index=True, width="stretch")


def render_final_review(performance_review: dict[str, Any]) -> None:
    overall = performance_review["overall_performance"]
    suggestions = performance_review["suggestions_to_improve"]

    render_section_heading("AI conclusion", "Overall performance and improvement priorities")
    render_score_summary_cards(
        [
            {
                "label": "Overall score",
                "value": f"{overall['overall_score']}/10",
                "status": overall["performance_level"],
            },
            {
                "label": "Level",
                "value": overall["performance_level"],
                "status": overall["performance_level"],
            },
        ]
    )
    st.write(overall["summary"])

    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.markdown("#### Strengths")
        render_text_list(overall.get("strengths", []))

    with detail_columns[1]:
        st.markdown("#### Risks")
        render_text_list(overall.get("risks", []))

    render_section_heading("Suggestions to improve", "Prioritized creative changes")
    render_suggestions(suggestions)


def render_improved_image_stage(
    image_bytes: bytes,
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
    settings: dict[str, Any],
    result: dict[str, Any],
    analysis_key: str,
) -> None:
    generation_key = f"{analysis_key}-{settings['image_model']}"

    render_section_heading(
        "AI generated image",
        "Creates an improved version from the original media and all scoring inputs",
    )

    if st.button("Generate improved image", type="secondary", width="stretch"):
        run_image_generation(
            image_bytes=image_bytes,
            metrics=metrics,
            basic_scores=basic_scores,
            settings=settings,
            result=result,
            generation_key=generation_key,
        )

    generated_result = st.session_state.get("improved_image_result")
    if not generated_result or generated_result.get("generation_key") != generation_key:
        return

    preview_columns = st.columns(2, vertical_alignment="top")
    with preview_columns[0]:
        st.image(image_bytes, caption="Original image", width="stretch")
    with preview_columns[1]:
        st.image(generated_result["image_bytes"], caption="AI improved image", width="stretch")
        st.download_button(
            "Download improved image",
            data=generated_result["image_bytes"],
            file_name="electrolux_improved_creative.png",
            mime="image/png",
            width="stretch",
        )

    with st.expander("Generation brief"):
        st.write(f"Image model: {generated_result['model']}")
        st.text(generated_result["prompt"])


def run_image_generation(
    image_bytes: bytes,
    metrics: dict[str, Any],
    basic_scores: dict[str, float],
    settings: dict[str, Any],
    result: dict[str, Any],
    generation_key: str,
) -> None:
    try:
        client = get_openai_client()

        with st.spinner("Generating improved image from AI scoring and suggestions..."):
            improved_image = generate_improved_image_with_openai(
                client=client,
                image_bytes=image_bytes,
                image_model=settings["image_model"],
                metrics=metrics,
                basic_scores=basic_scores,
                ai_scores=result["ai_result"],
                performance_review=result["performance_review"],
                target_format=settings["target_format"],
            )

        st.session_state["improved_image_result"] = {
            "generation_key": generation_key,
            **improved_image,
        }
    except Exception as error:
        st.error(str(error))


def render_section_heading(title: str, note: str) -> None:
    note_html = f"<div class='section-note'>{html.escape(note)}</div>" if note else ""

    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                <div class="section-kicker">Report</div>
                <div class="section-title">{title}</div>
            </div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_list(items: list[str]) -> None:
    if not items:
        st.write("None returned.")
        return

    for item in items:
        st.markdown(f"- {item}")


def render_suggestions(suggestions: list[dict[str, Any]]) -> None:
    if not suggestions:
        st.write("No suggestions returned.")
        return

    for index, item in enumerate(suggestions, start=1):
        with st.expander(f"{index}. {item['priority']} priority", expanded=index <= 3):
            st.write(item["suggestion"])
            if item.get("reason"):
                st.caption(item["reason"])
            if item.get("based_on"):
                st.write("Based on: " + ", ".join(item["based_on"]))
            if item.get("expected_impact"):
                st.write("Expected impact: " + item["expected_impact"])


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --elx-blue: #011e41;
                --elx-blue-2: #082b59;
                --elx-teal: #00a6b2;
                --elx-ink: #172033;
                --elx-muted: #667085;
                --elx-border: #d9e2ec;
                --elx-surface: #f4f7fb;
                --elx-white: #ffffff;
            }

            #MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
                display: none !important;
            }

            .stApp {
                background: var(--elx-surface);
                color: var(--elx-ink);
            }

            .block-container {
                max-width: 1120px;
                padding-top: 1rem;
                padding-bottom: 3rem;
            }

            .top-bar {
                align-items: center;
                display: flex;
                justify-content: space-between;
                margin-bottom: 1.25rem;
                padding: 0.25rem 0 0.9rem;
                border-bottom: 1px solid var(--elx-border);
            }

            .brand-lockup img {
                display: block;
                height: 31px;
                width: auto;
            }

            .brand-lockup {
                align-items: center;
                display: flex;
                gap: 0.85rem;
            }

            .brand-lockup span {
                border-left: 1px solid var(--elx-border);
                color: var(--elx-blue);
                font-size: 0.95rem;
                font-weight: 760;
                line-height: 1;
                padding-left: 0.85rem;
            }

            .brand-fallback {
                color: var(--elx-blue);
                font-size: 1.15rem;
                font-weight: 780;
                letter-spacing: 0;
            }

            .top-status {
                color: var(--elx-muted);
                font-size: 0.86rem;
                font-weight: 650;
            }

            .eyebrow, .section-kicker {
                color: var(--elx-teal);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--elx-white);
                border: 1px solid var(--elx-border);
                border-radius: 8px;
                box-shadow: 0 12px 30px rgba(1, 30, 65, 0.05);
            }

            .section-heading {
                align-items: flex-end;
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                margin: 1.3rem 0 0.65rem;
            }

            .section-heading.compact {
                margin: 0 0 0.45rem;
            }

            .upload-heading-clean {
                align-items: flex-end;
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.6rem;
                gap: 1rem;
            }

            .side-panel-heading {
                margin-bottom: 0.55rem;
            }

            .section-title {
                color: var(--elx-blue);
                font-size: 1.12rem;
                font-weight: 760;
                letter-spacing: 0;
                line-height: 1.2;
            }

            .section-note {
                color: var(--elx-muted);
                font-size: 0.84rem;
                line-height: 1.4;
                max-width: 380px;
                text-align: right;
            }

            .credential-row {
                align-items: center;
                display: flex;
                justify-content: space-between;
                margin-top: 0.75rem;
                padding: 0.55rem 0;
                border-top: 1px solid var(--elx-border);
            }

            .credential-row.flat {
                background: transparent;
                border-radius: 0;
            }

            .credential-row span {
                color: var(--elx-muted);
                font-size: 0.82rem;
                font-weight: 650;
            }

            .credential-row strong {
                color: var(--elx-blue);
                font-size: 0.84rem;
            }

            .empty-state {
                background: var(--elx-white);
                border: 1px solid var(--elx-border);
                border-radius: 8px;
                box-shadow: 0 12px 30px rgba(1, 30, 65, 0.05);
                margin-top: 1rem;
                padding: 2rem;
            }

            .empty-state.clean {
                background: transparent;
                border: 0;
                box-shadow: none;
                color: var(--elx-muted);
                font-size: 0.95rem;
                margin-top: 0.85rem;
                padding: 0;
            }

            .empty-title {
                color: var(--elx-blue);
                font-size: 1.28rem;
                font-weight: 760;
                margin-bottom: 0.35rem;
            }

            .empty-copy {
                color: var(--elx-muted);
                font-size: 0.98rem;
                line-height: 1.55;
                max-width: 720px;
            }

            .detail-grid {
                display: grid;
                gap: 0.75rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-top: 0.65rem;
            }

            .detail-tile {
                background: var(--elx-white);
                border: 1px solid var(--elx-border);
                border-radius: 8px;
                padding: 0.85rem 0.95rem;
            }

            .detail-tile span {
                color: var(--elx-muted);
                display: block;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.32rem;
            }

            .detail-tile strong {
                color: var(--elx-blue);
                display: block;
                font-size: 1.05rem;
                font-weight: 760;
                line-height: 1.2;
                overflow-wrap: anywhere;
            }

            .metric-list-header {
                align-items: center;
                border-bottom: 1px solid var(--elx-border);
                color: var(--elx-muted);
                display: grid;
                font-size: 0.78rem;
                font-weight: 760;
                gap: 1rem;
                grid-template-columns: 2.2fr 1.2fr 1.05fr 1.1fr 0.55fr;
                margin-top: 1rem;
                padding: 0 0.35rem 0.55rem;
                text-transform: uppercase;
            }

            .metric-name,
            .metric-value,
            .metric-score,
            .metric-status {
                align-items: center;
                border-bottom: 1px solid var(--elx-border);
                color: var(--elx-ink);
                display: flex;
                min-height: 2.85rem;
                padding: 0.5rem 0.35rem;
            }

            .metric-name {
                color: var(--elx-blue);
                font-weight: 560;
            }

            .metric-value,
            .metric-score {
                font-variant-numeric: tabular-nums;
                font-weight: 500;
            }

            .metric-status {
                font-weight: 500;
            }

            .metric-status .status-badge {
                font-weight: 600;
            }

            .score-card-grid {
                display: grid;
                gap: 0.85rem;
                grid-template-columns: repeat(auto-fit, minmax(165px, 220px));
                margin: 0.85rem 0 1rem;
            }

            .score-card {
                --status-bg: #f7f9fc;
                --status-border: var(--elx-border);
                --status-text: var(--elx-blue);
                background: var(--elx-white);
                border: 1px solid var(--status-border);
                border-left: 5px solid var(--status-border);
                border-radius: 8px;
                padding: 0.85rem 0.95rem;
            }

            .score-card span:first-child {
                color: var(--elx-muted);
                display: block;
                font-size: 0.78rem;
                font-weight: 720;
                margin-bottom: 0.35rem;
            }

            .score-card strong {
                color: var(--status-text);
                display: block;
                font-size: 1.45rem;
                font-weight: 760;
                line-height: 1.15;
                margin-bottom: 0.5rem;
            }

            .status-badge {
                background: var(--status-bg);
                border: 1px solid var(--status-border);
                border-radius: 999px;
                color: var(--status-text);
                display: inline-flex;
                font-size: 0.76rem;
                font-weight: 800;
                justify-content: center;
                line-height: 1;
                padding: 0.32rem 0.55rem;
                width: fit-content;
            }

            .status-green {
                --status-bg: #eaf7ef;
                --status-border: #8fd0aa;
                --status-text: #087345;
            }

            .status-yellow {
                --status-bg: #fff7d8;
                --status-border: #e6c75f;
                --status-text: #876400;
            }

            .status-orange {
                --status-bg: #fff0e2;
                --status-border: #e6a162;
                --status-text: #a85605;
            }

            .status-red {
                --status-bg: #ffe8e5;
                --status-border: #e58e88;
                --status-text: #b42318;
            }

            .status-neutral {
                --status-bg: #f2f4f7;
                --status-border: #c8d1dc;
                --status-text: var(--elx-muted);
            }

            [data-testid="stPopover"] button {
                align-items: center;
                background: var(--elx-white);
                border: 1px solid #c8d1dc;
                border-radius: 999px;
                color: var(--elx-blue);
                display: inline-flex;
                font-weight: 500;
                justify-content: center;
                min-height: 2rem;
                padding: 0;
                width: 2rem;
            }

            [data-testid="stPopover"] button p {
                font-weight: 500 !important;
                margin: 0 !important;
            }

            [data-testid="stPopover"] button svg,
            [data-testid="stPopover"] button [data-testid="stIconMaterial"],
            [data-testid="stPopover"] button .material-symbols-rounded,
            [data-testid="stPopover"] button .material-icons {
                display: none !important;
            }

            div[data-testid="stMetric"] {
                background: var(--elx-white);
                border: 1px solid var(--elx-border);
                border-radius: 8px;
                box-shadow: 0 8px 18px rgba(1, 30, 65, 0.04);
                padding: 0.8rem 0.9rem;
            }

            [data-testid="stFileUploader"] section {
                background: var(--elx-white) !important;
                border: 1.5px dashed #9db0c6 !important;
                border-radius: 8px !important;
                color: var(--elx-ink) !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 0.85rem !important;
                min-height: 150px;
                padding: 1.4rem !important;
            }

            [data-testid="stFileUploader"] section:hover {
                border-color: var(--elx-teal) !important;
                background: #fbfdff !important;
            }

            [data-testid="stFileUploader"] button {
                background: var(--elx-blue) !important;
                border: 1px solid var(--elx-blue) !important;
                border-radius: 6px !important;
                color: var(--elx-white) !important;
                font-weight: 750 !important;
                margin: 0 auto !important;
            }

            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploader"] button * {
                color: var(--elx-white) !important;
                fill: var(--elx-white) !important;
            }

            [data-testid="stFileUploader"] section > div:first-child {
                align-items: center !important;
                display: flex !important;
                flex-direction: column !important;
                gap: 0.85rem !important;
                justify-content: center !important;
                width: 100% !important;
            }

            [data-testid="stFileUploader"] section > div {
                align-items: center !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                text-align: center !important;
                width: 100% !important;
            }

            [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {
                color: var(--elx-muted) !important;
                margin: 0 !important;
                text-align: center !important;
            }

            [data-testid="stFileUploader"] button span,
            [data-testid="stFileUploader"] button p,
            [data-testid="stFileUploader"] button svg,
            [data-testid="stFileUploader"] button div {
                color: var(--elx-white) !important;
                fill: var(--elx-white) !important;
            }

            div[data-baseweb="select"] > div {
                background: var(--elx-white);
                border-color: var(--elx-border);
                border-radius: 6px;
            }

            label, [data-testid="stWidgetLabel"] p {
                color: var(--elx-blue) !important;
                font-size: 0.84rem !important;
                font-weight: 720 !important;
            }

            .stButton > button {
                background: var(--elx-blue);
                border: 1px solid var(--elx-blue);
                border-radius: 6px;
                color: var(--elx-white);
                font-weight: 700;
                min-height: 2.8rem;
            }

            .stButton > button:hover {
                background: var(--elx-blue-2);
                border-color: var(--elx-blue-2);
                color: var(--elx-white);
            }

            h3, h4 {
                color: var(--elx-blue);
                letter-spacing: 0;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--elx-border);
                border-radius: 8px;
                overflow: hidden;
            }

            div[data-testid="stAlert"] {
                border-radius: 8px;
            }

            @media (max-width: 760px) {
                .top-bar, .section-heading, .upload-heading-clean {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .section-note {
                    text-align: left;
                }

            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


if __name__ == "__main__":
    main()