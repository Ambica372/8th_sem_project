"""
generate_dataset_doc.py
Generates a well-formatted PDF explaining every file in objective2/processed_data.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import PageBreak

OUTPUT_PATH = r"c:\Users\Rose J Thachil\Desktop\8th_sem_project\objective2\processed_data_explanation.pdf"

# ── Colour palette ──────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#1a2744")
STEEL      = colors.HexColor("#2c4a7c")
ACCENT     = colors.HexColor("#4a90d9")
LIGHT_BG   = colors.HexColor("#eef3fb")
GREEN      = colors.HexColor("#2e7d32")
ORANGE     = colors.HexColor("#e65100")
LIGHT_GRAY = colors.HexColor("#f5f5f5")
DARK_GRAY  = colors.HexColor("#424242")
WHITE      = colors.white

def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", parent=base["Normal"],
        fontSize=26, textColor=WHITE, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=8
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", parent=base["Normal"],
        fontSize=13, textColor=colors.HexColor("#c5d8f5"),
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4
    )
    styles["section_header"] = ParagraphStyle(
        "section_header", parent=base["Normal"],
        fontSize=15, textColor=WHITE, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6, leftIndent=0
    )
    styles["file_title"] = ParagraphStyle(
        "file_title", parent=base["Normal"],
        fontSize=13, textColor=NAVY, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=4
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=10, textColor=DARK_GRAY, fontName="Helvetica",
        leading=15, alignment=TA_JUSTIFY, spaceAfter=6
    )
    styles["body_bold"] = ParagraphStyle(
        "body_bold", parent=base["Normal"],
        fontSize=10, textColor=NAVY, fontName="Helvetica-Bold",
        leading=15, spaceAfter=4
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=base["Normal"],
        fontSize=10, textColor=DARK_GRAY, fontName="Helvetica",
        leading=15, leftIndent=18, bulletIndent=6,
        spaceAfter=3
    )
    styles["warning"] = ParagraphStyle(
        "warning", parent=base["Normal"],
        fontSize=10, textColor=ORANGE, fontName="Helvetica-Bold",
        leading=14, spaceAfter=5, leftIndent=12
    )
    styles["ok"] = ParagraphStyle(
        "ok", parent=base["Normal"],
        fontSize=10, textColor=GREEN, fontName="Helvetica-Bold",
        leading=14, spaceAfter=5, leftIndent=12
    )
    styles["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"],
        fontSize=9, textColor=colors.HexColor("#78909c"),
        fontName="Helvetica-Oblique", alignment=TA_CENTER,
        spaceAfter=6
    )
    styles["mono"] = ParagraphStyle(
        "mono", parent=base["Normal"],
        fontSize=9, textColor=NAVY, fontName="Courier",
        leading=13, spaceAfter=4, leftIndent=14,
        backColor=LIGHT_BG
    )
    return styles


def section_banner(text, styles):
    """Returns a coloured banner paragraph used as a section header."""
    data = [[Paragraph(text, styles["section_header"])]]
    tbl = Table(data, colWidths=[17.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), STEEL),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tbl


def info_table(rows, styles):
    """Produces a two-column property table."""
    data = []
    for label, value in rows:
        data.append([
            Paragraph(f"<b>{label}</b>", styles["body"]),
            Paragraph(str(value), styles["body"])
        ])
    tbl = Table(data, colWidths=[5 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("BACKGROUND", (1, 0), (1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def label_dist_table(styles):
    headers = ["Label", "Emotion", "Sample Count", "% of Total"]
    rows_data = [
        ["0", "Neutral", "9,480", "25.2 %"],
        ["1", "Sad",     "9,510", "25.3 %"],
        ["2", "Fear",    "8,430", "22.4 %"],
        ["3", "Happy",  "10,155", "27.0 %"],
    ]
    all_rows = [[Paragraph(f"<b>{h}</b>", styles["body"]) for h in headers]]
    for r in rows_data:
        all_rows.append([Paragraph(c, styles["body"]) for c in r])

    tbl = Table(all_rows, colWidths=[3*cm, 4*cm, 5*cm, 5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), STEEL),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    S = build_styles()
    story = []

    # ── Cover block ─────────────────────────────────────────────────────────
    cover_data = [[
        Paragraph("Processed Data — File Reference", S["cover_title"]),
    ], [
        Paragraph("objective2 / processed_data", S["cover_sub"]),
    ], [
        Paragraph("SEED-IV Multimodal Emotion Recognition Dataset", S["cover_sub"]),
    ]]
    cover = Table(cover_data, colWidths=[17.5*cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING", (0, 0), (-1, 0), 28),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 28),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.5*cm))

    # ── Overview ─────────────────────────────────────────────────────────────
    story.append(section_banner("📋  Overview", S))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This folder contains <b>four NumPy binary files (.npy)</b> — the final, model-ready dataset "
        "for an <b>emotion recognition task</b>. The data originates from the <b>SEED-IV</b> dataset, "
        "which records brain (EEG) activity and eye-tracking signals from human subjects as they watch "
        "emotionally-evocative video clips.", S["body"]))
    story.append(Paragraph(
        "All files share the <b>same row order</b> — row <i>i</i> in any feature matrix corresponds "
        "exactly to label <i>i</i> in the label array. The data has been fully preprocessed and is "
        "ready for direct consumption by machine learning models.", S["body"]))
    story.append(Spacer(1, 0.3*cm))

    # Emotion label table
    story.append(Paragraph("<b>Emotion Labels (4-class classification)</b>", S["body_bold"]))
    story.append(label_dist_table(S))
    story.append(Paragraph(
        "The class distribution is reasonably balanced — no single emotion dominates overwhelmingly, "
        "which is a healthy sign for training classifiers.", S["caption"]))
    story.append(Spacer(1, 0.4*cm))

    # ── FILE 1: y.npy ────────────────────────────────────────────────────────
    story.append(KeepTogether([
        section_banner("📄  File 1 — y.npy   (The Answer Key / Labels)", S),
        Spacer(1, 0.2*cm),
        info_table([
            ("Shape",       "(37,575,)  — a 1-D array"),
            ("Data Type",   "int64  (integer)"),
            ("Value Range", "0, 1, 2, or 3"),
            ("NaN Present", "❌  No — completely clean"),
            ("File Size",   "~293 KB"),
        ], S),
        Spacer(1, 0.2*cm),
    ]))

    story.append(Paragraph("<b>What it is</b>", S["body_bold"]))
    story.append(Paragraph(
        "This is a simple list of 37,575 integers. Each integer is the <b>emotion label</b> "
        "for one data sample — one time-slice of a recording session.", S["body"]))

    story.append(Paragraph("<b>In plain English</b>", S["body_bold"]))
    story.append(Paragraph(
        "Imagine you have 37,575 flashcards. Each card says: \"For observation #X, the subject "
        "was feeling emotion Y.\" This file is that entire stack of answer cards. It is the "
        "<b>ground truth</b> that all models try to predict.", S["body"]))

    story.append(Paragraph("✅ This is the cleanest, most critical file — handle it carefully. "
                            "Never shuffle it without applying the same shuffle to all feature files.",
                            S["ok"]))
    story.append(Spacer(1, 0.4*cm))

    # ── FILE 2: X_eeg_pca.npy ───────────────────────────────────────────────
    story.append(KeepTogether([
        section_banner("📄  File 2 — X_eeg_pca.npy   (EEG Brain Signal Features)", S),
        Spacer(1, 0.2*cm),
        info_table([
            ("Shape",       "(37,575 rows × 29 columns)"),
            ("Data Type",   "float64"),
            ("Value Range", "–42.84  to  +210.69"),
            ("Mean",        "≈ 0.0  (zero-centred after Z-score normalisation)"),
            ("NaN Present", "❌  No — completely clean"),
            ("File Size",   "~8.3 MB"),
        ], S),
        Spacer(1, 0.2*cm),
    ]))

    story.append(Paragraph("<b>What it is</b>", S["body_bold"]))
    story.append(Paragraph(
        "A 2-D matrix where <b>each row is one observation</b> and <b>each column is one "
        "Principal Component</b> extracted from raw EEG brain signals. 37,575 rows means "
        "37,575 individual time-sliced observations; 29 columns means 29 compressed features.", S["body"]))

    story.append(Paragraph("<b>How it was created — step by step</b>", S["body_bold"]))
    steps = [
        "Raw EEG signals were recorded from <b>62 scalp electrodes</b> while subjects watched video clips.",
        "<b>Differential Entropy (DE)</b> features were computed across 5 frequency bands "
        "(δ delta, θ theta, α alpha, β beta, γ gamma) — producing hundreds of raw features.",
        "<b>PCA (Principal Component Analysis)</b> compressed those features down to just "
        "<b>29 components</b> that capture the largest variance in the brain signal.",
        "<b>Z-score normalisation</b> was applied, making the data zero-centred (confirmed by mean ≈ 0.0).",
    ]
    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"  {i}.  {step}", S["bullet"]))

    story.append(Paragraph("<b>In plain English</b>", S["body_bold"]))
    story.append(Paragraph(
        "Think of it like taking a full-resolution photo of someone's brain electrical activity "
        "and compressing it into a 29-number fingerprint that still captures the most important "
        "patterns. Each row is one such fingerprint for one moment in time.", S["body"]))
    story.append(Paragraph("✅ This is the cleanest feature file — fully preprocessed, no missing values, "
                            "zero-centred. Ideal for standalone modelling.", S["ok"]))
    story.append(Spacer(1, 0.4*cm))

    # ── FILE 3: X_eye_clean.npy ─────────────────────────────────────────────
    story.append(KeepTogether([
        section_banner("📄  File 3 — X_eye_clean.npy   (Eye-Tracking Features)", S),
        Spacer(1, 0.2*cm),
        info_table([
            ("Shape",       "(37,575 rows × 29 columns)"),
            ("Data Type",   "float64"),
            ("NaN Present", "⚠️  Yes — 26 rows affected, across 6 specific columns (cols 4–9)"),
            ("File Size",   "~8.3 MB"),
        ], S),
        Spacer(1, 0.2*cm),
    ]))

    story.append(Paragraph("<b>What it is</b>", S["body_bold"]))
    story.append(Paragraph(
        "A 2-D matrix of <b>eye-tracking measurements</b> for the same 37,575 observations. "
        "It mirrors the EEG file in shape (37,575 × 29) so they can be used together or independently.", S["body"]))

    story.append(Paragraph("<b>What the 29 features likely represent</b>", S["body_bold"]))
    eye_features = [
        "Pupil diameter (left and right)",
        "Blink rate and average blink duration",
        "Saccade speed and frequency (rapid eye movements between fixation points)",
        "Fixation duration (how long the gaze stays at one spot)",
        "Eye openness percentage",
        "Gaze direction coordinates (x, y on screen)",
    ]
    for feat in eye_features:
        story.append(Paragraph(f"  •  {feat}", S["bullet"]))

    story.append(Paragraph("<b>In plain English</b>", S["body_bold"]))
    story.append(Paragraph(
        "While the EEG file captures what is happening <i>inside</i> the brain, this file "
        "captures what the <i>eyes</i> were doing at those same moments. Emotions influence "
        "pupil dilation, blinking, and gaze patterns — so this is a complementary signal.", S["body"]))

    story.append(Paragraph(
        "⚠️  NaN Warning: 26 rows (< 0.07% of data) contain missing values in 6 columns. "
        "These likely occur at blink moments when the pupil was undetectable. "
        "Must be handled via imputation or row-dropping before feeding to most ML models.",
        S["warning"]))
    story.append(Spacer(1, 0.4*cm))

    # ── FILE 4: X_fused.npy ─────────────────────────────────────────────────
    story.append(KeepTogether([
        section_banner("📄  File 4 — X_fused.npy   (Combined Multimodal Matrix)", S),
        Spacer(1, 0.2*cm),
        info_table([
            ("Shape",       "(37,575 rows × 58 columns)"),
            ("Data Type",   "float64"),
            ("Composition", "Columns 0–28: EEG PCA features  |  Columns 29–57: Eye-tracking features"),
            ("NaN Present", "⚠️  Yes — same 26 rows as X_eye_clean (eye-tracking half)"),
            ("File Size",   "~16.6 MB"),
        ], S),
        Spacer(1, 0.2*cm),
    ]))

    story.append(Paragraph("<b>What it is</b>", S["body_bold"]))
    story.append(Paragraph(
        "This is <b>X_eeg_pca and X_eye_clean concatenated side-by-side</b> into one unified matrix. "
        "The math is simple: 29 EEG columns + 29 Eye columns = <b>58 total feature columns</b>.", S["body"]))

    story.append(Paragraph("<b>Why does this file exist?</b>", S["body_bold"]))
    story.append(Paragraph(
        "This is the <b>multimodal feature representation</b>. The hypothesis is that combining "
        "brain signals (EEG) and eye movements into a single feature vector gives a richer, "
        "more complete signal for emotion classification than either modality alone.", S["body"]))

    story.append(Paragraph("<b>In plain English</b>", S["body_bold"]))
    story.append(Paragraph(
        "If EEG is like reading someone's internal brain state, and eye-tracking is reading their "
        "external physical reaction — this file reads <i>both simultaneously</i>. It provides the "
        "\"full picture\" input for multimodal models.", S["body"]))

    story.append(Paragraph(
        "⚠️  The same 26 NaN rows from the eye-tracking data propagate into this file. "
        "The EEG half (first 29 columns) within this file remains clean.",
        S["warning"]))
    story.append(Spacer(1, 0.4*cm))

    # ── How files relate ─────────────────────────────────────────────────────
    story.append(section_banner("🗺️  How the Files Relate to Each Other", S))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "All four files are aligned row-by-row. They were generated together from the same "
        "source recording sessions. Here is how they connect:", S["body"]))
    story.append(Spacer(1, 0.15*cm))

    rel_data = [
        [Paragraph("<b>File</b>", S["body"]),
         Paragraph("<b>Role</b>", S["body"]),
         Paragraph("<b>Shape</b>", S["body"]),
         Paragraph("<b>Clean?</b>", S["body"])],
        [Paragraph("y.npy", S["mono"]),
         Paragraph("Emotion labels (ground truth)", S["body"]),
         Paragraph("(37575,)", S["mono"]),
         Paragraph("✅ Yes", S["ok"])],
        [Paragraph("X_eeg_pca.npy", S["mono"]),
         Paragraph("EEG brain features (PCA)", S["body"]),
         Paragraph("(37575, 29)", S["mono"]),
         Paragraph("✅ Yes", S["ok"])],
        [Paragraph("X_eye_clean.npy", S["mono"]),
         Paragraph("Eye-tracking features", S["body"]),
         Paragraph("(37575, 29)", S["mono"]),
         Paragraph("⚠️ 26 NaN rows", S["warning"])],
        [Paragraph("X_fused.npy", S["mono"]),
         Paragraph("EEG + Eye combined", S["body"]),
         Paragraph("(37575, 58)", S["mono"]),
         Paragraph("⚠️ 26 NaN rows", S["warning"])],
    ]

    rel_tbl = Table(rel_data, colWidths=[4.5*cm, 6*cm, 3.5*cm, 3.5*cm])
    rel_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), STEEL),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(rel_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Key Gotchas ──────────────────────────────────────────────────────────
    story.append(section_banner("⚠️  Key Gotchas — Must Read Before Using This Data", S))
    story.append(Spacer(1, 0.2*cm))

    gotchas = [
        ("<b>NaN values exist</b> in X_eye_clean and X_fused",
         "26 rows across 6 columns (columns 4–9) have missing values. Any model or pipeline "
         "using these files must handle NaNs via imputation (e.g., mean/median fill) or by dropping those rows."),
        ("<b>X_eeg_pca is clean and normalised</b>",
         "It is the safest file to use standalone. The near-zero mean confirms Z-score normalisation was applied."),
        ("<b>All four files share the same row order</b>",
         "Row i in X_eeg_pca, X_eye_clean, and X_fused all correspond to label i in y.npy. "
         "Never shuffle one file without applying the identical shuffle to all others."),
        ("<b>This is preprocessed data — not raw</b>",
         "The original SEED-IV .mat files have already been consumed. What you have here is "
         "the final extracted, Z-scored, PCA-transformed dataset. You do not need the raw files to use this."),
        ("<b>No subject/trial metadata stored here</b>",
         "Subject IDs and trial splits needed for LOSO (Leave-One-Subject-Out) cross-validation "
         "are NOT embedded in these files. That mapping must be maintained separately."),
        ("<b>X_fused = X_eeg_pca || X_eye_clean</b>",
         "The fused file is a direct horizontal concatenation. Columns 0–28 are EEG, columns 29–57 are eye. "
         "There is no additional transformation between the component files and the fused file."),
    ]

    for title, detail in gotchas:
        story.append(Paragraph(f"  •  {title}", S["body_bold"]))
        story.append(Paragraph(f"     {detail}", S["body"]))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cfd8dc")))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "Generated automatically from objective2/processed_data  ·  SEED-IV Emotion Recognition Project",
        S["caption"]))

    doc.build(story)
    print(f"PDF saved to:\n    {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
