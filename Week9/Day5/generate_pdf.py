from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "meta_analysis_llms.md"
OUT = BASE / "meta_analysis_llms.pdf"


def inline_md(text: str) -> str:
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", text)
    return text


def paragraph(text: str, style):
    return Paragraph(inline_md(text), style)


def parse_table(lines, start, styles):
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        line = lines[i].strip()
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not all(set(c) <= {"-", ":", " "} for c in cells):
            style = styles["TableHead"] if not rows else styles["TableCell"]
            rows.append([Paragraph(inline_md(c), style) for c in cells])
        i += 1

    if not rows:
        return None, i

    col_count = len(rows[0])
    usable_width = A4[0] - 2.4 * cm
    first = 2.25 * cm
    remaining = usable_width - first
    widths = [first] + [remaining / (col_count - 1)] * (col_count - 1)
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9ca3af")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table, i


def build_story():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ProjectTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=14.5,
            leading=17,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=12,
            spaceBefore=6,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=8.9,
            leading=10.3,
            spaceBefore=4.5,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=7.9,
            leading=9.55,
            spaceAfter=2.7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["BodySmall"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontSize=5.55,
            leading=6.45,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHead",
            parent=styles["TableCell"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    buffer = []

    def flush():
        if buffer:
            story.append(paragraph(" ".join(buffer).strip(), styles["BodySmall"]))
            buffer.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line:
            flush()
            i += 1
            continue

        if line.startswith("|"):
            flush()
            table, i = parse_table(lines, i, styles)
            if table is not None:
                story.append(table)
                story.append(Spacer(1, 4))
            continue

        if line.startswith("# "):
            flush()
            story.append(paragraph(line[2:].strip(), styles["ProjectTitle"]))
        elif line.startswith("## "):
            flush()
            story.append(paragraph(line[3:].strip(), styles["Section"]))
        elif line.startswith("### "):
            flush()
            story.append(paragraph(line[4:].strip(), styles["Subsection"]))
        elif re.match(r"^\d+\.\s+", line):
            flush()
            story.append(paragraph(line, styles["BodySmall"]))
        elif line.startswith("- "):
            flush()
            story.append(paragraph("- " + line[2:], styles["BodySmall"]))
        else:
            buffer.append(line)
        i += 1

    flush()
    return story


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.05 * cm,
        bottomMargin=1.05 * cm,
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(A4[0] / 2, 0.55 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(f"PDF generated: {OUT}")
