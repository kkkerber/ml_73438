"""Render the canonical proposal Markdown as a one-page A4 PDF (bundled Python)."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/proposal.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleLab', fontName='Helvetica-Bold', fontSize=19, leading=23,
                          textColor=colors.HexColor('#16324F'), spaceAfter=10))
styles.add(ParagraphStyle(name='SectionLab', fontName='Helvetica-Bold', fontSize=10.5, leading=13,
                          textColor=colors.HexColor('#196A73'), spaceBefore=9, spaceAfter=4))
styles.add(ParagraphStyle(name='BodyLab', fontName='Helvetica', fontSize=9.6, leading=13,
                          spaceAfter=5))

def inline(text):
    text = escape(text.strip())
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return text

story = []
for block in (ROOT / 'results/proposal.md').read_text().split('\n\n'):
    block = block.strip()
    if block.startswith('# '):
        story.append(Paragraph(inline(block[2:]), styles['TitleLab']))
    elif block.startswith('## '):
        story.append(Paragraph(inline(block[3:]), styles['SectionLab']))
    else:
        story.append(Paragraph('<br/>'.join(inline(line) for line in block.splitlines()), styles['BodyLab']))

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#CCD6DF'))
    canvas.line(42, 34, A4[0] - 42, 34)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526474'))
    canvas.drawString(42, 23, 'Lab 6 | Individual project proposal | Valeriia Khapilina, 73438')
    canvas.drawRightString(A4[0] - 42, 23, str(doc.page))

SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=42, leftMargin=42,
                  topMargin=35, bottomMargin=45).build(story, onFirstPage=footer, onLaterPages=footer)
assert len(PdfReader(OUT).pages) == 1, 'Proposal must fit on one page'
print(OUT)
