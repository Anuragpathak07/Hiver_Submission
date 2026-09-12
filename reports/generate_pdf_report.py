import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.units import inch

def create_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F1F5F9'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1E293B')
    )

    story = []

    # Title & Header Block
    story.append(Paragraph("Hiver AI Customer Support Agent for AppleSupport", title_style))
    story.append(Paragraph("<b>Author:</b> Anurag Pathak &nbsp;|&nbsp; <b>Submission For:</b> Hiver SDE Intern Take-Home Assignment &nbsp;|&nbsp; <b>Target Brand:</b> AppleSupport", subtitle_style))
    story.append(Paragraph("<b>GitHub Repository:</b> https://github.com/Anuragpathak07/Hiver_Submission.git &nbsp;|&nbsp; <b>Decision Log:</b> refer to decision_log.md", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("Executive Summary & Headline Results", h1_style))
    
    table_data = [
        [
            Paragraph("Evaluation Metric", table_header_style),
            Paragraph("System Result", table_header_style),
            Paragraph("Benchmark Baseline", table_header_style),
            Paragraph("Production Significance", table_header_style)
        ],
        [
            Paragraph("<b>Intent Classification Accuracy</b>", table_cell_style),
            Paragraph("<b>85.00%</b>", table_cell_style),
            Paragraph("15.50% (Majority)<br/>79.50% (TF-IDF)", table_cell_style),
            Paragraph("<b>+5.50% improvement</b> over TF-IDF baseline across 8 intents.", table_cell_style)
        ],
        [
            Paragraph("<b>Intent Classification Macro-F1</b>", table_cell_style),
            Paragraph("<b>0.8512</b>", table_cell_style),
            Paragraph("0.0335 (Majority)<br/>0.7954 (TF-IDF)", table_cell_style),
            Paragraph("High balanced performance with zero class collapse.", table_cell_style)
        ],
        [
            Paragraph("<b>Historical Retrieval Match @ 3</b>", table_cell_style),
            Paragraph("<b>87.50%</b>", table_cell_style),
            Paragraph("69.50% (Lexical TF-IDF)", table_cell_style),
            Paragraph("<b>+18.00% improvement</b> using semantic vector search.", table_cell_style)
        ],
        [
            Paragraph("<b>Retrieval Top-1 Avg Similarity</b>", table_cell_style),
            Paragraph("<b>0.7142</b>", table_cell_style),
            Paragraph("0.3711 (Lexical TF-IDF)", table_cell_style),
            Paragraph("Accurately captures technical symptom paraphrasing.", table_cell_style)
        ],
        [
            Paragraph("<b>LLM-as-Judge Quality Score</b>", table_cell_style),
            Paragraph("<b>4.25 / 5.0</b>", table_cell_style),
            Paragraph("80.00% Pass Rate", table_cell_style),
            Paragraph("Evaluated across 6 criteria (correctness, grounding, tone).", table_cell_style)
        ],
        [
            Paragraph("<b>Human-Judge Agreement</b>", table_cell_style),
            Paragraph("<b>90.00%</b>", table_cell_style),
            Paragraph("30-case validation", table_cell_style),
            Paragraph("Strong alignment between automated judge & human review.", table_cell_style)
        ],
        [
            Paragraph("<b>Auto-Handle Rate</b>", table_cell_style),
            Paragraph("<b>73.00%</b>", table_cell_style),
            Paragraph("146 / 200 Golden Set", table_cell_style),
            Paragraph("Safely resolves routine diagnostic & update queries.", table_cell_style)
        ],
        [
            Paragraph("<b>Escalation Rate</b>", table_cell_style),
            Paragraph("<b>27.00%</b>", table_cell_style),
            Paragraph("54 / 200 Golden Set", table_cell_style),
            Paragraph("Safely escalates identity, billing, or low-confidence queries.", table_cell_style)
        ]
    ]

    t = Table(table_data, colWidths=[1.5*inch, 1.0*inch, 1.8*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Core Engineering Principles
    story.append(Paragraph("Core Engineering Principles & Production Philosophy", h1_style))
    principles = [
        "<b>Comprehensive Multi-Layer Evaluation ('Proof Over System'):</b> Evaluated every tier independently—Intent Classifier, RAG Retrieval Engine, Grounded Generation, Deterministic Escalation, and LLM-as-Judge.",
        "<b>Production Safety & Risk-Aware Guardrails:</b> Our <b>27.0% Escalation Rate</b> reflects intentional safety design: high-risk queries (account security, billing) and low-confidence predictions (<0.50) route to specialists.",
        "<b>Engineering Integrity & Self-Awareness:</b> Includes a dedicated <i>'What is Misleading About My Headline Number?'</i> analysis dissecting confidence intervals (N=200), temporal drift (2017 iOS), and pre-annotation bias.",
        "<b>Instant Offline Reproducibility:</b> Evaluators can run unit/integration tests (<code>pytest tests/</code>) or single-query inference (<code>python src/pipeline.py</code>) in under 10 seconds without mandatory API keys."
    ]
    for p in principles:
        story.append(Paragraph(f"• {p}", bullet_style))
    story.append(Spacer(1, 10))

    # Section 1: Problem Framing
    story.append(Paragraph("1. Problem Framing & Brand Selection", h1_style))
    story.append(Paragraph("<b>1.1 Brand Selection Analysis (AppleSupport):</b> Analyzed ~3 million tweets across brands. AppleSupport was chosen due to high multi-turn diagnostic volume (106,000+ tweets), technical symptom depth (iOS 11 battery drain, autocorrect glitch), and actionable resolution ground truth (support.apple.com URLs).", body_style))
    story.append(Paragraph("<b>1.2 What 'Good' Means for AppleSupport:</b> (1) Grounded Precision (no hallucinated steps), (2) Conservative Escalation (high-risk identity/billing queries routed to humans), (3) Reproducible Proof over trivial baselines.", body_style))
    story.append(Spacer(1, 10))

    # Section 2: Dataset Journey
    story.append(Paragraph("2. Dataset Journey & Intent Discovery", h1_style))
    story.append(Paragraph("<b>2.1 Recursive Tree Reconstruction:</b> Reconstructed flat tweets into <b>80,672 complete support cases</b> using parent tweet ID chains.", body_style))
    story.append(Paragraph("<b>2.2 Bottom-Up Intent Discovery:</b> Used <code>all-MiniLM-L6-v2</code> dense embeddings + KMeans ($k=12$) + Groq LLM taxonomy consolidation to produce a frozen 8-intent taxonomy: <i>ios_update_issues, battery_life_issue, macbook_issues, keyboard_autocorrect_issue, music_audio_issue, iphone_hardware_issue, account_identity_issue, other_unclear</i>.", body_style))
    story.append(Spacer(1, 10))

    # Section 3: Golden Set
    story.append(Paragraph("3. Hand-Annotation & Golden Set Methodology", h1_style))
    story.append(Paragraph("Created a 200-example Golden Set benchmark via stratified sampling (seed=42). Maintained <b>strict data isolation</b> by excluding all Golden Set case IDs from training corpora and vector search indices. Human verification showed <b>92.50% agreement</b> with silver labels (15 disagreement edge cases corrected).", body_style))
    story.append(Spacer(1, 10))

    # Section 4: System Architecture
    story.append(Paragraph("4. System Architecture", h1_style))
    arch_box = (
        "Customer Query ➔ Intent Classifier (SentenceTransformer + LogReg 85% Acc)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;➔ Dual Vector Retrieval (87.5% Match@3)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;➔ Multi-Tiered Escalation Engine (Confidence < 0.50 or Sensitive Keywords ➔ ESCALATE)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;➔ Grounded Response Generation (Groq LLM openai/gpt-oss-20b + Fallbacks)"
    )
    story.append(Paragraph(arch_box, code_style))
    story.append(Spacer(1, 10))

    # Section 5: Baseline Comparisons
    story.append(Paragraph("5. Quantitative Results vs. Two Baselines", h1_style))
    story.append(Paragraph("• <b>Majority Class Baseline:</b> Accuracy 15.50% | Macro-F1 0.0335", bullet_style))
    story.append(Paragraph("• <b>TF-IDF + Logistic Regression:</b> Accuracy 79.50% | Macro-F1 0.7954", bullet_style))
    story.append(Paragraph("• <b>SentenceTransformer + LogReg (Our Model):</b> Accuracy <b>85.00%</b> | Macro-F1 <b>0.8512</b> (+5.50% boost)", bullet_style))
    story.append(Spacer(1, 10))

    # Section 6: Top 5 Failure Modes
    story.append(Paragraph("6. Top 5 Real Failure Modes Analysis", h1_style))
    failures = [
        "<b>Hardware vs. Software Ambiguity:</b> Update-induced hardware symptoms (e.g. contacts glitch after iOS 11) misclassified as hardware.",
        "<b>Short / Noisy Customer Queries:</b> Expletive/short queries (e.g. 'what the fuck') yield low similarity (safely escalated).",
        "<b>Taxonomy Boundary Blur:</b> Store purchasing inquiries overlapping between account issues and general unclear queries.",
        "<b>Retrieval Response Divergence:</b> High query similarity where historical support response was an incomplete DM request.",
        "<b>Emoji & Informal Slang Over-conservatism:</b> High informal syntax variance lowering classifier confidence score."
    ]
    for f in failures:
        story.append(Paragraph(f"• {f}", bullet_style))
    story.append(Spacer(1, 10))

    # Section 7: Misleading Headline Numbers
    story.append(Paragraph("7. 'What is Misleading About My Headline Number?'", h1_style))
    misleading = [
        "<b>Margin of Error on Small Golden Set (N=200):</b> 95% confidence interval spans [80.0%, 90.0%] (±5.0% error margin).",
        "<b>Stratified vs. Natural Class Distribution:</b> Stratified sampling overrepresents rare intents relative to real-world skewed Twitter traffic.",
        "<b>Silver Pre-Annotation Confirmation Bias:</b> Accepting silver pre-labels in 92.5% of cases introduces mild verification confirmation bias.",
        "<b>Temporal Shift (2017 Dataset):</b> Models trained on 2017 iOS 11 issues will degrade on modern iOS 17/18 queries without continuous retraining.",
        "<b>Retrieval Metric vs. Resolution:</b> Intent match @ 3 measures class overlap, not whether the historical response actually solved the user's issue."
    ]
    for m in misleading:
        story.append(Paragraph(f"• {m}", bullet_style))
    story.append(Spacer(1, 10))

    # Section 8: Next Week Plan
    story.append(Paragraph("8. Next-Week Improvement Plan", h1_style))
    plans = [
        "<b>Blind Double-Annotation:</b> Annotate 1,000 cases blindly without silver pre-labels to eliminate confirmation bias.",
        "<b>URL Canonicalization:</b> Map raw 2017 t.co shortlinks to live support.apple.com canonical article endpoints.",
        "<b>Contrastive Fine-Tuning:</b> Fine-tune sentence-transformers using Multiple Negatives Ranking (MNR) loss on AppleSupport QA pairs.",
        "<b>LLM Response Filtering:</b> Filter out historical responses that are mere DM requests before populating vector index."
    ]
    for pl in plans:
        story.append(Paragraph(f"• {pl}", bullet_style))

    doc.build(story)
    print(f"PDF generated successfully at {output_path}")

if __name__ == '__main__':
    out_pdf = "d:/All_Project/Hiver_Submission/hiver-support-agent/reports/Hiver_AI_Support_Agent_Report.pdf"
    create_pdf(out_pdf)
