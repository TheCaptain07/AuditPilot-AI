# =============================
# FILE: app.py
# =============================
# Streamlit all-in-one portal for:
# 1) ISO 27001:2022 Readiness Assessment (questionnaire + scoring + gap analysis)
# 2) Risk Register & Heatmap (quantitative + CVSS-like scoring)
# 3) Log Anomaly Detection (IsolationForest)
# 4) AI Advisor (local rule-based + optional OpenAI API if key provided)
# 5) Report Generator (PDF via reportlab)
#
# One-file design for easy free hosting on Hugging Face Spaces.
# Repo only needs: app.py and requirements.txt
#
# Author: Your Name (Amoha Security Consultants)
# License: MIT

import os
import io
import json
import base64
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

# Optional OpenAI (only used if OPENAI_API_KEY present)
OPENAI_AVAILABLE = False
try:
    import openai  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        OPENAI_AVAILABLE = True
        openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="ISO 27001 Readiness & AI Cyber Portal", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0f172a 50%, #111827 100%);
}

/* Main container */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17,24,39,0.95);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
    padding: 18px;
    transition: transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: white !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
textarea {
    border-radius: 10px !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Progress bar */
[data-testid="stProgressBar"] div {
    border-radius: 999px;
}

/* Headings */
h1 {
    color: #60a5fa !important;
}

h2, h3 {
    color: #38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Data Models
# -----------------------------
@dataclass
class Question:
    id: str
    clause: str
    text: str
    domain: str
    weight: float = 1.0

# Minimal but representative question bank covering major Annex A themes
# Scale: 0 (No) – 5 (Fully Implemented)
QUESTION_BANK: List[Question] = [
    # Context & Leadership
    Question("Q1", "4/5", "Has the organization's context and interested parties been documented?", "Leadership"),
    Question("Q2", "5.3", "Are roles & responsibilities for information security formally assigned?", "Leadership"),
    # Planning & Risk
    Question("Q3", "6.1.2", "Is there a documented risk assessment methodology?", "Risk"),
    Question("Q4", "6.1.3", "Is a risk treatment plan approved and tracked?", "Risk"),
    Question("Q5", "A.5.1", "Are information security policies established and communicated?", "Policy"),
    # Asset
    Question("Q6", "A.5.9", "Is there an up-to-date asset inventory including owners?", "Asset Mgmt"),
    Question("Q7", "A.5.10", "Is information classified and handled per classification?", "Asset Mgmt"),
    # HR/Supplier
    Question("Q8", "A.6.3", "Are supplier security requirements defined and monitored?", "Supplier"),
    Question("Q9", "A.6.1", "Are background checks and terms for employees/contractors in place?", "HR"),
    # Access Control
    Question("Q10", "A.5.15", "Is IAM lifecycle (provisioning, reviews, deprovisioning) implemented?", "Access"),
    Question("Q11", "A.5.17", "Are privileged accounts controlled with MFA and monitoring?", "Access"),
    # Cryptography
    Question("Q12", "A.8.24", "Are cryptographic controls defined (algorithms, key mgmt, rotation)?", "Crypto"),
    # Physical
    Question("Q13", "A.7.1", "Are physical entry controls in place for sensitive areas?", "Physical"),
    # Operations
    Question("Q14", "A.8.4", "Are changes controlled with approvals & rollback plans?", "Ops"),
    Question("Q15", "A.8.5", "Are capacity/performance monitored with thresholds?", "Ops"),
    # Logging/Monitoring
    Question("Q16", "A.8.15", "Are logs collected, retained, and reviewed for critical systems?", "Monitoring"),
    # Backup
    Question("Q17", "A.8.13", "Are backups performed, tested, and protected (e.g., immutability)?", "Ops"),
    # Vulnerability Mgmt
    Question("Q18", "A.8.8", "Is vulnerability management scheduled with remediation SLAs?", "Vuln Mgmt"),
    # Development
    Question("Q19", "A.8.28", "Are secure dev practices (SAST/DAST/SDLC) implemented?", "DevSecOps"),
    # Incident
    Question("Q20", "A.5.24", "Is there an incident response plan with roles & exercises?", "IR"),
    # Business Continuity
    Question("Q21", "A.5.30", "Are BCM/DR plans documented and tested?", "BCM"),
    # Compliance
    Question("Q22", "9/10", "Are internal audits conducted and management reviews performed?", "Compliance"),
    # Cloud/SaaS
    Question("Q23", "A.5.23", "Are cloud security responsibilities clarified with providers?", "Cloud"),
    # Network
    Question("Q24", "A.8.20", "Are networks segmented and traffic filtered (FW/WAF/IDS)?", "Network"),
    # Endpoint
    Question("Q25", "A.8.7", "Are endpoints managed with EDR/AV and hardening baseline?", "Endpoint"),
    # Awareness
    Question("Q26", "A.6.2", "Are security awareness trainings periodic and tracked?", "Awareness"),
    # Data Protection
    Question("Q27", "A.5.34", "Is data protection by design/default addressed (privacy)?", "Privacy"),
    # Third Party Risk
    Question("Q28", "A.5.20", "Is third-party risk assessment conducted pre/post onboarding?", "TPRM"),
    # Secure Config
    Question("Q29", "A.8.9", "Are secure configuration baselines defined & enforced?", "Config"),
    # Patch Mgmt
    Question("Q30", "A.8.7/8", "Are patches prioritized by risk and applied within SLA?", "Patch Mgmt"),
]

DOMAIN_ORDER = [
    "Leadership","Risk","Policy","Asset Mgmt","Supplier","HR","Access","Crypto","Physical",
    "Ops","Monitoring","Vuln Mgmt","DevSecOps","IR","BCM","Compliance","Cloud","Network",
    "Endpoint","Awareness","Privacy","TPRM","Config","Patch Mgmt"
]

# Remediation knowledge base (simple templates used by AI advisor)
REMEDIATIONS: Dict[str, str] = {
    "Leadership": "Establish and communicate ISMS scope, assign accountable owners, and integrate objectives into KPIs.",
    "Risk": "Define risk criteria, implement consistent assessment, maintain a living risk treatment plan with owners and dates.",
    "Policy": "Draft policy set (IS policy, Acceptable Use, Access Control, Backup, IR, BYOD), approve and communicate.",
    "Asset Mgmt": "Build CMDB/asset register with owners, classification labels, and handling procedures.",
    "Supplier": "Define supplier security clauses, run due diligence, monitor SLAs and security attestations.",
    "HR": "Include background checks, security clauses, onboarding/offboarding checklists, and periodic awareness.",
    "Access": "Implement role-based access, periodic access reviews, MFA for privileged, and joiner-mover-leaver flows.",
    "Crypto": "Publish crypto standard (TLS1.2+, AES-256, RSA-2048+), manage keys in HSM/KMS with rotation.",
    "Physical": "Restrict physical access, maintain visitor logs, CCTV retention, and secure areas controls.",
    "Ops": "Use change management workflow, monitor capacity, protect backups (3-2-1, offline/immutable).",
    "Monitoring": "Centralize logs (SIEM), define use cases, alerting, and periodic review with incident tickets.",
    "Vuln Mgmt": "Schedule scans, track findings with SLAs by severity, verify remediation and exceptions.",
    "DevSecOps": "Integrate SAST/DAST/Secrets scanning, code reviews, and pipeline gates with SBOM.",
    "IR": "Maintain incident runbooks, RACI, exercises, and post-incident reviews with lessons learned.",
    "BCM": "Conduct BIA, define RTO/RPO, run DR drills, and maintain communication plans.",
    "Compliance": "Plan internal audits, corrective actions, and management reviews with metrics.",
    "Cloud": "Define shared responsibility, harden cloud configs (CIS), use CSPM and IAM least privilege.",
    "Network": "Segment networks, enforce FW/WAF, IDS/IPS, and secure remote access.",
    "Endpoint": "Deploy EDR/AV, hardening baselines, disk encryption, and patching automation.",
    "Awareness": "Run role-based training, phishing simulations, and track completion.",
    "Privacy": "Map personal data, DPIAs, consent, retention schedules, and data subject rights.",
    "TPRM": "Standardize due diligence questionnaires, risk tiering, and continuous monitoring.",
    "Config": "Define hardened baselines (CIS), config drift monitoring, and periodic audits.",
    "Patch Mgmt": "Risk-based prioritization, maintenance windows, and emergency patch process."
}

# -----------------------------
# Utility
# -----------------------------

def _badge(text, color="gray"):
    st.markdown(f"<span style='background:{color};padding:4px 8px;border-radius:8px;color:white;font-size:12px'>{text}</span>", unsafe_allow_html=True)


def weighted_score(responses: Dict[str, int]) -> Dict[str, Any]:
    domain_scores: Dict[str, List[float]] = {d: [] for d in DOMAIN_ORDER}
    for q in QUESTION_BANK:
        v = responses.get(q.id, 0)
        domain_scores[q.domain].append(v * q.weight)
    domain_avgs = {d: (np.mean(v) if len(v) else 0.0) for d, v in domain_scores.items()}
    overall = float(np.mean(list(domain_avgs.values()))) if len(domain_avgs) else 0.0
    return {"domain_avgs": domain_avgs, "overall": overall}


def maturity_label(x: float) -> str:
    if x >= 4.5: return "Optimized (5)"
    if x >= 3.5: return "Managed (4)"
    if x >= 2.5: return "Defined (3)"
    if x >= 1.5: return "Repeatable (2)"
    if x > 0: return "Initial (1)"
    return "Not Implemented (0)"


def gen_pdf_report(org, assessor, domain_avgs, overall, gaps):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER

    elements = []

    elements.append(
        Paragraph(
            "AI-Enabled ISO 27001 Readiness Assessment Report",
            title_style
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph(f"<b>Organization:</b> {org}", styles["Normal"])
    )

    elements.append(
        Paragraph(f"<b>Assessor:</b> {assessor}", styles["Normal"])
    )

    elements.append(
        Paragraph(
            f"<b>Overall Readiness:</b> {overall * 20:.1f}%",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph(
            "<b>Domain-wise Assessment</b>",
            styles["Heading2"]
        )
    )

    table_data = [["Domain", "Score", "Status"]]

    for domain, score in domain_avgs.items():

        if score >= 4:
            status = "Good"

        elif score >= 3:
            status = "Moderate"

        else:
            status = "Needs Improvement"

        table_data.append([
            domain,
            f"{score:.2f}",
            status
        ])

    table = Table(table_data)

    table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("GRID", (0, 0), (-1, -1), 1, colors.grey),

        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),

        ("ALIGN", (0, 0), (-1, -1), "CENTER"),

        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),

    ]))

    elements.append(table)

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph(
            "<b>AI Recommendations</b>",
            styles["Heading2"]
        )
    )

    if gaps:

        for domain, recommendation in gaps.items():

            elements.append(

                Paragraph(

                    f"• <b>{domain}</b>: {recommendation}",

                    styles["Normal"]

                )

            )

    else:

        elements.append(

            Paragraph(

                "No significant gaps identified. Continue continuous improvement.",

                styles["Normal"]

            )

        )

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(

        Paragraph(

            "Generated by AuditPilot AI",

            styles["Italic"]

        )

    )

    doc.build(elements)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf


def split_text(text: str, width_chars: int) -> List[str]:
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) for x in line) + len(line) + len(w) <= width_chars:
            line.append(w)
        else:
            lines.append(" ".join(line)); line = [w]
    if line:
        lines.append(" ".join(line))
    return lines

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🛡️ Navigation")
PAGES = [
    "🏠 Home",
    "📝 Assessment",
    "📊 Risk Register",
    "🔎 Log Anomalies",
    "🧠 AI Advisor",
    "📄 Report"
]
st.sidebar.info("""
AI-Powered ISO 27001
Readiness Assessment

Features:
- Readiness Score
- Log Anomaly Detection
- Risk Register
- AI Advisor
- PDF Report
""")
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption("Stronger, more Independent Audit System.")

# Session state
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "domain_avgs" not in st.session_state:
    st.session_state.domain_avgs = {}
if "overall" not in st.session_state:
    st.session_state.overall = 0.0
if "gaps" not in st.session_state:
    st.session_state.gaps = {}

# -----------------------------
# Home
# -----------------------------
# -----------------------------
# Home
# -----------------------------
if page == "🏠 Home":

    # Calculate live metrics
    overall_score = st.session_state.get("overall", 0.0)
    overall_percent = round((overall_score / 5.0) * 100, 1)

    weak_domains = len([
        d for d, v in st.session_state.get("domain_avgs", {}).items()
        if v < 3.0
    ])

    total_controls = len(QUESTION_BANK)

    # Hero
    st.markdown("""
    # 🛡️ AuditPilot AI

    ### AI-Powered ISO 27001 Readiness & Cyber Risk Assessment Platform

    Transform manual security audits into intelligent, data-driven compliance assessments using Artificial Intelligence and Machine Learning.
    """)

    st.divider()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🛡️ Readiness",
            f"{overall_percent:.1f}%"
        )

    with col2:
        st.metric(
            "🚨 Weak Domains",
            weak_domains
        )

    with col3:
        st.metric(
            "📋 ISO Controls",
            total_controls
        )

    with col4:
        st.metric(
            "🤖 AI Engine",
            "Active"
        )

    st.divider()

    # Progress
    st.subheader("Overall ISO 27001 Readiness")

    st.progress(overall_percent / 100)

    if overall_percent >= 80:
        st.success("Excellent readiness level. The organization demonstrates strong alignment with ISO 27001 controls.")

    elif overall_percent >= 60:
        st.warning("Moderate readiness level. Some security domains require improvement.")

    else:
        st.error("Low readiness level. Significant remediation is recommended before certification.")

    st.divider()

    left, right = st.columns([2, 1])

    with left:

        st.subheader("🚀 Platform Features")

        st.markdown("""
- ✅ ISO 27001:2022 Readiness Assessment
- 🤖 AI-Based Log Anomaly Detection
- 📊 Interactive Risk Register & Heatmap
- 🧠 Automated Security Recommendations
- 📄 Executive PDF Report Generation
- ☁️ Cloud Deployment Ready
        """)

    with right:

        st.subheader("📈 Executive Summary")

        st.info(f"""
**Current Readiness:** {overall_percent:.1f}%

**Weak Domains:** {weak_domains}

**Assessment Controls:** {total_controls}

**Machine Learning:** Isolation Forest

**Status:** Ready for Assessment
        """)

    st.divider()

    st.subheader("💡 Why AuditPilot AI?")

    st.write("""
Traditional ISO 27001 assessments are often manual, time-consuming, and subjective.

AuditPilot AI enhances the process by combining compliance evaluation with Artificial Intelligence. The platform analyzes ISO control maturity, detects anomalies in uploaded log files, identifies weak security domains, and provides actionable remediation guidance to support faster and more informed decision-making.
    """)

    st.caption("Built using Python • Streamlit • Scikit-learn • Pandas • NumPy • ReportLab")

    st.subheader("🤖 AI Executive Verdict")

overall = (st.session_state.overall / 5) * 100

if overall >= 85:

    st.success("""
The organization demonstrates a mature cybersecurity posture and is well-positioned for ISO 27001 readiness. Focus should remain on continuous improvement and periodic audits.
""")

elif overall >= 70:

    st.info("""
The organization has a solid foundation but should strengthen weaker domains to improve resilience and certification readiness.
""")

elif overall >= 50:

    st.warning("""
Several control areas require attention. Prioritized remediation and stronger governance are recommended before certification efforts.
""")

else:

    st.error("""
The current security posture indicates significant compliance gaps. Immediate corrective actions should be taken to reduce organizational risk.
""")
# -----------------------------
# -----------------------------
# Assessment Page
# -----------------------------
if page == "📝 Assessment":

    st.title("📝 ISO 27001:2022 Readiness Assessment")
    st.caption("Evaluate your organization's implementation maturity across key ISO/IEC 27001 controls.")

    # Organization details
    col1, col2 = st.columns(2)

    with col1:
        org = st.text_input(
            "🏢 Organization Name",
            value=st.session_state.get("org", "Acme Corp")
        )

    with col2:
        assessor = st.text_input(
            "👨‍💼 Assessor Name",
            value=st.session_state.get("assessor", "Harsh Shah")
        )

    st.session_state.org = org
    st.session_state.assessor = assessor

    st.info("""
Rate each control from:

0️⃣ Not Implemented

1️⃣ Initial

2️⃣ Repeatable

3️⃣ Defined

4️⃣ Managed

5️⃣ Optimized
""")

    # Progress
    completed = sum(
        1 for q in QUESTION_BANK
        if st.session_state.responses.get(q.id, 0) > 0
    )

    progress = completed / len(QUESTION_BANK)

    st.progress(progress)

    st.caption(
        f"Progress: {completed}/{len(QUESTION_BANK)} controls completed"
    )

    st.divider()

    # Prevent reruns while sliding
    with st.form("assessment_form"):

        cols = st.columns(2)

        for i, q in enumerate(QUESTION_BANK):

            with cols[i % 2]:

                st.session_state.responses[q.id] = st.slider(
                    label=f"{q.text}",
                    min_value=0,
                    max_value=5,
                    value=int(
                        st.session_state.responses.get(q.id, 0)
                    ),
                    help=f"Clause: {q.clause} | Domain: {q.domain}"
                )

        submitted = st.form_submit_button(
            "🚀 Calculate Maturity Score"
        )

    if submitted:

        result = weighted_score(
            st.session_state.responses
        )

        st.session_state.domain_avgs = result["domain_avgs"]
        st.session_state.overall = result["overall"]

        gaps = {
            d: (
                f"Maturity {v:.2f} → "
                f"{REMEDIATIONS.get(d,'Implement Annex A controls.')}"
            )
            for d, v in result["domain_avgs"].items()
            if v < 3.0
        }

        st.session_state.gaps = gaps

        st.success("✅ Assessment Completed Successfully")

    if st.session_state.domain_avgs:

        st.divider()

        st.subheader("📊 Results Summary")

        overall_percent = (
            st.session_state.overall / 5
        ) * 100

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Overall Readiness",
                f"{overall_percent:.1f}%"
            )

        with c2:
            st.metric(
                "Weak Domains",
                len(st.session_state.gaps)
            )

        with c3:
            st.metric(
                "Controls Assessed",
                len(QUESTION_BANK)
            )

        df = pd.DataFrame({

            "Domain": list(
                st.session_state.domain_avgs.keys()
            ),

            "Score": list(
                st.session_state.domain_avgs.values()
            ),

            "Maturity": [

                maturity_label(x)

                for x in
                st.session_state.domain_avgs.values()

            ]

        })

        st.dataframe(
            df.sort_values("Score"),
            use_container_width=True
        )

        st.subheader("🔴 Domains Requiring Attention")

        if st.session_state.gaps:

            for domain, rec in st.session_state.gaps.items():

                st.warning(
                    f"**{domain}**\n\n{rec}"
                )

        else:

            st.success(
                "🎉 Great! No significant gaps detected."
            )
# -----------------------------
# Risk Register Page
# -----------------------------
# -----------------------------
# Risk Register Page
# -----------------------------
if page == "📊 Risk Register":

    st.title("📊 Cyber Risk Register & Heatmap")
    st.caption(
        "Quantify and visualize cybersecurity risks using likelihood × impact scoring."
    )

    uploaded = st.file_uploader(
        "Upload Risk Register CSV (optional)",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame({
            "Asset": [
                "ERP Server",
                "HR Portal",
                "Database",
                "Email Gateway",
                "Cloud Storage"
            ],
            "Threat": [
                "Ransomware",
                "Account Takeover",
                "Data Breach",
                "Phishing",
                "Misconfiguration"
            ],
            "Likelihood": [3, 4, 2, 5, 3],
            "Impact": [5, 4, 5, 3, 4]
        })

    st.subheader("📝 Edit Risk Register")

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True
    )

    edited["Risk Score"] = (
        edited["Likelihood"] * edited["Impact"]
    )

    def classify(score):
        if score >= 20:
            return "🔴 Critical"
        elif score >= 12:
            return "🟠 High"
        elif score >= 6:
            return "🟡 Medium"
        return "🟢 Low"

    edited["Risk Level"] = edited["Risk Score"].apply(classify)

    st.divider()

    # Executive KPIs
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Risks",
            len(edited)
        )

    with c2:
        st.metric(
            "Critical",
            (edited["Risk Score"] >= 20).sum()
        )

    with c3:
        st.metric(
            "High",
            (
                (edited["Risk Score"] >= 12)
                &
                (edited["Risk Score"] < 20)
            ).sum()
        )

    with c4:
        avg_risk = round(
            edited["Risk Score"].mean(),
            1
        )

        st.metric(
            "Average Score",
            avg_risk
        )

    st.divider()

    st.subheader("📋 Risk Register")

    st.dataframe(
        edited,
        use_container_width=True
    )

    st.divider()

    st.subheader("🔥 Risk Heatmap")

    import matplotlib.pyplot as plt

    heat = np.zeros((5, 5))

    for _, row in edited.iterrows():

try:
    l = int(float(row["Likelihood"]))
    i = int(float(row["Impact"]))

    l = max(1, min(l, 5))
    i = max(1, min(i, 5))

except:
    l = 1
    i = 1

        heat[5 - i, l - 1] += 1

    fig = plt.figure(figsize=(6, 5))

    plt.imshow(heat)

    plt.xticks(
        range(5),
        [1, 2, 3, 4, 5]
    )

    plt.yticks(
        range(5),
        [5, 4, 3, 2, 1]
    )

    plt.xlabel("Likelihood")

    plt.ylabel("Impact")

    plt.title("Cyber Risk Matrix")

    st.pyplot(fig)

    st.divider()

    st.subheader("🤖 Executive Recommendation")

    highest = edited.sort_values(
        "Risk Score",
        ascending=False
    ).head(3)

    for _, row in highest.iterrows():

        st.warning(
            f"""
**{row['Asset']}**

Threat: **{row['Threat']}**

Risk Score: **{row['Risk Score']}**

Recommendation:
Prioritize mitigation, implement monitoring,
and assign an owner with a remediation timeline.
"""
        )
# -----------------------------
# Log Anomalies Page
# -----------------------------
# -----------------------------
# Log Anomaly Detection Page
# -----------------------------
if page == "🔎 Log Anomalies":

    st.title("🔎 AI-Powered Log Anomaly Detection")
    st.caption(
        "Upload system logs and let the AI engine detect suspicious events using Isolation Forest."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload CSV Log File",
        type=["csv"]
    )

    if uploaded_file:

        logs = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Log Preview")
        st.dataframe(logs.head(), use_container_width=True)

        features = pd.DataFrame()

        # Numeric columns
        if "bytes" in logs.columns:
            features["bytes"] = pd.to_numeric(
                logs["bytes"],
                errors="coerce"
            ).fillna(0)

        if "status" in logs.columns:
            features["status"] = pd.to_numeric(
                logs["status"],
                errors="coerce"
            ).fillna(0)

        # TF-IDF on action
        if "action" in logs.columns:

            tfidf = TfidfVectorizer(max_features=20)

            action_matrix = tfidf.fit_transform(
                logs["action"].fillna("")
            ).toarray()

            action_df = pd.DataFrame(
                action_matrix,
                columns=[
                    f"action_{i}"
                    for i in range(action_matrix.shape[1])
                ]
            )

            features = pd.concat(
                [features.reset_index(drop=True), action_df],
                axis=1
            )

        if features.empty:

            st.error(
                "No usable columns found. Please include bytes/status/action."
            )

        else:

            scaler = StandardScaler()

            X = scaler.fit_transform(features)

            model = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            preds = model.fit_predict(X)

            scores = -model.score_samples(X)

            logs["Anomaly Score"] = scores
            logs["AI Prediction"] = np.where(
                preds == -1,
                "🚨 Suspicious",
                "✅ Normal"
            )

            anomaly_count = (
                logs["AI Prediction"] == "🚨 Suspicious"
            ).sum()

            total_logs = len(logs)

            normal_count = total_logs - anomaly_count

            st.divider()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Total Events",
                    total_logs
                )

            with c2:
                st.metric(
                    "Suspicious Events",
                    anomaly_count
                )

            with c3:
                pct = round(
                    anomaly_count / total_logs * 100,
                    2
                )

                st.metric(
                    "Anomaly %",
                    f"{pct}%"
                )

            st.divider()

            st.subheader("🚨 Top Suspicious Logs")

            suspicious = logs[
                logs["AI Prediction"] == "🚨 Suspicious"
            ].sort_values(
                "Anomaly Score",
                ascending=False
            )

            st.dataframe(
                suspicious,
                use_container_width=True
            )

            st.divider()

            st.subheader("📊 AI Detection Summary")

            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(5, 4))

            plt.bar(
                ["Normal", "Suspicious"],
                [normal_count, anomaly_count]
            )

            plt.title("AI Classification Results")

            st.pyplot(fig)

            st.divider()

            st.subheader("🤖 AI Interpretation")

            if anomaly_count == 0:

                st.success(
                    """
No major anomalies detected.

The uploaded logs appear consistent with expected behaviour.
                    """
                )

            else:

                st.warning(
                    f"""
The AI model detected **{anomaly_count} suspicious events**.

Possible reasons include:
- Multiple failed logins
- Abnormal file transfers
- Privileged access attempts
- Configuration changes
- Unusual activity patterns

Recommendation:
Investigate these events and correlate them with SIEM or SOC alerts.
                    """
                )

    else:

        st.info(
            """
Upload a CSV containing:

- timestamp
- user
- action
- bytes
- status
- src_ip
- dst_ip

The Isolation Forest model will automatically identify anomalous behaviour.
            """
        )
# -----------------------------
# AI Advisor Page
# -----------------------------
# -----------------------------
# AI Advisor Page
# -----------------------------
if page == "🧠 AI Advisor":

    st.title("🧠 AI Security Advisor")
    st.caption(
        "Get AI-assisted remediation recommendations based on your ISO 27001 assessment."
    )

    if not st.session_state.domain_avgs:
        st.warning(
            "⚠️ Please complete the ISO 27001 Assessment first."
        )

    else:

        overall_percent = round(
            (st.session_state.overall / 5) * 100,
            1
        )

        st.metric(
            "Overall Readiness",
            f"{overall_percent}%"
        )

        st.divider()

        st.subheader("🔴 High Priority Actions")

        high_priority = []

        medium_priority = []

        low_priority = []

        for domain, score in st.session_state.domain_avgs.items():

            if score < 2:

                high_priority.append(domain)

            elif score < 3.5:

                medium_priority.append(domain)

            else:

                low_priority.append(domain)

        if high_priority:

            for domain in high_priority:

                st.error(
                    f"""
### {domain}

{REMEDIATIONS.get(domain)}

**Priority:** Immediate

**Suggested Timeline:** 30 Days
                    """
                )

        else:

            st.success(
                "No critical weaknesses detected."
            )

        st.divider()

        st.subheader("🟡 Medium Priority Improvements")

        if medium_priority:

            for domain in medium_priority:

                st.warning(
                    f"""
### {domain}

{REMEDIATIONS.get(domain)}

**Suggested Timeline:** 60–90 Days
                    """
                )

        else:

            st.success(
                "No medium-priority findings."
            )

        st.divider()

        st.subheader("🟢 Strong Domains")

        if low_priority:

            for domain in low_priority:

                st.success(
                    f"""
### {domain}

Current maturity is satisfactory.

Recommendation:
Continue monitoring and perform periodic reviews.
                    """
                )

        st.divider()

        st.subheader("🤖 Executive AI Summary")

        weak = len(high_priority)

        moderate = len(medium_priority)

        if weak == 0:

            risk_level = "Low"

        elif weak <= 3:

            risk_level = "Moderate"

        else:

            risk_level = "High"

        st.info(
            f"""
### Overall Security Posture

• ISO 27001 Readiness: **{overall_percent}%**

• Critical Weak Domains: **{weak}**

• Medium Priority Domains: **{moderate}**

• Estimated Organizational Risk: **{risk_level}**

### AI Recommendation

Prioritize remediation of low-scoring domains,
strengthen access management and incident response,
perform regular vulnerability assessments,
and maintain continuous compliance monitoring.
            """
        )
    st.subheader("🤖 AI Recommendations")

for domain, score in st.session_state.domain_avgs.items():

    if score < 2:
        st.error(
            f"""
### 🔴 Critical - {domain}

• Immediate action required.
• {REMEDIATIONS.get(domain, 'Implement missing controls immediately.')}
• Recommended timeline: Within 30 days.
"""
        )

    elif score < 3:
        st.warning(
            f"""
### 🟠 High - {domain}

• Controls need strengthening.
• {REMEDIATIONS.get(domain, 'Review and improve this domain.')}
• Recommended timeline: Within 60 days.
"""
        )

    elif score < 4:
        st.info(
            f"""
### 🟡 Medium - {domain}

• Continue improving documentation and monitoring.
• Schedule periodic reviews.
"""
        )

    else:
        st.success(
            f"""
### 🟢 Good - {domain}

Current maturity is satisfactory.
Continue monitoring and continuous improvement.
"""
        )
        st.subheader("📋 Recommended Roadmap")

        roadmap = pd.DataFrame({

            "Timeline": [
                "0–30 Days",
                "30–60 Days",
                "60–90 Days",
                "Quarterly"
            ],

            "Recommended Action": [

                "Fix critical gaps and enforce MFA",

                "Review policies and supplier security",

                "Conduct vulnerability assessments",

                "Perform internal audits and management reviews"

            ]

        })

        st.dataframe(
            roadmap,
            use_container_width=True
        )
# -----------------------------
# Report Page
# -----------------------------
# -----------------------------
# Report Page
# -----------------------------
# ==========================================================
# REPORT PAGE
# ==========================================================

if page == "📄 Report":

    st.title("📄 Executive ISO 27001 Assessment Report")

    if not st.session_state.domain_avgs:

        st.warning(
            "Please complete the Assessment page first."
        )

    else:

        overall_percent = round(
            st.session_state.overall * 20,
            1
        )

        st.subheader("📊 Executive Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Overall Readiness",
                f"{overall_percent}%"
            )

        with c2:
            st.metric(
                "Weak Domains",
                len(st.session_state.gaps)
            )

        with c3:

            if overall_percent >= 80:
                status = "🟢 Good"

            elif overall_percent >= 60:
                status = "🟡 Moderate"

            else:
                status = "🔴 Needs Improvement"

            st.metric(
                "Security Status",
                status
            )

        st.divider()

        st.subheader("📋 Domain Scores")

        report_df = pd.DataFrame({

            "Domain":
                list(
                    st.session_state.domain_avgs.keys()
                ),

            "Score":
                list(
                    st.session_state.domain_avgs.values()
                )

        })

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.divider()

        st.subheader("🤖 AI Recommendations")

        for domain, score in st.session_state.domain_avgs.items():

            if score < 2:

                st.error(
                    f"""
🔴 **{domain}**

Immediate remediation required.

Recommendation:

{REMEDIATIONS.get(domain,
'Implement missing controls immediately.')}
"""
                )

            elif score < 3:

                st.warning(
                    f"""
🟠 **{domain}**

Needs improvement.

Recommendation:

{REMEDIATIONS.get(domain,
'Strengthen controls and documentation.')}
"""
                )

            elif score < 4:

                st.info(
                    f"""
🟡 **{domain}**

Continue improving monitoring and governance.
"""
                )

            else:

                st.success(
                    f"""
🟢 **{domain}**

Current implementation appears satisfactory.
Maintain continuous improvement.
"""
                )

        st.divider()

    pdf_bytes = gen_pdf_report(
    st.session_state.get("org", "N/A"),
    st.session_state.get("assessor", "N/A"),
    st.session_state.domain_avgs,
    st.session_state.overall,
    st.session_state.gaps
    )

    st.download_button(
    "📄 Download Executive PDF",
    data=pdf_bytes,
    file_name="AuditPilot_AI_Report.pdf",
    mime="application/pdf"
    )

# END OF FILE app.py
# =============================
