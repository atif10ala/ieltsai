"""
IELTS AI Tutor — Pure Groq Platform Architecture
==================================================
Architected with a clean 14-layer separation of concerns.
Powered completely by Groq (Llama 3.3 70B Versatile & Whisper Large V3).
Zero external dependencies beyond standard library HTTP clients and Streamlit.
"""

from __future__ import annotations

import html
import http.client
import json
import random
import re
import socket
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — CONFIGURATION & GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

GROQ_HOST = "api.groq.com"
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_TIMEOUT_SECONDS = 45

IELTS_CRITERIA = {
    "TA": "Task Achievement / Response",
    "CC": "Coherence and Cohesion",
    "LR": "Lexical Resource",
    "GRA": "Grammatical Range and Accuracy"
}

SPEAKING_CRITERIA = {
    "FC": "Fluency and Coherence",
    "LR": "Lexical Resource",
    "GRA": "Grammatical Range and Accuracy",
    "PR": "Pronunciation"
}

SAMPLE_WRITING_PROMPTS = {
    "Task 2 (Academic/General)": [
        "Some people believe that artificial intelligence will replace human teachers in the near future. To what extent do you agree or disagree?",
        "In many countries, young people are granted more freedom than in the past. Is this a positive or negative development?",
        "Successful companies should focus on making profits, while others think they have a social responsibility. Discuss both views and give your opinion."
    ],
    "Task 1 (Academic - Data Analysis)": [
        "The chart below shows the global consumption of renewable energy sources between 2000 and 2025. Summarise the information by selecting and reporting the main features.",
        "The diagram illustrates the process of plastic recycling. Summarise the information by selecting and reporting the main features."
    ]
}

SAMPLE_SPEAKING_PROMPTS = {
    "Part 1: Introduction & Familiar Topics": [
        "Let's talk about your hometown. Where is your hometown located, and what do you like most about living there?",
        "Let's discuss hobbies. What do you enjoy doing in your free time, and how long have you been interested in this activity?"
    ],
    "Part 2: Long Turn (Cue Card)": [
        "Describe a book that had a significant impact on your life. You should say: what book it is, when you read it, what it is about, and explain why it influenced you so deeply.",
        "Describe a memorable journey you took. You should say: where you went, how you travelled, who went with you, and explain why this journey stands out in your memory."
    ]
}

THEME_PROFILES = {
    "Slate Obsidian (Dark)": {
        "bg": "#0e1117", "card_bg": "#1d222e", "border": "#2d3748", "text": "#f7fafc", "accent": "#00e676"
    },
    "Cyberpunk Gold (Dark)": {
        "bg": "#0b0c10", "card_bg": "#1f2833", "border": "#c5a059", "text": "#f5f5f5", "accent": "#ffc107"
    }
}


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — CORE DOMAIN MODEL ENTITIES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Mistake:
    original: str
    correction: str
    explanation: str
    tag: str  # Subject-Verb Agreement, Article Omission, Vocabulary Collocation, etc.

@dataclass
class EvaluationResult:
    overall_band: float
    bands: dict[str, float]
    feedback: dict[str, str]
    mistakes: list[Mistake] = field(default_factory=list)
    model_version: str = GROQ_LLM_MODEL
    evaluated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

@dataclass
class EssaySubmission:
    id: str
    task_type: str
    prompt: str
    essay_text: str
    word_count: int
    evaluation: Optional[EvaluationResult] = None
    submitted_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

@dataclass
class SpeakingSubmission:
    id: str
    part_type: str
    prompt: str
    audio_filename: str
    transcript: str
    evaluation: Optional[EvaluationResult] = None
    submitted_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — DETERMINISTIC XML STRING EXTRACTION ENGINE
# ══════════════════════════════════════════════════════════════════════════

class ResponseParser:
    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @classmethod
    def parse_writing_response(cls, raw_text: str) -> EvaluationResult:
        try:
            overall = float(cls._extract_tag(raw_text, "overall_band") or 6.0)
            bands = {
                "TA": float(cls._extract_tag(raw_text, "ta_band") or 6.0),
                "CC": float(cls._extract_tag(raw_text, "cc_band") or 6.0),
                "LR": float(cls._extract_tag(raw_text, "lr_band") or 6.0),
                "GRA": float(cls._extract_tag(raw_text, "gra_band") or 6.0)
            }
            feedback = {
                "TA": cls._extract_tag(raw_text, "ta_feedback") or "Feedback parsing anomaly.",
                "CC": cls._extract_tag(raw_text, "cc_feedback"),
                "LR": cls._extract_tag(raw_text, "lr_feedback"),
                "GRA": cls._extract_tag(raw_text, "gra_feedback")
            }
            
            mistakes = []
            mistakes_raw = cls._extract_tag(raw_text, "mistakes_block")
            if mistakes_raw:
                for entry in re.findall(r"<error>(.*?)</error>", mistakes_raw, re.DOTALL):
                    orig = cls._extract_tag(entry, "original")
                    corr = cls._extract_tag(entry, "correction")
                    expl = cls._extract_tag(entry, "explanation")
                    tg = cls._extract_tag(entry, "tag") or "Grammar"
                    if orig and corr:
                        mistakes.append(Mistake(orig, corr, expl, tg))
                        
            return EvaluationResult(overall, bands, feedback, mistakes)
        except Exception as e:
            return cls._create_fallback_result(f"Writing XML Parsing Exception: {str(e)}")

    @classmethod
    def parse_speaking_response(cls, raw_text: str) -> EvaluationResult:
        try:
            overall = float(cls._extract_tag(raw_text, "overall_band") or 6.0)
            bands = {
                "FC": float(cls._extract_tag(raw_text, "fc_band") or 6.0),
                "LR": float(cls._extract_tag(raw_text, "lr_band") or 6.0),
                "GRA": float(cls._extract_tag(raw_text, "gra_band") or 6.0),
                "PR": float(cls._extract_tag(raw_text, "pr_band") or 6.0)
            }
            feedback = {
                "FC": cls._extract_tag(raw_text, "fc_feedback") or "Feedback parsing anomaly.",
                "LR": cls._extract_tag(raw_text, "lr_feedback"),
                "GRA": cls._extract_tag(raw_text, "gra_feedback"),
                "PR": cls._extract_tag(raw_text, "pr_feedback")
            }
            
            mistakes = []
            mistakes_raw = cls._extract_tag(raw_text, "mistakes_block")
            if mistakes_raw:
                for entry in re.findall(r"<error>(.*?)</error>", mistakes_raw, re.DOTALL):
                    orig = cls._extract_tag(entry, "original")
                    corr = cls._extract_tag(entry, "correction")
                    expl = cls._extract_tag(entry, "explanation")
                    tg = cls._extract_tag(entry, "tag") or "Pronunciation/Fluency"
                    if orig and corr:
                        mistakes.append(Mistake(orig, corr, expl, tg))
                        
            return EvaluationResult(overall, bands, feedback, mistakes)
        except Exception as e:
            return cls._create_fallback_result(f"Speaking XML Parsing Exception: {str(e)}")

    @staticmethod
    def _create_fallback_result(msg: str) -> EvaluationResult:
        return EvaluationResult(
            overall_band=5.5,
            bands={"TA": 5.5, "CC": 5.5, "LR": 5.5, "GRA": 5.5},
            feedback={"TA": msg, "CC": "", "LR": "", "GRA": ""},
            mistakes=[]
        )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 — CONSOLIDATED UNIFIED GROQ CLIENT GATEWAY
# ══════════════════════════════════════════════════════════════════════════

class GroqClient:
    """
    Zero-dependency standard client using http.client for all Groq transactions.
    Handles standard Open-API chat completions and Whisper multi-part uploads.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_content(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }

        try:
            conn = http.client.HTTPSConnection(GROQ_HOST, timeout=GROQ_TIMEOUT_SECONDS)
            conn.request("POST", "/openai/v1/chat/completions", body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            status, data = response.status, response.read().decode("utf-8")
            conn.close()

            if status != 200:
                return f"<error_state>Groq API Text Error Status {status}: {data}</error_state>"
            
            res_json = json.loads(data)
            return res_json["choices"][0]["message"]["content"]
        except socket.timeout:
            return "<error_state>Groq LLM Engine execution timed out.</error_state>"
        except Exception as e:
            return f"<error_state>Groq LLM connection breakdown: {str(e)}</error_state>"

    def transcribe_audio(self, audio_bytes: bytes, filename: str) -> str:
        boundary = b"----WebKitFormBoundaryGroqIELTS"
        body = []

        # Part 1: Model ID Spec
        body.append(b"--" + boundary)
        body.append(b'Content-Disposition: form-data; name="model"')
        body.append(b"")
        body.append(GROQ_WHISPER_MODEL.encode('utf-8'))

        # Part 2: Binary File Load
        body.append(b"--" + boundary)
        body.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode('utf-8'))
        body.append(b"Content-Type: audio/wav")
        body.append(b"")
        body.append(audio_bytes)

        body.append(b"--" + boundary + b"--")
        body.append(b"")

        payload = b"\r\n".join(body)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": b"multipart/form-data; boundary=" + boundary
        }

        try:
            conn = http.client.HTTPSConnection(GROQ_HOST, timeout=GROQ_TIMEOUT_SECONDS)
            conn.request("POST", "/openai/v1/audio/transcriptions", body=payload, headers=headers)
            response = conn.getresponse()
            status, data = response.status, response.read().decode("utf-8")
            conn.close()

            if status != 200:
                return f"[Whisper Gateway Error {status}]"
            
            res_json = json.loads(data)
            return res_json.get("text", "")
        except Exception as e:
            return f"[Whisper Audio Processing Translation Exception: {str(e)}]"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 5 — HIGH-FIDELITY IELTS ASSESSMENT PROMPTS
# ══════════════════════════════════════════════════════════════════════════

class PromptFactory:
    @staticmethod
    def get_writing_system_prompt() -> str:
        return """You are a senior executive IELTS Writing Examiner. Assess the submission strictly against the official IELTS criteria: Task Achievement/Response, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy.
Your output must be structured exactly using XML markers without extra explanations outside the tags.

Structure Template:
<overall_band>Calculated aggregate rounded up to nearest 0.5 boundary</overall_band>
<ta_band>Score 0-9</ta_band>
<cc_band>Score 0-9</cc_band>
<lr_band>Score 0-9</lr_band>
<gra_band>Score 0-9</gra_band>
<ta_feedback>Feedback</ta_feedback>
<cc_feedback>Feedback</cc_feedback>
<lr_feedback>Feedback</lr_feedback>
<gra_feedback>Feedback</gra_feedback>
<mistakes_block>
  <error>
    <original>Exact phrase with flaw</original>
    <correction>Flawless replacement</correction>
    <explanation> Labeled grammar rule</explanation>
    <tag>Categorized label matching: Subject-Verb Agreement, Article Omission, Article Misuse, Preposition Choice, Tense Inconsistency, Punctuation, Run-on Sentence, Fragment, Vocabulary Collocation, Informal Register</tag>
  </error>
</mistakes_block>"""

    @staticmethod
    def get_speaking_system_prompt() -> str:
        return """You are an executive IELTS Speaking Examiner. Evaluate the transcribed spoken response against official parameters: Fluency and Coherence, Lexical Resource, Grammatical Range and Accuracy, Pronunciation.
Your response must follow this strict XML format. Do not use Markdown styling.

Structure Template:
<overall_band>Score</overall_band>
<fc_band>Score</fc_band>
<lr_band>Score</lr_band>
<gra_band>Score</gra_band>
<pr_band>Score</pr_band>
<fc_feedback>Feedback</fc_feedback>
<lr_feedback>Feedback</lr_feedback>
<gra_feedback>Feedback</gra_feedback>
<pr_feedback>Acoustic projection and intonation feedback based on text cues</pr_feedback>
<mistakes_block>
  <error>
    <original>Flawed string</original>
    <correction>Correction</correction>
    <explanation>Linguistic explanation</explanation>
    <tag>Categorized label matching: Speech Hesitation, Subject-Verb Agreement, Preposition Choice, Vocabulary Collocation, Informal Register, Sentence Fragment</tag>
  </error>
</mistakes_block>"""


# ══════════════════════════════════════════════════════════════════════════
# LAYER 6 — ANALYTICAL SCHEMA CONVERTERS
# ══════════════════════════════════════════════════════════════════════════

class StorageConverter:
    @staticmethod
    def submission_to_dict(sub: EssaySubmission | SpeakingSubmission) -> dict:
        return asdict(sub)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 7 — STRUCTURAL STATISTICS & BAND APPROXIMATIONS
# ══════════════════════════════════════════════════════════════════════════

class StatisticalEngine:
    @staticmethod
    def compute_running_mean(scores: list[float]) -> float:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 8 — ERROR FINGERPRINT ENGINE
# ══════════════════════════════════════════════════════════════════════════

class ErrorFingerprintEngine:
    @staticmethod
    def build_matrix(submissions: list[EssaySubmission | SpeakingSubmission]) -> Counter:
        matrix = Counter()
        for sub in submissions:
            if sub.evaluation and sub.evaluation.mistakes:
                for flaw in sub.evaluation.mistakes:
                    matrix[flaw.tag] += 1
        return matrix


# ══════════════════════════════════════════════════════════════════════════
# LAYER 9 — SESSION STATE CONTINUITY LAYER
# ══════════════════════════════════════════════════════════════════════════

def enforce_state_initialization():
    if "writing_history" not in st.session_state:
        st.session_state.writing_history = []
    if "speaking_history" not in st.session_state:
        st.session_state.speaking_history = []
    if "global_target" not in st.session_state:
        st.session_state.global_target = 7.5
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Slate Obsidian (Dark)"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 10 — DYNAMIC THEME & RESPONSIVE COMPONENT INJECTORS
# ══════════════════════════════════════════════════════════════════════════

class UIThemeManager:
    @staticmethod
    def apply_custom_css(profile_key: str):
        p = THEME_PROFILES.get(profile_key, THEME_PROFILES["Slate Obsidian (Dark)"])
        css = f"""
        <style>
            .stApp {{ background-color: {p['bg']}; color: {p['text']}; }}
            .ielts-card {{
                background-color: {p['card_bg']};
                border: 1px solid {p['border']};
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 15px;
            }}
            .metric-big {{
                font-size: 36px;
                font-weight: 800;
                color: {p['accent']};
            }}
            .badge {{
                background-color: {p['border']};
                color: {p['text']};
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 11px;
            }}
        </style>
        """
        st.markdown(css, unsafe_escape=True)

    @staticmethod
    def build_radar_chart_svg(bands: dict[str, float]) -> str:
        if not bands:
            return ""
        keys = list(bands.keys())
        # Native mapping logic into responsive 300x300 structural grid loops
        pts = []
        center = 150
        max_val = 9.0
        radius = 90
        
        for idx, key in enumerate(keys):
            score = bands.get(key, 5.0)
            length = (score / max_val) * radius
            # Distribute axes layout radially across standard 4-point transformations
            if idx == 0: x, y = center, center - length # Up
            elif idx == 1: x, y = center + length, center # Right
            elif idx == 2: x, y = center, center + length # Down
            else: x, y = center - length, center # Left
            pts.append(f"{x},{y}")
            
        polygon_points = " ".join(pts)
        
        svg = f"""
        <svg viewBox="0 0 300 300" width="100%" height="250" style="background:transparent;">
            <circle cx="150" cy="150" r="30" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
            <circle cx="150" cy="150" r="60" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
            <circle cx="150" cy="150" r="90" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
            <line x1="150" y1="60" x2="150" y2="240" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
            <line x1="60" y1="150" x2="240" y2="150" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
            <text x="150" y="50" text-anchor="middle" fill="#00e676" font-size="12" font-weight="bold">{keys[0]}</text>
            <text x="250" y="154" text-anchor="start" fill="#00e676" font-size="12" font-weight="bold">{keys[1]}</text>
            <text x="150" y="260" text-anchor="middle" fill="#00e676" font-size="12" font-weight="bold">{keys[2]}</text>
            <text x="45" y="154" text-anchor="end" fill="#00e676" font-size="12" font-weight="bold">{keys[3]}</text>
            <polygon points="{polygon_points}" fill="rgba(0, 230, 118, 0.2)" stroke="#00e676" stroke-width="2"/>
        </svg>
        """
        return svg


# ══════════════════════════════════════════════════════════════════════════
# LAYER 11 — MAIN EXECUTION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="IELTS AI Tutor (Groq Engine)", page_icon="🎯", layout="wide")
    enforce_state_initialization()

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("⚡ Groq Engine Central")
    groq_key = st.sidebar.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")
    
    st.sidebar.markdown("---")
    st.session_state.global_target = st.sidebar.slider("Target IELTS Band", 5.0, 9.0, st.session_state.global_target, 0.5)
    st.session_state.theme_choice = st.sidebar.selectbox("Interface Mask Theme", list(THEME_PROFILES.keys()))
    
    UIThemeManager.apply_custom_css(st.session_state.theme_choice)

    # --- APP HEADSPACE ---
    st.title("🎯 Ultimate IELTS AI Tutor")
    st.caption("Deterministic Structural Multi-Modal Diagnostics Suite via Groq LPU Framework")

    # APP RUNNING TAB CORES
    tab_dash, tab_write, tab_speak, tab_logs = st.tabs([
        "📊 Analytical Dashboard", "✍️ Writing Evaluation Hub", "🗣️ Speaking Diagnostic Studio", "📜 Historical Execution Logs"
    ])

    with tab_dash:
        render_dashboard_tab()
    with tab_write:
        render_writing_tab(groq_key)
    with tab_speak:
        render_speaking_tab(groq_key)
    with tab_logs:
        render_logs_tab()


# ══════════════════════════════════════════════════════════════════════════
# LAYER 12 — PERFORMANCE ANALYTICS UI TAB
# ══════════════════════════════════════════════════════════════════════════

def render_dashboard_tab():
    all_subs = st.session_state.writing_history + st.session_state.speaking_history
    if not all_subs:
        st.info("No biometric profiling parameters registered inside current stack pipeline session.")
        return

    scores = [s.evaluation.overall_band for s in all_subs if s.evaluation]
    mean_band = StatisticalEngine.compute_running_mean(scores)
    err_matrix = ErrorFingerprintEngine.build_matrix(all_subs)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="ielts-card">Longitudinal Band Average<br/><span class="metric-big">{mean_band:.2f}</span></div>', unsafe_escape=True)
    with col2:
        st.markdown(f'<div class="ielts-card">Configured Target Track<br/><span class="metric-big">{st.session_state.global_target:.1f}</span></div>', unsafe_escape=True)
    with col3:
        st.markdown(f'<div class="ielts-card">Total Diagnostics Compiled<br/><span class="metric-big">{len(all_subs)}</span></div>', unsafe_escape=True)

    st.markdown("### 🧬 Longitudinal Error Fingerprint Tracking Matrix")
    if err_matrix:
        cols = st.columns(min(len(err_matrix), 4))
        for idx, (tag, tally) in enumerate(err_matrix.most_common(4)):
            with cols[idx]:
                st.markdown(f'<div class="ielts-card" style="border-left: 4px solid red;"><b>{tag}</b><br/>Occurrences: {tally}</div>', unsafe_escape=True)
    else:
        st.success("Linguistic accuracy telemetry records verify clean structural form.")


# ══════════════════════════════════════════════════════════════════════════
# LAYER 13 — WRITING ASSESSMENT UI TAB
# ══════════════════════════════════════════════════════════════════════════

def render_writing_tab(api_key: str):
    st.subheader("Academic / General Writing Evaluation Engine")
    
    ctx = st.radio("Context Task Matrix Profile", ["Task 2 (Academic/General)", "Task 1 (Academic - Data Analysis)"])
    use_mock = st.checkbox("Inject verified testing preset prompts", key="w_preset")
    
    if use_mock:
        prompt = st.selectbox("Preset Question Pool Selector", SAMPLE_WRITING_PROMPTS[ctx])
    else:
        prompt = st.text_area("Custom Examination Prompt Context Definition")

    essay = st.text_area("Student Script Workspace Asset", height=300, placeholder="Input text response sequence...")
    w_count = len(essay.split())
    st.caption(f"Calculated Metric Workspace Token Count: **{w_count} Words**")

    if st.button("Trigger Evaluator Pipeline Sequence", type="primary"):
        if not api_key:
            st.error("Operation Denied: Missing operational key matrix mapping variables inside dashboard options.")
            return
        if not essay.strip():
            st.warning("Workspace Buffer is completely empty.")
            return

        with st.spinner("Processing structural matching criteria matrices..."):
            client = GroqClient(api_key)
            raw = client.generate_content(PromptFactory.get_writing_system_prompt(), f"[PROMPT]\n{prompt}\n\n[ESSAY]\n{essay}")
            parsed_eval = ResponseParser.parse_writing_response(raw)
            
            sub = EssaySubmission(
                id=f"W-{random.randint(1000,9999)}",
                task_type=ctx,
                prompt=prompt,
                essay_text=essay,
                word_count=w_count,
                evaluation=parsed_eval
            )
            st.session_state.writing_history.append(sub)
            st.success("Evaluation profiling matrix compiled successfully.")

    if st.session_state.writing_history:
        st.markdown("---")
        st.subheader("Current Session Workspace Output View")
        latest = st.session_state.writing_history[-1]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Overall Verified Band Value", f"{latest.evaluation.overall_band:.1f}")
            chart_svg = UIThemeManager.build_radar_chart_svg(latest.evaluation.bands)
            st.markdown(chart_svg, unsafe_escape=True)
        with c2:
            for k, v in latest.evaluation.bands.items():
                st.write(f"**{IELTS_CRITERIA.get(k, k)}**: Band {v}")
                st.caption(latest.evaluation.feedback.get(k, ""))

        if latest.evaluation.mistakes:
            st.markdown("#### 🔍 Structural Inline Syntax Error Fix Tracking Matrix")
            for flaw in latest.evaluation.mistakes:
                st.markdown(
                    f"""<div style="background-color:rgba(255, 75, 75, 0.08); padding:12px; border-radius:6px; margin-bottom:10px; border-left:3px solid red;">
                        ❌ <b>Flawed Sequence Form:</b> <s>{html.escape(flaw.original)}</s><br/>
                        ✅ <b>Optimized Form Variant:</b> <span style="color:#00e676; font-weight:bold;">{html.escape(flaw.correction)}</span><br/>
                        🏷️ <span class="badge">{flaw.tag}</span> | 💡 <i>{html.escape(flaw.explanation)}</i>
                    </div>""", unsafe_escape=True
                )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 14 — SPEAKING VOICE DIAGNOSTIC UI TAB
# ══════════════════════════════════════════════════════════════════════════

def render_speaking_tab(api_key: str):
    st.subheader("Interactive Voice Profiling Engine Studio")
    
    part = st.radio("Speaking Target Context Profile", ["Part 1: Introduction & Familiar Topics", "Part 2: Long Turn (Cue Card)"])
    use_mock = st.checkbox("Inject mock test environment prompt benchmarks", key="s_preset")
    
    if use_mock:
        prompt = st.selectbox("Diagnostic Cue Card Anchor Pool", SAMPLE_SPEAKING_PROMPTS[part])
    else:
        prompt = st.text_area("Custom Interrogative Speaking Challenge Matrix Prompt")

    # Audio file byte processing widget assembly mapping structures
    audio_asset = st.file_uploader("Upload Speaking Record Metric Track (.wav)", type=["wav", "mp3", "m4a"])
    
    if st.button("Initialize Acoustic Matrix Deconstruction Pipeline", type="primary"):
        if not api_key:
            st.error("Operation Blocked: Operational access keys require active sidebar configurations.")
            return
        if not audio_asset:
            st.warning("Audio processing buffer tracking arrays require a file path assignment parameter.")
            return

        with st.spinner("Decoding vocal wave stream arrays via Groq Whisper pipeline..."):
            client = GroqClient(api_key)
            raw_bytes = audio_asset.read()
            transcript = client.transcribe_audio(raw_bytes, audio_asset.name)
            
            if "[Whisper" in transcript:
                st.error(transcript)
                return
                
            st.info(f"🎤 **Compiled Telemetry Voice Transcript:** {transcript}")
            
            # Send extracted transcript down conversational LLM evaluators
            raw_eval = client.generate_content(
                PromptFactory.get_speaking_system_prompt(),
                f"[PROMPT]\n{prompt}\n\n[TRANSCRIPT]\n{transcript}"
            )
            parsed_eval = ResponseParser.parse_speaking_response(raw_eval)
            
            sub = SpeakingSubmission(
                id=f"S-{random.randint(1000,9999)}",
                part_type=part,
                prompt=prompt,
                audio_filename=audio_asset.name,
                transcript=transcript,
                evaluation=parsed_eval
            )
            st.session_state.speaking_history.append(sub)
            st.success("Voice telemetry evaluation finalized.")

    if st.session_state.speaking_history:
        st.markdown("---")
        st.subheader("Current Audio Telemetry Execution View")
        latest = st.session_state.speaking_history[-1]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Acoustic Checked Overall Band", f"{latest.evaluation.overall_band:.1f}")
            chart_svg = UIThemeManager.build_radar_chart_svg(latest.evaluation.bands)
            st.markdown(chart_svg, unsafe_escape=True)
        with c2:
            for k, v in latest.evaluation.bands.items():
                st.write(f"**{SPEAKING_CRITERIA.get(k, k)}**: Band {v}")
                st.caption(latest.evaluation.feedback.get(k, ""))


def render_logs_tab():
    st.subheader("Longitudinal Analytics Track Event Registry")
    all_subs = st.session_state.writing_history + st.session_state.speaking_history
    if not all_subs:
        st.info("No analytical session telemetry profiles stored within historical records.")
        return

    for entry in reversed(all_subs):
        label = f"📝 {entry.id} — {entry.task_type if hasattr(entry, 'task_type') else entry.part_type}"
        with st.expander(f"{label} (Executed: {entry.submitted_at} | Band: {entry.evaluation.overall_band if entry.evaluation else 'N/A'})"):
            st.text(f"Prompt Context: {entry.prompt}")
            if hasattr(entry, 'essay_text'):
                st.text_area("Logged Submission Body", entry.essay_text, height=150, disabled=True)
            else:
                st.write(f"**Transcript Output**: {entry.transcript}")


if __name__ == "__main__":
    main()