"""
IELTS AI Tutor — Production Platform
======================================

A single-file Streamlit application architected with clear separation of
concerns: configuration, API integration, domain logic, and presentation.
Although delivered as one file (for fast, dependency-free deployment on
Streamlit Community Cloud), each layer below is self-contained and could be
lifted into its own module without modification once the project grows
beyond a single file — see the Architectural Overview at the bottom of
this file for the migration path into Phase B and Phase C.

Author note: This is the founder's actual product wedge — real AI feedback
graded against the official IELTS band descriptors, plus a persistent
"Error Fingerprint" that tracks each student's recurring mistakes across
every submission, not just per-essay. That longitudinal signal is the
defensible, hard-to-clone feature referenced in the roadmap's "AI
Personalisation Engine" section, and it is implemented for real below,
not just described.
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
# LAYER 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
# Keys are read from Streamlit secrets (.streamlit/secrets.toml locally, or
# the "Secrets" panel in Streamlit Community Cloud) rather than hardcoded
# in this file. secrets.toml is never committed to GitHub, so your real
# keys never appear in the codebase at all. See the setup instructions in
# the comment block below for exactly what to put where.
#
# The whole app now runs on a single free Groq API key — one key powers
# both the text-grading model (writing/speaking feedback) and the Whisper
# transcription model (voice recording). No other provider is needed.
#
# LOCAL SETUP:
#   1. Create a folder named ".streamlit" next to this app.py file.
#   2. Inside it, create a file named "secrets.toml" with this content:
#
#        GROQ_API_KEY = "your-real-groq-key-here"
#
#   3. Never commit .streamlit/secrets.toml to GitHub — add it to .gitignore.
#
# STREAMLIT COMMUNITY CLOUD SETUP:
#   In your app's dashboard, go to Settings -> Secrets, and paste the same
#   line shown above into the box provided. No file needed there.
#
# GET A FREE KEY:
#   Sign in at https://console.groq.com -> API Keys -> Create API Key.
#   Groq's free tier requires no credit card.

def _get_secret(key: str, fallback: str) -> str:
    """Reads a key from st.secrets if present, else returns a placeholder."""
    try:
        return st.secrets[key]
    except Exception:
        return fallback


GROQ_API_KEY: str = _get_secret("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

GROQ_HOST: str = "api.groq.com"
GROQ_CHAT_PATH: str = "/openai/v1/chat/completions"
GROQ_CHAT_MODEL: str = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"
GROQ_TIMEOUT_SECONDS: int = 60

APP_NAME: str = "IELTS AI Tutor"
APP_TAGLINE: str = "Your personal examiner. Available 24/7. In your language. At no cost."

SUPPORTED_LANGUAGES: tuple[str, ...] = ("English", "Hindi", "Bangla", "Chinese")
TASK_TYPES: tuple[str, ...] = ("Task 1 Academic", "Task 1 General", "Task 2 Essay")

# IELTS band thresholds used for visual severity coding throughout the UI.
BAND_EXCELLENT = 7.5
BAND_GOOD = 6.5
BAND_DEVELOPING = 5.5

# Exam timer presets, in seconds, keyed by the task the student selects in
# the Interactive Exam Timer module. Sourced from the real IELTS time
# allowance per task.
TIMER_PRESETS_SECONDS: dict[str, int] = {
    "Task 1 (20 Minutes)": 20 * 60,
    "Task 2 (40 Minutes)": 40 * 60,
}
TIMER_WARNING_THRESHOLD_SECONDS: int = 5 * 60  # flash red under 5 minutes

# Word-count goals per task, used by the Dynamic Word Count Goal Bar.
WORD_GOALS: dict[str, int] = {
    "Task 1 (20 Minutes)": 150,
    "Task 2 (40 Minutes)": 250,
}


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — DOMAIN MODELS
# ══════════════════════════════════════════════════════════════════════════
# Plain dataclasses representing the core entities of the platform. These
# are framework-agnostic — they know nothing about Streamlit or Groq —
# which is what makes them portable to a real database layer in Phase B.

@dataclass
class WritingScore:
    """The four official IELTS Writing band criteria plus an overall band."""

    overall: float = 0.0
    task_achievement: float = 0.0
    coherence_cohesion: float = 0.0
    lexical_resource: float = 0.0
    grammatical_range: float = 0.0

    def as_radar_values(self) -> list[float]:
        """Returns scores in the fixed order used by the radar chart."""
        return [
            self.task_achievement,
            self.coherence_cohesion,
            self.lexical_resource,
            self.grammatical_range,
        ]


@dataclass
class MistakeTag:
    """A single categorised error extracted from examiner feedback."""

    category: str
    example: str


@dataclass
class WritingSubmission:
    """One graded essay submission, stored in the student's session history."""

    task_type: str
    target_band: float
    word_count: int
    score: WritingScore
    feedback: str
    upgrades_table_md: str
    mistakes: list[MistakeTag]
    timestamp: str

    def to_history_row(self) -> dict:
        return {
            "Date": self.timestamp,
            "Task": self.task_type,
            "Words": self.word_count,
            "Overall Band": self.score.overall,
        }


@dataclass
class SpeakingSubmission:
    """One graded speaking response, stored in the student's session history."""

    part: str
    prompt: str
    response_text: str
    overall_band: float
    fluency: float
    vocabulary: float
    grammar: float
    pronunciation_note: str
    feedback: str
    timestamp: str


@dataclass
class StudentProfile:
    """
    The student's evolving skill profile. This is intentionally kept as a
    plain in-memory dataclass for the MVP; in Phase B this maps 1:1 onto a
    `students` table in PostgreSQL, and `mistake_counter` maps onto a
    `mistake_events` table partitioned by native_language for the
    cross-cohort analytics described in the roadmap's Phase C section.
    """

    name: str = "Guest Student"
    native_language: str = "English"
    target_band: float = 7.0
    streak_days: int = 1
    last_active_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    writing_history: list[WritingSubmission] = field(default_factory=list)
    speaking_history: list[SpeakingSubmission] = field(default_factory=list)
    mistake_counter: Counter = field(default_factory=Counter)

    @property
    def latest_writing_score(self) -> WritingScore:
        if self.writing_history:
            return self.writing_history[-1].score
        return WritingScore()

    @property
    def total_sessions(self) -> int:
        return len(self.writing_history) + len(self.speaking_history)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════

class GroqAPIError(Exception):
    """Raised for any failure in a Groq API request/response cycle (text or audio)."""


class GroqParsingError(Exception):
    """Raised when a Groq chat response cannot be parsed into the expected shape."""


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 — AI GATEWAY (API HANDLING)
# ══════════════════════════════════════════════════════════════════════════
# This is the ONLY part of the codebase that knows the Groq chat-completions
# API exists. Every other layer talks to `GroqClient`, never to http.client
# directly. This isolation is what lets us swap models, add retries, or add
# a caching layer without touching a single line of UI code.

class GroqClient:
    """
    Thin, dependency-free client for Groq's OpenAI-compatible
    `/openai/v1/chat/completions` endpoint.

    Uses the standard library exclusively (`http.client` + `json`) so that
    Streamlit Community Cloud deployments install instantly with zero
    extra wheels — a deliberate Phase A constraint that keeps the build
    pipeline trivial to debug. Groq's free tier needs no credit card and
    is the sole AI provider for this app — both grading (this client) and
    voice transcription (GroqWhisperClient below) run on the same key.
    """

    def __init__(self, api_key: str, model: str = GROQ_CHAT_MODEL, host: str = GROQ_HOST) -> None:
        self.api_key = api_key
        self.model = model
        self.host = host

    @property
    def is_configured(self) -> bool:
        """True once a real key has replaced the placeholder."""
        return bool(self.api_key) and self.api_key != "YOUR_GROQ_API_KEY_HERE"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
        """
        Sends a system + user prompt pair to Groq and returns the raw
        text of the model's reply.

        Raises:
            GroqAPIError: on missing key, network failure, timeout, or
                a non-200 response from the API.
        """
        if not self.is_configured:
            raise GroqAPIError(
                "No Groq API key configured. Set GROQ_API_KEY in .streamlit/secrets.toml."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": 4096,
        }

        connection: Optional[http.client.HTTPSConnection] = None
        try:
            connection = http.client.HTTPSConnection(self.host, timeout=GROQ_TIMEOUT_SECONDS)
            connection.request(
                "POST",
                GROQ_CHAT_PATH,
                body=json.dumps(payload),
                headers=headers,
            )
            response = connection.getresponse()
            raw_body = response.read().decode("utf-8")
            status = response.status
        except socket.timeout as exc:
            raise GroqAPIError("The examiner took too long to respond. Please try again.") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise GroqAPIError(f"Could not reach the Groq API: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

        if status != 200:
            hint = ""
            if status == 429:
                hint = " (Free-tier rate limit hit — wait a moment and try again.)"
            elif status in (401, 403):
                hint = " (Check that GROQ_API_KEY is correct and not expired.)"
            raise GroqAPIError(f"Groq API returned HTTP {status}: {raw_body[:300]}{hint}")

        try:
            parsed = json.loads(raw_body)
            return parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise GroqParsingError(f"Unexpected response shape from Groq: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4B — VOICE GATEWAY (Groq Whisper transcription)
# ══════════════════════════════════════════════════════════════════════════
# Mirrors GroqClient's isolation principle: this is the ONLY place that
# hand-builds the Whisper multipart/form-data request with http.client
# (no requests/groq SDK), keeping the zero-dependency Streamlit Community
# Cloud deploy story intact end-to-end. Used by the Speaking Simulator tab
# to turn a recorded mic clip into a transcript, which then flows into the
# exact same SPEAKING_EXAMINER_PROMPT pipeline used for typed responses.
# Same GROQ_API_KEY as GroqClient above — one key, one provider, two endpoints.


class GroqWhisperClient:
    """Thin, dependency-free client for Groq's Whisper transcription endpoint."""

    HOST = GROQ_HOST
    PATH = "/openai/v1/audio/transcriptions"
    MODEL = GROQ_WHISPER_MODEL

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "YOUR_GROQ_API_KEY_HERE"

    def transcribe(self, audio_bytes: bytes, filename: str = "speech.wav") -> str:
        """
        Sends a recorded audio clip to Groq Whisper and returns the
        transcribed text.

        Raises:
            GroqAPIError: on missing key, network failure, or a non-200
                response from the API.
        """
        if not self.is_configured:
            raise GroqAPIError(
                "No Groq API key configured. Add GROQ_API_KEY at the top of app.py."
            )

        boundary = "----IELTSAITutorBoundary7f3a9c"
        body_parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            body_parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
            )

        add_field("model", self.MODEL)
        add_field("response_format", "text")
        add_field("language", "en")

        body_parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8")
        )
        body_parts.append(audio_bytes)
        body_parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(body_parts)

        connection: Optional[http.client.HTTPSConnection] = None
        try:
            connection = http.client.HTTPSConnection(self.HOST, timeout=GROQ_TIMEOUT_SECONDS)
            connection.request(
                "POST",
                self.PATH,
                body=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            response = connection.getresponse()
            raw_body = response.read().decode("utf-8")
            status = response.status
        except socket.timeout as exc:
            raise GroqAPIError("Transcription took too long. Please try a shorter recording.") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise GroqAPIError(f"Could not reach the Groq API: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

        if status != 200:
            raise GroqAPIError(f"Groq API returned HTTP {status}: {raw_body[:300]}")

        # response_format=text returns plain text directly, not JSON
        return raw_body.strip()


# ══════════════════════════════════════════════════════════════════════════
# LAYER 5 — PROMPT ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
# Prompts are versioned, named constants — never inlined in UI code — so
# that prompt iteration (the highest-leverage activity per the roadmap's
# "AI Integration Specialist" role) happens in exactly one place.

WRITING_EXAMINER_PROMPT = """You are a certified, official IELTS Writing examiner. You mark strictly \
against the real IELTS band descriptors and never inflate a score out of kindness. You are reviewing \
a submission from a non-native English speaker preparing for migration or study abroad, so accuracy \
matters more to this student than encouragement — give both, but never sacrifice the first for the second.

You will receive: the IELTS Task Type, the candidate's Target Band Score, and their essay text.

Mark the essay against these four official criteria: Task Achievement/Response, Coherence and \
Cohesion, Lexical Resource, and Grammatical Range and Accuracy.

Respond using ONLY this exact XML structure, each tag appearing exactly once, with no text before or \
after it and no markdown code fences wrapping the whole response:

<overall>X.X</overall>
<task_achievement>X.X</task_achievement>
<coherence_cohesion>X.X</coherence_cohesion>
<lexical_resource>X.X</lexical_resource>
<grammatical_range>X.X</grammatical_range>
<feedback>
Write specific, examiner-grade bullet points (each starting with "- ") covering all four criteria. \
Quote exact words or sentences from the essay. Be direct about weaknesses. State precisely what is \
needed to reach the candidate's target band.
</feedback>
<upgrades>
A Markdown table with this exact header, containing at least 5 real examples from the candidate's \
own essay:

| Weak Word Used | Band 9.0 Alternative | Context Sentence |
|---|---|---|
| word | alternative | example sentence using the alternative |
</upgrades>
<mistake_tags>
A comma-separated list of error categories actually found in this essay, chosen ONLY from this fixed \
vocabulary: article_error, tense_error, run_on_sentence, subject_verb_agreement, preposition_error, \
weak_collocation, repetition, comma_splice, informal_register, missing_linking_word, sentence_fragment, \
plural_error, word_order. List only categories genuinely present, no duplicates.
</mistake_tags>

All band scores must be valid IELTS bands in 0.5 increments (5.0, 5.5, 6.0 ... 9.0)."""


SPEAKING_EXAMINER_PROMPT = """You are a certified, official IELTS Speaking examiner reviewing a \
transcript of a candidate's spoken response (this text was produced by speech-to-text transcription \
of the candidate's actual spoken answer). Mark strictly against the four official IELTS Speaking \
criteria: Fluency and Coherence, Lexical Resource, Grammatical Range and Accuracy, and Pronunciation. \
Since you cannot hear audio, infer pronunciation risk only from spelling patterns, filler-word density, \
and phonetic clues in the transcript, and say so honestly rather than guessing confidently.

You will receive: the Speaking Part (1, 2, or 3), the question/cue card, and the candidate's transcribed \
response.

Respond using ONLY this exact XML structure, each tag exactly once, no text outside the tags:

<overall>X.X</overall>
<fluency>X.X</fluency>
<vocabulary>X.X</vocabulary>
<grammar>X.X</grammar>
<pronunciation_note>One honest sentence on inferred pronunciation risk from the transcript, noting this is an estimate since no audio was analysed.</pronunciation_note>
<feedback>
Specific bullet points (each starting with "- ") on fluency, vocabulary range, grammar, and what to \
improve to reach a higher band. Quote exact phrases the candidate used.
</feedback>
<mistake_tags>
A comma-separated list of error categories actually found, chosen ONLY from this fixed vocabulary: \
article_error, tense_error, run_on_sentence, subject_verb_agreement, preposition_error, \
weak_collocation, repetition, filler_overuse, informal_register, missing_linking_word, \
limited_vocabulary_range, word_order. List only categories genuinely present.
</mistake_tags>

All band scores must be valid IELTS bands in 0.5 increments."""


REWRITE_TO_BAND9_PROMPT = """You are an expert IELTS Writing tutor. You will receive a candidate's original \
essay, the IELTS Task Type, and the essay topic/prompt. Your job is NOT to grade it — it is to rewrite \
it into a polished, natural-sounding Band 9.0 sample answer that keeps the candidate's own ideas, \
opinions, examples, and overall argument structure, but expresses them with the range of grammar, \
precise vocabulary, and cohesive flow expected at the top band.

Rules:
- Preserve the candidate's actual stance, examples, and the order of their main points — this is a \
polish, not a different essay.
- Fix every grammar, vocabulary, and cohesion issue silently as you rewrite.
- Keep the length comparable to the original (do not pad it dramatically longer).
- Respond with ONLY the rewritten essay text. No preamble, no headings, no markdown, no commentary, \
no "Here is the rewritten essay" framing — just the essay itself."""


MISTAKE_LABELS: dict[str, str] = {
    "article_error": "Articles (a / an / the)",
    "tense_error": "Verb tense consistency",
    "run_on_sentence": "Run-on sentences",
    "subject_verb_agreement": "Subject–verb agreement",
    "preposition_error": "Preposition misuse",
    "weak_collocation": "Weak word pairings",
    "repetition": "Repetitive vocabulary",
    "comma_splice": "Comma splices",
    "informal_register": "Informal tone",
    "missing_linking_word": "Missing linking words",
    "sentence_fragment": "Sentence fragments",
    "plural_error": "Plural/singular errors",
    "word_order": "Word order",
    "filler_overuse": "Filler word overuse",
    "limited_vocabulary_range": "Limited vocabulary range",
}


# ══════════════════════════════════════════════════════════════════════════
# LAYER 6 — RESPONSE PARSING (DATA PROCESSING)
# ══════════════════════════════════════════════════════════════════════════

def _extract_tag(tag: str, text: str) -> str:
    """Extracts the inner text of the first <tag>...</tag> match, or ''."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _safe_band(raw: str, fallback: float = 0.0) -> float:
    """Coerces a noisy model output into a valid float band score."""
    cleaned = re.sub(r"[^0-9.]", "", raw)
    try:
        value = float(cleaned) if cleaned else fallback
    except ValueError:
        return fallback
    return round(value * 2) / 2  # snap to nearest 0.5, matching IELTS bands


def parse_mistake_tags(raw_tag_list: str) -> list[MistakeTag]:
    """Converts the comma-separated <mistake_tags> output into typed tags."""
    tags: list[MistakeTag] = []
    for chunk in raw_tag_list.split(","):
        key = chunk.strip().lower().replace(" ", "_")
        if key and key in MISTAKE_LABELS:
            tags.append(MistakeTag(category=key, example=MISTAKE_LABELS[key]))
    return tags


def parse_writing_response(raw_text: str, task_type: str, target_band: float, word_count: int) -> WritingSubmission:
    """Parses a raw Groq writing-evaluation response into a WritingSubmission."""
    score = WritingScore(
        overall=_safe_band(_extract_tag("overall", raw_text)),
        task_achievement=_safe_band(_extract_tag("task_achievement", raw_text)),
        coherence_cohesion=_safe_band(_extract_tag("coherence_cohesion", raw_text)),
        lexical_resource=_safe_band(_extract_tag("lexical_resource", raw_text)),
        grammatical_range=_safe_band(_extract_tag("grammatical_range", raw_text)),
    )
    return WritingSubmission(
        task_type=task_type,
        target_band=target_band,
        word_count=word_count,
        score=score,
        feedback=_extract_tag("feedback", raw_text),
        upgrades_table_md=_extract_tag("upgrades", raw_text),
        mistakes=parse_mistake_tags(_extract_tag("mistake_tags", raw_text)),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def parse_speaking_response(raw_text: str, part: str, prompt_text: str, response_text: str) -> tuple[SpeakingSubmission, list[MistakeTag]]:
    """Parses a raw Groq speaking-evaluation response into a SpeakingSubmission."""
    submission = SpeakingSubmission(
        part=part,
        prompt=prompt_text,
        response_text=response_text,
        overall_band=_safe_band(_extract_tag("overall", raw_text)),
        fluency=_safe_band(_extract_tag("fluency", raw_text)),
        vocabulary=_safe_band(_extract_tag("vocabulary", raw_text)),
        grammar=_safe_band(_extract_tag("grammar", raw_text)),
        pronunciation_note=_extract_tag("pronunciation_note", raw_text),
        feedback=_extract_tag("feedback", raw_text),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    mistakes = parse_mistake_tags(_extract_tag("mistake_tags", raw_text))
    return submission, mistakes


# ══════════════════════════════════════════════════════════════════════════
# LAYER 7 — CONTENT BANK
# ══════════════════════════════════════════════════════════════════════════
# In Phase A this is a static Python list. In Phase B this becomes a
# `vocabulary` and `speaking_prompts` table in PostgreSQL, seeded from the
# founder's existing tuition-centre material (see roadmap §5.1/5.2).

@dataclass(frozen=True)
class VocabWord:
    word: str
    definition: str
    example: str
    band_level: str


WORD_OF_THE_DAY_BANK: tuple[VocabWord, ...] = (
    VocabWord("Substantial", "Of considerable size, worth, or importance", "There has been a substantial increase in remote work since 2020.", "Band 7+"),
    VocabWord("Mitigate", "To make something less severe or harmful", "Governments are introducing policies to mitigate the effects of climate change.", "Band 7+"),
    VocabWord("Pertinent", "Relevant or applicable to a particular matter", "She raised a pertinent point about funding constraints.", "Band 7+"),
    VocabWord("Discrepancy", "A lack of compatibility between facts or claims", "There is a notable discrepancy between the two surveys' findings.", "Band 7+"),
    VocabWord("Inevitable", "Certain to happen; unavoidable", "Some argue that urbanisation is an inevitable consequence of economic growth.", "Band 7+"),
    VocabWord("Proliferation", "Rapid increase in number or amount", "The proliferation of smartphones has transformed communication.", "Band 8+"),
    VocabWord("Detrimental", "Tending to cause harm", "Excessive screen time can be detrimental to children's development.", "Band 7+"),
    VocabWord("Ambiguous", "Open to more than one interpretation; unclear", "The wording of the question was ambiguous.", "Band 7+"),
    VocabWord("Consensus", "A general agreement among a group", "There is a growing consensus among scientists on this issue.", "Band 7+"),
    VocabWord("Paramount", "More important than anything else; supreme", "Safety is of paramount importance in this industry.", "Band 8+"),
)


@dataclass(frozen=True)
class WritingPrompt:
    """A single hardcoded IELTS Task 2-style essay prompt for the Prompt Generator Engine."""

    category: str
    prompt: str


# Hardcoded prompt bank for the "Surprise Me" generator — at least 15 real,
# exam-realistic Task 2 prompts spread across the categories IELTS actually
# draws from. No API call needed to populate this; it's instant and free.
WRITING_PROMPT_BANK: tuple[WritingPrompt, ...] = (
    WritingPrompt("Global Environment", "Some people believe that environmental problems are too big for individual countries to solve, and that they can only be solved by international cooperation. To what extent do you agree or disagree?"),
    WritingPrompt("Global Environment", "Many governments think that economic development is the most important factor for a country, while others think protecting the environment is more important. Discuss both views and give your own opinion."),
    WritingPrompt("Global Environment", "Plastic bags and packaging contribute significantly to environmental pollution. What problems does this cause and what measures could be taken to address it?"),
    WritingPrompt("Automation & AI", "Artificial intelligence is increasingly being used to make decisions that were previously made by humans, such as in hiring and medical diagnosis. Do the advantages of this outweigh the disadvantages?"),
    WritingPrompt("Automation & AI", "In the future, robots and automated systems will replace most jobs currently done by humans. Do you agree or disagree with this statement?"),
    WritingPrompt("Automation & AI", "Some people think that the increasing use of automation in the workplace will lead to widespread unemployment, while others believe it will create new types of jobs. Discuss both views and give your opinion."),
    WritingPrompt("Public Education", "Some people believe that university education should be free for all students, while others think students should pay for their own tuition. Discuss both views and give your opinion."),
    WritingPrompt("Public Education", "In many countries, children are required to study a foreign language from a very young age. Do the advantages of this outweigh the disadvantages?"),
    WritingPrompt("Public Education", "Some educators argue that examinations are not an effective way to assess a student's abilities, and that alternative methods should be used. To what extent do you agree or disagree?"),
    WritingPrompt("Urban Infrastructure", "As cities grow larger, many people are choosing to live in high-rise apartment buildings rather than houses with gardens. What are the advantages and disadvantages of this trend?"),
    WritingPrompt("Urban Infrastructure", "Traffic congestion is becoming an increasingly serious problem in many major cities around the world. What are the causes of this problem and what solutions can you suggest?"),
    WritingPrompt("Urban Infrastructure", "Some city planners argue that public transport should be free for all residents in order to reduce traffic and pollution. To what extent do you agree or disagree?"),
    WritingPrompt("Health & Lifestyle", "In many countries, obesity rates are rising rapidly among both children and adults. What are the causes of this trend and what measures could be taken to address it?"),
    WritingPrompt("Health & Lifestyle", "Some people believe that the best way to improve public health is to invest more in sports facilities, while others believe investing in healthcare and medicine is more effective. Discuss both views and give your own opinion."),
    WritingPrompt("Technology & Society", "The widespread use of smartphones and social media has changed the way people communicate with one another. Do the advantages of this development outweigh the disadvantages?"),
    WritingPrompt("Technology & Society", "Some people think that the internet has made it easier for people to access accurate information, while others believe it has increased the spread of misinformation. Discuss both views and give your own opinion."),
    WritingPrompt("Crime & Society", "Some people believe that the most effective way to reduce crime is to impose tougher prison sentences, while others believe addressing the root causes of crime is more effective. Discuss both views and give your opinion."),
    WritingPrompt("Work & Career", "Many people believe that a good salary is more important than job satisfaction when choosing a career. To what extent do you agree or disagree?"),
    WritingPrompt("Work & Career", "In some countries, employees are required to retire at a fixed age, while in others there is no fixed retirement age. Discuss both approaches and give your own opinion."),
)


@dataclass(frozen=True)
class SpeakingPrompt:
    part: str
    prompt: str
    topic: str


SPEAKING_PROMPT_BANK: tuple[SpeakingPrompt, ...] = (
    SpeakingPrompt("Part 1", "Can you describe the area where you live?", "Hometown"),
    SpeakingPrompt("Part 1", "Do you prefer studying alone or with other people? Why?", "Study habits"),
    SpeakingPrompt("Part 1", "What kind of music do you enjoy listening to?", "Music"),
    SpeakingPrompt("Part 2", "Describe a skill you would like to learn. You should say: what it is, why you want to learn it, how you would learn it, and explain how it would benefit you.", "Cue Card — Skill"),
    SpeakingPrompt("Part 2", "Describe a memorable journey you have taken. You should say: where you went, who you went with, what you did, and explain why it was memorable.", "Cue Card — Journey"),
    SpeakingPrompt("Part 3", "Do you think technology has made people more isolated or more connected?", "Technology & society"),
    SpeakingPrompt("Part 3", "How important is it for governments to invest in public transport?", "Urban planning"),
)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 8 — THE ERROR FINGERPRINT ENGINE  (signature feature)
# ══════════════════════════════════════════════════════════════════════════
# This is the platform's core defensible IP. Most "AI feedback" tools grade
# each submission in isolation and the student forgets the feedback within
# a day. This engine accumulates every mistake category across EVERY
# writing and speaking submission in the session into a persistent
# "fingerprint" — so the student sees, in one glance, the handful of
# specific errors quietly capping their score across dozens of attempts,
# and watches that fingerprint visibly shrink as they improve. This is the
# concrete implementation of the roadmap's "AI Personalisation Engine"
# (§2.2) and is the single feature most worth defending as IP in Phase C's
# "Advanced analytics... feeds back into better AI training" line, since it
# is this exact event stream that would train a proprietary error-detection
# model down the line.

@dataclass
class FingerprintEntry:
    category: str
    label: str
    count: int
    first_seen: str
    trend: str  # "improving" | "steady" | "worsening" | "new"


class ErrorFingerprintEngine:
    """
    Maintains and analyses the student's cumulative mistake profile.

    The engine is stateless with respect to storage — it operates purely
    on the `StudentProfile.mistake_counter` and submission history handed
    to it, which makes it trivial to back with a real database query in
    Phase B without changing any of this logic.
    """

    @staticmethod
    def record(profile: StudentProfile, mistakes: list[MistakeTag]) -> None:
        """Folds a new submission's mistakes into the persistent counter."""
        for tag in mistakes:
            profile.mistake_counter[tag.category] += 1

    @staticmethod
    def top_recurring(profile: StudentProfile, limit: int = 5) -> list[FingerprintEntry]:
        """Returns the most frequent recurring error categories, ranked."""
        entries: list[FingerprintEntry] = []
        for category, count in profile.mistake_counter.most_common(limit):
            trend = ErrorFingerprintEngine._trend_for(profile, category)
            entries.append(
                FingerprintEntry(
                    category=category,
                    label=MISTAKE_LABELS.get(category, category.replace("_", " ").title()),
                    count=count,
                    first_seen=ErrorFingerprintEngine._first_seen(profile, category),
                    trend=trend,
                )
            )
        return entries

    @staticmethod
    def _trend_for(profile: StudentProfile, category: str) -> str:
        """
        Compares frequency of `category` in the most recent half of the
        student's submissions against the earlier half, to flag whether a
        specific mistake is fading out or getting worse.
        """
        all_subs = profile.writing_history + profile.speaking_history  # type: ignore[operator]
        if len(all_subs) < 2:
            return "new"

        midpoint = len(all_subs) // 2
        earlier, recent = all_subs[:midpoint], all_subs[midpoint:]

        def count_in(subset) -> int:
            total = 0
            for sub in subset:
                tags = getattr(sub, "mistakes", None)
                if tags:
                    total += sum(1 for t in tags if t.category == category)
            return total

        earlier_count, recent_count = count_in(earlier), count_in(recent)
        if earlier_count == 0 and recent_count == 0:
            return "new"
        if recent_count < earlier_count:
            return "improving"
        if recent_count > earlier_count:
            return "worsening"
        return "steady"

    @staticmethod
    def _first_seen(profile: StudentProfile, category: str) -> str:
        for sub in profile.writing_history:
            if any(t.category == category for t in sub.mistakes):
                return sub.timestamp
        return "this session"

    @staticmethod
    def focus_recommendation(profile: StudentProfile) -> Optional[str]:
        """Returns one human-readable coaching line for the single biggest issue."""
        top = ErrorFingerprintEngine.top_recurring(profile, limit=1)
        if not top:
            return None
        entry = top[0]
        return (
            f"Your #1 recurring issue is <b>{html.escape(entry.label)}</b> — it has appeared "
            f"{entry.count} time{'s' if entry.count != 1 else ''} across your submissions. "
            f"Fixing this one pattern will likely move your score faster than anything else."
        )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 8B — BAND SCORE CALCULATOR (PURE-CODE, NO API CALL)
# ══════════════════════════════════════════════════════════════════════════
# Maps raw Listening/Reading correct-answer counts onto official IELTS band
# scores using the published Academic conversion tables. This is deliberately
# zero-AI: it's arithmetic against a lookup table, so it's instant, free,
# and works even with no Groq key configured at all — useful as a standalone
# diagnostic students can use before they've written a single word.

def listening_band(correct: int) -> float:
    """Converts a raw Listening score (0-40) into an IELTS band, per the
    official Academic/General Listening conversion table."""
    correct = max(0, min(40, correct))
    thresholds = [
        (39, 9.0), (37, 8.5), (35, 8.0), (32, 7.5), (30, 7.0),
        (26, 6.5), (23, 6.0), (18, 5.5), (16, 5.0), (13, 4.5),
        (10, 4.0), (8, 3.5), (6, 3.0), (4, 2.5),
    ]
    for cutoff, band in thresholds:
        if correct >= cutoff:
            return band
    return 2.0


def reading_band_academic(correct: int) -> float:
    """Converts a raw Reading score (0-40) into an IELTS Academic Reading
    band, per the official conversion table (Academic Reading is marked
    slightly more strictly than Listening at the high end)."""
    correct = max(0, min(40, correct))
    thresholds = [
        (39, 9.0), (37, 8.5), (35, 8.0), (33, 7.5), (30, 7.0),
        (27, 6.5), (23, 6.0), (19, 5.5), (15, 5.0), (13, 4.5),
        (10, 4.0), (8, 3.5), (6, 3.0), (4, 2.5),
    ]
    for cutoff, band in thresholds:
        if correct >= cutoff:
            return band
    return 2.0


def overall_band_from_four(listening: float, reading: float, writing: float, speaking: float) -> float:
    """
    Averages the four component bands and rounds to the nearest 0.5 using
    IELTS's official rounding rule: a mean ending in .25 rounds UP to the
    next .5, and a mean ending in .75 rounds UP to the next whole band.
    """
    mean = (listening + reading + writing + speaking) / 4
    remainder = mean - int(mean)
    if abs(remainder - 0.25) < 1e-9:
        return int(mean) + 0.5
    if abs(remainder - 0.75) < 1e-9:
        return int(mean) + 1.0
    return round(mean * 2) / 2


# ══════════════════════════════════════════════════════════════════════════
# LAYER 9 — SESSION STATE (APPLICATION STATE MANAGEMENT)
# ══════════════════════════════════════════════════════════════════════════
# Streamlit's session_state is our in-memory substitute for a real session
# store. Everything here maps onto a `students` row + related child tables
# the moment Phase B adds PostgreSQL — see the Architectural Overview.

def init_session_state() -> None:
    """Idempotently initialises all session-scoped state. Safe to call every rerun."""
    if "profile" not in st.session_state:
        st.session_state.profile = StudentProfile()
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = GroqClient(GROQ_API_KEY)
    if "groq_whisper_client" not in st.session_state:
        st.session_state.groq_whisper_client = GroqWhisperClient(GROQ_API_KEY)
    if "word_of_day_index" not in st.session_state:
        st.session_state.word_of_day_index = random.randint(0, len(WORD_OF_THE_DAY_BANK) - 1)
    if "active_speaking_prompt" not in st.session_state:
        st.session_state.active_speaking_prompt = random.choice(SPEAKING_PROMPT_BANK)
    if "last_writing_submission" not in st.session_state:
        st.session_state.last_writing_submission = None
    if "last_speaking_submission" not in st.session_state:
        st.session_state.last_speaking_submission = None

    # --- Exam Timer state ---
    if "timer_preset" not in st.session_state:
        st.session_state.timer_preset = "Task 2 (40 Minutes)"
    if "timer_remaining_seconds" not in st.session_state:
        st.session_state.timer_remaining_seconds = TIMER_PRESETS_SECONDS[st.session_state.timer_preset]
    if "timer_running" not in st.session_state:
        st.session_state.timer_running = False
    if "timer_last_tick" not in st.session_state:
        st.session_state.timer_last_tick = None

    # --- Prompt Generator state ---
    if "generated_prompt_text" not in st.session_state:
        st.session_state.generated_prompt_text = ""

    # --- Band 9 rewrite (side-by-side comparison) cache ---
    if "band9_rewrite" not in st.session_state:
        st.session_state.band9_rewrite = None

    # --- Session history log (full essay + feedback snapshots) ---
    if "essay_history_log" not in st.session_state:
        st.session_state.essay_history_log = []
    if "selected_history_index" not in st.session_state:
        st.session_state.selected_history_index = None


def register_daily_visit() -> None:
    """
    Updates the streak counter. If the student last visited yesterday, the
    streak increments; if they visited today already, it's a no-op; any
    larger gap resets the streak to 1. This is the entire Phase A
    "gamification" requirement from the roadmap, implemented honestly
    rather than as a static display number.
    """
    profile: StudentProfile = st.session_state.profile
    today = datetime.now().strftime("%Y-%m-%d")
    if profile.last_active_date == today:
        return
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    profile.streak_days = profile.streak_days + 1 if profile.last_active_date == yesterday else 1
    profile.last_active_date = today


def tick_exam_timer() -> None:
    """
    Advances the exam timer based on real elapsed wall-clock time since the
    last rerun, rather than a naive per-rerun decrement — this keeps the
    countdown accurate regardless of Streamlit's rerun frequency.
    """
    if not st.session_state.timer_running:
        return
    now = datetime.now()
    last_tick = st.session_state.timer_last_tick
    if last_tick is not None:
        elapsed = (now - last_tick).total_seconds()
        st.session_state.timer_remaining_seconds = max(0, st.session_state.timer_remaining_seconds - elapsed)
        if st.session_state.timer_remaining_seconds <= 0:
            st.session_state.timer_running = False
    st.session_state.timer_last_tick = now


def format_mm_ss(total_seconds: float) -> str:
    """Formats a float seconds count as MM:SS for the timer display."""
    total_seconds = max(0, int(total_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 10 — VISUAL BUILDERS (hand-built SVG, no charting dependency)
# ══════════════════════════════════════════════════════════════════════════
# Deliberately framework-free SVG generation. This keeps the certificate
# and radar chart pixel-perfect, on-brand, and dependency-free for instant
# Streamlit Community Cloud cold-starts — no matplotlib, no plotly.

def render_skill_radar_svg(scores: WritingScore, size: int = 320) -> str:
    """
    Builds a 4-axis radar chart (Task Achievement, Coherence, Lexical
    Resource, Grammar) as raw SVG, scaled to IELTS bands 0-9.
    """
    labels = ["Task Achievement", "Coherence & Cohesion", "Lexical Resource", "Grammatical Range"]
    values = scores.as_radar_values()
    center = size / 2
    radius = size * 0.34
    max_band = 9.0
    angle_step = 2 * 3.14159265 / 4

    def point(index: int, value: float) -> tuple[float, float]:
        angle = -3.14159265 / 2 + index * angle_step
        r = radius * (value / max_band)
        return center + r * _cos(angle), center + r * _sin(angle)

    def axis_point(index: int) -> tuple[float, float]:
        angle = -3.14159265 / 2 + index * angle_step
        return center + radius * _cos(angle), center + radius * _sin(angle)

    grid_rings = "".join(
        f'<circle cx="{center}" cy="{center}" r="{radius * frac:.1f}" '
        f'fill="none" stroke="#3A3528" stroke-width="0.5" opacity="0.5"/>'
        for frac in (0.25, 0.5, 0.75, 1.0)
    )
    axis_lines = "".join(
        f'<line x1="{center}" y1="{center}" x2="{axis_point(i)[0]:.1f}" y2="{axis_point(i)[1]:.1f}" '
        f'stroke="#3A3528" stroke-width="0.5" opacity="0.6"/>'
        for i in range(4)
    )
    polygon_points = " ".join(f"{point(i, v)[0]:.1f},{point(i, v)[1]:.1f}" for i, v in enumerate(values))

    label_offsets = [(0, -16), (18, 0), (0, 16), (-18, 0)]
    label_anchors = ["middle", "start", "middle", "end"]
    labels_svg = ""
    for i, label in enumerate(labels):
        lx, ly = axis_point(i)
        dx, dy = label_offsets[i]
        labels_svg += (
            f'<text x="{lx + dx:.1f}" y="{ly + dy:.1f}" text-anchor="{label_anchors[i]}" '
            f'font-family="Inter, sans-serif" font-size="11" font-weight="600" fill="#C9A227">'
            f"{html.escape(label)}</text>"
        )
        labels_svg += (
            f'<text x="{lx + dx:.1f}" y="{ly + dy + 13:.1f}" text-anchor="{label_anchors[i]}" '
            f'font-family="Inter, sans-serif" font-size="13" font-weight="700" fill="#F7F5F0">'
            f"{values[i]:.1f}</text>"
        )

    return f"""
    <svg viewBox="0 0 {size} {size}" width="100%" height="{size}" role="img"
         aria-label="Radar chart of writing band scores across four IELTS criteria">
        {grid_rings}
        {axis_lines}
        <polygon points="{polygon_points}" fill="#C9A227" fill-opacity="0.22"
                 stroke="#C9A227" stroke-width="2" stroke-linejoin="round"/>
        {"".join(f'<circle cx="{point(i, v)[0]:.1f}" cy="{point(i, v)[1]:.1f}" r="3.5" fill="#C9A227"/>' for i, v in enumerate(values))}
        {labels_svg}
    </svg>
    """


def _cos(angle: float) -> float:
    import math
    return math.cos(angle)


def _sin(angle: float) -> float:
    import math
    return math.sin(angle)


def render_band_dial_svg(band: float, label: str = "OVERALL BAND", size: int = 260) -> str:
    """
    Renders a half-circle dial gauge (0-9 IELTS scale) as raw SVG, with a
    needle pointing at the live band score. Used by the standalone Band
    Score Calculator tab for instant, no-API, pure-math visual feedback.
    """
    import math as _math

    cx, cy = size / 2, size * 0.58
    radius = size * 0.40
    max_band = 9.0
    band_clamped = max(0.0, min(max_band, band))

    start_angle = _math.pi  # 180°, left side
    end_angle = 0.0  # 0°, right side
    frac = band_clamped / max_band
    angle = start_angle - frac * (start_angle - end_angle)

    def arc_point(a: float, r: float) -> tuple[float, float]:
        return cx + r * _math.cos(a), cy - r * _math.sin(a)

    # Colour-graded ticks: brick (low) -> gold (mid) -> sage (high)
    tick_colors = ["#9B4A3F", "#9B4A3F", "#C9A227", "#C9A227", "#C9A227", "#5E8C6A", "#5E8C6A", "#5E8C6A", "#5E8C6A"]
    ticks_svg = ""
    for i in range(10):
        a = start_angle - (i / 9) * (start_angle - end_angle)
        x1, y1 = arc_point(a, radius + 6)
        x2, y2 = arc_point(a, radius - 6)
        color = tick_colors[min(i, len(tick_colors) - 1)]
        ticks_svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="3"/>'

    arc_segments = 48
    arc_path_parts = []
    for i in range(arc_segments + 1):
        a = start_angle - (i / arc_segments) * (start_angle - end_angle)
        x, y = arc_point(a, radius)
        arc_path_parts.append(f"{x:.1f},{y:.1f}")
    track_points = " ".join(arc_path_parts)

    needle_x, needle_y = arc_point(angle, radius - 14)

    return f"""
    <svg viewBox="0 0 {size} {size * 0.66:.0f}" width="100%" role="img"
         aria-label="Dial gauge showing the calculated IELTS band score">
        <polyline points="{track_points}" fill="none" stroke="#232B36" stroke-width="10" stroke-linecap="round"/>
        {ticks_svg}
        <line x1="{cx:.1f}" y1="{cy:.1f}" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
              stroke="#C9A227" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#C9A227"/>
        <text x="{cx:.1f}" y="{cy + 38:.1f}" text-anchor="middle" font-family="'Source Serif Pro', Georgia, serif"
              font-size="40" font-weight="700" fill="#F7F5F0">{band_clamped:.1f}</text>
        <text x="{cx:.1f}" y="{cy + 58:.1f}" text-anchor="middle" font-family="Inter, sans-serif"
              font-size="10" letter-spacing="1.5" fill="#8A8775">{html.escape(label)}</text>
    </svg>
    """


def render_band_certificate_svg(overall: float, profile: StudentProfile) -> str:
    """
    Renders a facsimile of the official IELTS Test Report Form layout —
    the actual certificate students recognise instantly — filled with the
    student's live AI-estimated band. This is the platform's signature
    visual element.
    """
    safe_name = html.escape(profile.name or "Guest Student")
    today_str = datetime.now().strftime("%d %b %Y")
    return f"""
    <svg viewBox="0 0 640 220" width="100%" role="img"
         aria-label="Facsimile IELTS Test Report Form showing the estimated overall band score">
        <rect x="1" y="1" width="638" height="218" rx="6" fill="#F7F5F0" stroke="#C9A227" stroke-width="1.5"/>
        <rect x="14" y="14" width="612" height="192" rx="3" fill="none" stroke="#C9A227" stroke-width="0.75" stroke-dasharray="2 3"/>
        <text x="32" y="42" font-family="'Source Serif Pro', Georgia, serif" font-size="15" font-weight="700" fill="#1A1A1A">
            TEST REPORT FORM (FACSIMILE)
        </text>
        <text x="32" y="60" font-family="Inter, sans-serif" font-size="10" fill="#5B5B5B">
            AI-estimated band — not an official IELTS score · Generated {html.escape(today_str)}
        </text>
        <line x1="32" y1="72" x2="608" y2="72" stroke="#1A1A1A" stroke-width="0.75"/>
        <text x="32" y="96" font-family="Inter, sans-serif" font-size="11" fill="#5B5B5B">CANDIDATE</text>
        <text x="32" y="114" font-family="Inter, sans-serif" font-size="14" font-weight="600" fill="#1A1A1A">{safe_name}</text>
        <text x="32" y="138" font-family="Inter, sans-serif" font-size="11" fill="#5B5B5B">TARGET BAND</text>
        <text x="32" y="156" font-family="Inter, sans-serif" font-size="14" font-weight="600" fill="#1A1A1A">{profile.target_band:.1f}</text>
        <circle cx="540" cy="120" r="58" fill="none" stroke="#1A1A1A" stroke-width="2"/>
        <circle cx="540" cy="120" r="50" fill="none" stroke="#C9A227" stroke-width="1"/>
        <text x="540" y="112" text-anchor="middle" font-family="Inter, sans-serif" font-size="10" fill="#5B5B5B">OVERALL BAND</text>
        <text x="540" y="146" text-anchor="middle" font-family="'Source Serif Pro', Georgia, serif"
              font-size="34" font-weight="700" fill="#1A1A1A">{overall:.1f}</text>
    </svg>
    """


def render_fingerprint_bars_html(entries: list[FingerprintEntry]) -> str:
    """Renders the recurring-mistake fingerprint as a clean horizontal bar list."""
    if not entries:
        return (
            '<div style="padding:1.2rem; color:#8A8775; font-size:0.9rem;">'
            "No recurring error pattern detected yet — submit a few more essays "
            "or speaking responses to build your fingerprint."
            "</div>"
        )
    max_count = max(e.count for e in entries) or 1
    trend_styles = {
        "improving": ("#4ADE80", "↓ improving"),
        "worsening": ("#F87171", "↑ worsening"),
        "steady": ("#C9A227", "→ steady"),
        "new": ("#94A3B8", "new"),
    }
    rows: list[str] = []
    for entry in entries:
        width_pct = max(8, int((entry.count / max_count) * 100))
        color, trend_label = trend_styles.get(entry.trend, ("#94A3B8", entry.trend))
        # NOTE: built as a single line with no embedded newlines/indentation.
        # Streamlit's markdown renderer treats a multi-line f-string here as
        # Markdown source, not pure HTML. Once two rows are joined, the
        # whitespace-only line left between them reads as a "blank line",
        # which ends the raw-HTML block early — every row after the first
        # then gets reinterpreted as an indented code block and prints as
        # literal "<div style=...>" text instead of rendering. Keeping each
        # row on one line sidesteps that entirely.
        row = (
            '<div style="margin-bottom: 0.85rem;">'
            '<div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.3rem;">'
            f'<span style="color:#F7F5F0; font-weight:600;">{html.escape(entry.label)}</span>'
            f'<span style="color:{color}; font-size:0.75rem; font-weight:600;">{trend_label} · ×{entry.count}</span>'
            "</div>"
            '<div style="background:#1E1B14; border-radius:6px; height:8px; overflow:hidden;">'
            f'<div style="background:#C9A227; width:{width_pct}%; height:100%; border-radius:6px;"></div>'
            "</div>"
            "</div>"
        )
        rows.append(row)
    return '<div style="padding: 0.4rem 0;">' + "".join(rows) + "</div>"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 11 — PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IELTS AI Tutor — Examiner-Grade Feedback",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 12 — DESIGN SYSTEM (CSS)
# ══════════════════════════════════════════════════════════════════════════
# Design concept: the "official examination document," not "AI SaaS
# dashboard." Ink-black surfaces, parchment panels, a single confident
# gold accent that reads as a certificate seal rather than a generic
# product blue. Serif display type for anything that resembles a band
# score or headline; a clean grotesque for body and controls; a monospace
# face for anything numeric and exam-like (timers, scores, word counts).

DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --ink: #0B0F14;
    --ink-raised: #11161D;
    --ink-border: #232B36;
    --parchment: #F7F5F0;
    --parchment-dim: #C9C6BC;
    --gold: #C9A227;
    --gold-bright: #E0BE4A;
    --slate: #8A8775;
    --brick: #9B4A3F;
    --sage: #5E8C6A;
}

html, body, .stApp {
    background-color: var(--ink) !important;
    color: var(--parchment);
    font-family: 'Inter', sans-serif;
}

/* ---- Kill default Streamlit chrome ---- */
#MainMenu, header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }

/* ---- Headline / serif treatment ---- */
h1, h2, h3 {
    font-family: 'Source Serif Pro', Georgia, serif !important;
    color: var(--parchment) !important;
    letter-spacing: -0.01em;
}

/* ---- Masthead ---- */
.masthead {
    border: 1px solid var(--ink-border);
    border-top: 3px solid var(--gold);
    background: var(--ink-raised);
    padding: 1.7rem 2rem;
    margin-bottom: 1.6rem;
    border-radius: 4px;
}
.masthead-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}
.masthead h1 {
    font-size: 2.0rem;
    margin: 0 0 0.35rem 0;
    font-weight: 700;
}
.masthead p {
    color: var(--slate);
    font-size: 0.98rem;
    margin: 0;
    max-width: 640px;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #080B0F;
    border-right: 1px solid var(--ink-border);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: var(--gold) !important;
    font-size: 1.0rem;
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
    color: var(--parchment-dim) !important;
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    background: var(--gold);
    color: #16130A;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    border: none;
    box-shadow: 0 3px 0 rgba(0,0,0,0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: scale(1.025) translateY(-1px);
    background: var(--gold-bright);
    box-shadow: 0 5px 0 rgba(0,0,0,0.3);
    color: #16130A;
}
.stButton > button:active {
    transform: scale(0.98) translateY(0px);
    box-shadow: 0 1px 0 rgba(0,0,0,0.3);
}
.stButton > button[kind="secondary"] {
    background: transparent;
    border: 1px solid var(--ink-border);
    color: var(--parchment-dim);
    box-shadow: none;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--gold);
    color: var(--gold);
    transform: none;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--ink-border);
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 8px 8px 0 0;
    padding: 0 1.3rem;
    background-color: transparent;
    color: var(--slate);
    font-weight: 600;
    font-size: 0.92rem;
}
.stTabs [aria-selected="true"] {
    background-color: var(--ink-raised) !important;
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold);
}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {
    background: var(--ink-raised);
    border: 1px solid var(--ink-border);
    border-radius: 10px;
    padding: 1.0rem 0.9rem;
}
div[data-testid="stMetricLabel"] {
    color: var(--slate) !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] {
    color: var(--parchment) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* ---- Inputs ---- */
.stTextArea textarea, .stTextInput input {
    background-color: var(--ink-raised) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--ink-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold) !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--ink-raised) !important;
    border-color: var(--ink-border) !important;
    color: var(--parchment) !important;
}

/* ---- Slider ---- */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--gold) !important;
}

/* ---- Cards ---- */
.doc-card {
    background: var(--ink-raised);
    border: 1px solid var(--ink-border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.0rem;
}
.doc-card h4 {
    font-family: 'Inter', sans-serif !important;
    color: var(--gold) !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    margin: 0 0 0.8rem 0;
}
.doc-card-body {
    color: var(--parchment-dim);
    line-height: 1.7;
    font-size: 0.96rem;
}

/* ---- Severity badges ---- */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.18rem 0.55rem;
    border-radius: 5px;
    letter-spacing: 0.03em;
}
.badge-gold { background: rgba(201,162,39,0.15); color: var(--gold-bright); border: 1px solid rgba(201,162,39,0.4); }
.badge-sage { background: rgba(94,140,106,0.15); color: var(--sage); border: 1px solid rgba(94,140,106,0.4); }
.badge-brick { background: rgba(155,74,63,0.15); color: #D98A7E; border: 1px solid rgba(155,74,63,0.4); }

/* ---- Streak / word-of-day chip ---- */
.chip-row { display: flex; gap: 0.7rem; margin-bottom: 1rem; flex-wrap: wrap; }
.chip {
    background: var(--ink-raised);
    border: 1px solid var(--ink-border);
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    font-size: 0.85rem;
    color: var(--parchment-dim);
}
.chip b { color: var(--gold); font-family: 'JetBrains Mono', monospace; }

/* ---- History list ---- */
.history-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--ink-raised);
    border: 1px solid var(--ink-border);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 0.6rem;
}
.history-meta { color: var(--slate); font-size: 0.8rem; }
.history-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--gold);
}

/* ---- Misc ---- */
hr { border-color: var(--ink-border) !important; }
[data-testid="stExpander"] {
    background-color: var(--ink-raised);
    border: 1px solid var(--ink-border);
    border-radius: 8px;
}
.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

/* ---- Exam timer ---- */
.timer-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    text-align: center;
    color: var(--parchment);
    letter-spacing: 0.02em;
    margin: 0.2rem 0 0.6rem 0;
}
.timer-display.timer-warning {
    color: #F87171;
    animation: timer-flash 1s infinite;
}
@keyframes timer-flash {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}
div[data-testid="stProgress"] div[role="progressbar"] > div {
    background-color: var(--gold) !important;
    transition: width 0.3s ease, background-color 0.3s ease;
}
.timer-bar-warning div[data-testid="stProgress"] div[role="progressbar"] > div {
    background-color: #F87171 !important;
    animation: timer-flash 1s infinite;
}

/* ---- Word-count goal bar ---- */
.word-goal-track {
    background: #1E1B14;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    border: 1px solid var(--ink-border);
    margin-top: 0.4rem;
}
.word-goal-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.25s ease, background 0.25s ease;
}
.word-goal-label {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--parchment-dim);
    margin-top: 0.35rem;
}

/* ---- Score dial (calculator tab) ---- */
.score-dial-wrap { text-align: center; padding: 0.6rem 0 0.2rem 0; }

/* ---- Clickable session-history sidebar buttons ---- */
.stButton > button.history-pill { font-size: 0.78rem !important; }
</style>
"""

st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 13 — BOOT SEQUENCE
# ══════════════════════════════════════════════════════════════════════════

init_session_state()
register_daily_visit()
profile: StudentProfile = st.session_state.profile
groq: GroqClient = st.session_state.groq_client
groq_whisper: GroqWhisperClient = st.session_state.groq_whisper_client


def severity_badge(band: float) -> str:
    """Returns an HTML badge classed by band severity, used across all tabs."""
    if band >= BAND_EXCELLENT:
        return f'<span class="badge badge-sage">BAND {band:.1f} · STRONG</span>'
    if band >= BAND_GOOD:
        return f'<span class="badge badge-gold">BAND {band:.1f} · ON TRACK</span>'
    if band >= BAND_DEVELOPING:
        return f'<span class="badge badge-gold">BAND {band:.1f} · DEVELOPING</span>'
    return f'<span class="badge badge-brick">BAND {band:.1f} · NEEDS WORK</span>'


def render_markdown_body(text: str) -> str:
    """Converts plain '- ' bullet feedback text into safe HTML list markup."""
    if not text:
        return "<p>No feedback was returned. Please try analysing again.</p>"
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    html_parts = []
    in_list = False
    for line in lines:
        if line.startswith("- "):
            if not in_list:
                html_parts.append("<ul style='margin:0 0 0 0; padding-left:1.2rem;'>")
                in_list = True
            html_parts.append(f"<li style='margin-bottom:0.5rem;'>{html.escape(line[2:])}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        html_parts.append("</ul>")
    return "".join(html_parts)


# ── Masthead ────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="masthead">
        <div class="masthead-eyebrow">Examiner-grade AI feedback · powered by Groq (free)</div>
        <h1>🎓 {APP_NAME}</h1>
        <p>{APP_TAGLINE} Built by an IELTS instructor, graded against the same four
        criteria a real examiner uses — and the only platform that remembers every
        mistake you've ever made, not just the one in front of you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Gamification strip: streak + word of the day ───────────────────────
word_today = WORD_OF_THE_DAY_BANK[st.session_state.word_of_day_index]
st.markdown(
    f"""
    <div class="chip-row">
        <div class="chip">🔥 Study streak: <b>{profile.streak_days} day{'s' if profile.streak_days != 1 else ''}</b></div>
        <div class="chip">📚 Word of the day: <b>{html.escape(word_today.word)}</b> — {html.escape(word_today.definition)}</div>
        <div class="chip">📊 Sessions completed: <b>{profile.total_sessions}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪪 Candidate Profile")
    profile.name = st.text_input("Name", value=profile.name)
    profile.native_language = st.selectbox(
        "Native language",
        options=SUPPORTED_LANGUAGES,
        index=SUPPORTED_LANGUAGES.index(profile.native_language)
        if profile.native_language in SUPPORTED_LANGUAGES else 0,
    )

    st.markdown("---")
    st.markdown("## ⚙️ Exam Configuration")
    task_type = st.selectbox("IELTS Task Type", options=TASK_TYPES, index=2)
    profile.target_band = st.slider(
        "🎯 Target Band Score", min_value=5.0, max_value=9.0, value=profile.target_band, step=0.5
    )

    st.markdown("---")
    st.markdown("## ⏱️ Exam Timer")
    new_preset = st.selectbox(
        "Timer preset",
        options=list(TIMER_PRESETS_SECONDS.keys()),
        index=list(TIMER_PRESETS_SECONDS.keys()).index(st.session_state.timer_preset),
        key="timer_preset_select",
    )
    if new_preset != st.session_state.timer_preset:
        st.session_state.timer_preset = new_preset
        st.session_state.timer_remaining_seconds = TIMER_PRESETS_SECONDS[new_preset]
        st.session_state.timer_running = False
        st.session_state.timer_last_tick = None

    tick_exam_timer()
    remaining = st.session_state.timer_remaining_seconds
    total_for_preset = TIMER_PRESETS_SECONDS[st.session_state.timer_preset]
    is_warning = remaining > 0 and remaining <= TIMER_WARNING_THRESHOLD_SECONDS
    progress_frac = remaining / total_for_preset if total_for_preset else 0.0

    timer_class = "timer-display timer-warning" if is_warning else "timer-display"
    st.markdown(f'<div class="{timer_class}">{format_mm_ss(remaining)}</div>', unsafe_allow_html=True)

    bar_wrap_class = "timer-bar-warning" if is_warning else ""
    st.markdown(f'<div class="{bar_wrap_class}">', unsafe_allow_html=True)
    st.progress(progress_frac)
    st.markdown("</div>", unsafe_allow_html=True)

    if remaining <= 0 and not st.session_state.timer_running and total_for_preset:
        st.error("⏰ Time's up! Wrap up your answer.")

    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        if st.button("▶ Start", use_container_width=True, key="timer_start"):
            if st.session_state.timer_remaining_seconds <= 0:
                st.session_state.timer_remaining_seconds = total_for_preset
            st.session_state.timer_running = True
            st.session_state.timer_last_tick = datetime.now()
            st.rerun()
    with tcol2:
        if st.button("⏸ Pause", use_container_width=True, key="timer_pause"):
            tick_exam_timer()
            st.session_state.timer_running = False
            st.rerun()
    with tcol3:
        if st.button("↺ Reset", use_container_width=True, key="timer_reset"):
            st.session_state.timer_remaining_seconds = total_for_preset
            st.session_state.timer_running = False
            st.session_state.timer_last_tick = None
            st.rerun()

    if st.session_state.timer_running:
        st.caption("⏳ Timer running — the display refreshes automatically every second.")
        st.markdown(
            '<meta http-equiv="refresh" content="1">',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## 🗂️ Current Session History")
    st.caption("Click any past attempt to instantly reload its score and feedback — no API call, no re-grading.")
    if not st.session_state.essay_history_log:
        st.markdown(
            '<div style="font-size:0.82rem; color:#8A8775;">No graded essays yet this visit. '
            "Submit one in Writing Evaluation and it will appear here.</div>",
            unsafe_allow_html=True,
        )
    else:
        for idx, entry in enumerate(reversed(st.session_state.essay_history_log)):
            real_idx = len(st.session_state.essay_history_log) - 1 - idx
            label = f"Band {entry['submission'].score.overall:.1f} · {entry['submission'].task_type} · {entry['submission'].timestamp}"
            if st.button(label, key=f"history_pill_{real_idx}", use_container_width=True):
                st.session_state.selected_history_index = real_idx
                st.session_state.last_writing_submission = entry["submission"]
                st.session_state.band9_rewrite = entry.get("band9_rewrite")
                st.rerun()

    st.markdown("---")
    st.markdown("## 🧬 Error Fingerprint")
    top_issue = ErrorFingerprintEngine.focus_recommendation(profile)
    if top_issue:
        st.markdown(
            f'<div style="font-size:0.82rem; color:#C9C6BC; line-height:1.5;">{top_issue}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.82rem; color:#8A8775;">Submit your first essay or speaking '
            "response to start building your mistake fingerprint.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.72rem; color: #8A8775; font-family: 'JetBrains Mono', monospace;">
            MODEL: {GROQ_CHAT_MODEL}<br>
            MODE: STRICT EXAMINER<br>
            STATUS: {"● GROQ GRADING CONNECTED" if groq.is_configured else "○ GROQ KEY NOT SET"}<br>
            STATUS: {"● GROQ VOICE CONNECTED" if groq_whisper.is_configured else "○ GROQ KEY NOT SET"}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 14 — MAIN TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════

tab_writing, tab_speaking, tab_vocab, tab_fingerprint, tab_history, tab_calculator, tab_about = st.tabs(
    [
        "📝 Writing Evaluation",
        "🗣️ Speaking Simulator",
        "💡 Vocabulary Upgrades",
        "🧬 Error Fingerprint",
        "📊 Progress History",
        "🎛️ Score Calculator",
        "👤 About",
    ]
)


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — WRITING EVALUATION
# ──────────────────────────────────────────────────────────────────────────
with tab_writing:
    # ── Prompt Generator Engine ─────────────────────────────────────────
    gen_col1, gen_col2 = st.columns([4, 1.4])
    with gen_col1:
        st.markdown('<div class="section-eyebrow">Essay topic / prompt</div>', unsafe_allow_html=True)
        topic_text = st.text_area(
            "Essay topic",
            value=st.session_state.generated_prompt_text,
            height=70,
            placeholder="Paste a prompt here, or click Surprise Me to generate one instantly →",
            label_visibility="collapsed",
            key="topic_text_input",
        )
        st.session_state.generated_prompt_text = topic_text
    with gen_col2:
        st.markdown('<div class="section-eyebrow">&nbsp;</div>', unsafe_allow_html=True)
        if st.button("🎲 Surprise Me", use_container_width=True, key="surprise_me_prompt"):
            choice = random.choice(WRITING_PROMPT_BANK)
            st.session_state.generated_prompt_text = f"[{choice.category}] {choice.prompt}"
            st.rerun()

    col_input, col_cert = st.columns([1.4, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-eyebrow">Submit your essay</div>', unsafe_allow_html=True)
        essay_text = st.text_area(
            "Essay input",
            height=260,
            placeholder=f"Paste your full {task_type} response here...",
            label_visibility="collapsed",
            key="essay_text_input",
        )
        word_count = len(essay_text.split()) if essay_text else 0

        # ── Dynamic Word Count Goal Bar ─────────────────────────────────
        word_goal = WORD_GOALS.get(st.session_state.timer_preset, 250)
        goal_frac = min(1.4, word_count / word_goal) if word_goal else 0.0
        goal_pct_of_target = (word_count / word_goal * 100) if word_goal else 0.0
        if goal_pct_of_target >= 100:
            bar_color, status_label = "#34D399", "Goal met ✓"
        elif goal_pct_of_target >= 50:
            bar_color, status_label = "#F59E0B", "Getting there"
        else:
            bar_color, status_label = "#DC2626", "Keep writing"
        fill_width = min(100, goal_frac * 100)
        st.markdown(
            f"""
            <div class="word-goal-track">
                <div class="word-goal-fill" style="width:{fill_width:.1f}%; background:{bar_color};"></div>
            </div>
            <div class="word-goal-label">
                <span>{word_count} / {word_goal} words</span>
                <span style="color:{bar_color}; font-weight:700;">{status_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 Submit for grading", use_container_width=True, key="analyze_writing")

        if analyze_clicked:
            if not essay_text or not essay_text.strip():
                st.warning("⚠️ Please paste an essay before requesting an evaluation.")
            elif not groq.is_configured:
                st.error("🔑 No Groq API key configured. Set GROQ_API_KEY in .streamlit/secrets.toml.")
            else:
                with st.spinner("🧐 Marking against the official IELTS Writing criteria..."):
                    try:
                        user_prompt = (
                            f"IELTS Task Type: {task_type}\n"
                            f"Candidate Target Band Score: {profile.target_band}\n\n"
                            f"Essay to mark:\n\"\"\"\n{essay_text}\n\"\"\""
                        )
                        raw_response = groq.generate(WRITING_EXAMINER_PROMPT, user_prompt)
                        submission = parse_writing_response(raw_response, task_type, profile.target_band, word_count)
                        profile.writing_history.append(submission)
                        ErrorFingerprintEngine.record(profile, submission.mistakes)
                        st.session_state.last_writing_submission = submission
                        st.session_state.band9_rewrite = None  # clear stale rewrite from a previous essay

                        # Log into the Current Session History (Browser Cache History System)
                        st.session_state.essay_history_log.append({
                            "submission": submission,
                            "essay_text": essay_text,
                            "band9_rewrite": None,
                        })
                        st.session_state.selected_history_index = len(st.session_state.essay_history_log) - 1

                        st.success("✅ Evaluation complete — see your scorecard.")
                    except (GroqAPIError, GroqParsingError) as exc:
                        st.error(f"❌ Evaluation failed: {exc}")

    with col_cert:
        st.markdown('<div class="section-eyebrow">Live report form</div>', unsafe_allow_html=True)
        cert_score = (
            st.session_state.last_writing_submission.score.overall
            if st.session_state.last_writing_submission else 0.0
        )
        st.markdown(render_band_certificate_svg(cert_score, profile), unsafe_allow_html=True)

    if st.session_state.last_writing_submission:
        result = st.session_state.last_writing_submission
        st.markdown("---")
        st.markdown('<div class="section-eyebrow">Criteria breakdown</div>', unsafe_allow_html=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Task Achievement", f"{result.score.task_achievement:.1f}")
        with kpi2:
            st.metric("Coherence & Cohesion", f"{result.score.coherence_cohesion:.1f}")
        with kpi3:
            st.metric("Lexical Resource", f"{result.score.lexical_resource:.1f}")
        with kpi4:
            st.metric("Grammatical Range", f"{result.score.grammatical_range:.1f}")

        col_radar, col_feedback = st.columns([1, 1.3], gap="large")
        with col_radar:
            st.markdown('<div class="section-eyebrow">Skill radar</div>', unsafe_allow_html=True)
            st.markdown(render_skill_radar_svg(result.score), unsafe_allow_html=True)
        with col_feedback:
            st.markdown(
                f"""
                <div class="doc-card">
                    <h4>Examiner feedback</h4>
                    <div class="doc-card-body">{render_markdown_body(result.feedback)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Side-by-Side Visual Comparison (Re-Write View) ──────────────
        st.markdown("---")
        st.markdown('<div class="section-eyebrow">Band 9 re-write comparison</div>', unsafe_allow_html=True)
        st.caption("See your original essay polished into a Band 9.0 sample, keeping your own ideas and structure.")

        gen_rewrite_clicked = st.button(
            "✨ Generate Band 9 Sample Version", key="generate_band9_rewrite", use_container_width=False
        )
        if gen_rewrite_clicked:
            if not groq.is_configured:
                st.error("🔑 No Groq API key configured. Set GROQ_API_KEY in .streamlit/secrets.toml.")
            else:
                source_essay = essay_text if essay_text and essay_text.strip() else None
                if not source_essay and st.session_state.selected_history_index is not None:
                    idx = st.session_state.selected_history_index
                    if 0 <= idx < len(st.session_state.essay_history_log):
                        source_essay = st.session_state.essay_history_log[idx]["essay_text"]
                if not source_essay:
                    st.warning("⚠️ No essay text available to rewrite — paste or reload an essay first.")
                else:
                    with st.spinner("✍️ Rewriting your essay to Band 9.0 standard..."):
                        try:
                            rewrite_user_prompt = (
                                f"IELTS Task Type: {result.task_type}\n"
                                f"Essay topic/prompt: {st.session_state.generated_prompt_text or '(not specified)'}\n\n"
                                f"Candidate's original essay:\n\"\"\"\n{source_essay}\n\"\"\""
                            )
                            band9_text = groq.generate(REWRITE_TO_BAND9_PROMPT, rewrite_user_prompt, temperature=0.5)
                            st.session_state.band9_rewrite = band9_text.strip()
                            if st.session_state.selected_history_index is not None:
                                idx = st.session_state.selected_history_index
                                if 0 <= idx < len(st.session_state.essay_history_log):
                                    st.session_state.essay_history_log[idx]["band9_rewrite"] = band9_text.strip()
                        except (GroqAPIError, GroqParsingError) as exc:
                            st.error(f"❌ Rewrite failed: {exc}")

        if st.session_state.band9_rewrite:
            compare_left, compare_right = st.columns(2, gap="large")
            display_original = essay_text if essay_text and essay_text.strip() else (
                st.session_state.essay_history_log[st.session_state.selected_history_index]["essay_text"]
                if st.session_state.selected_history_index is not None
                and 0 <= st.session_state.selected_history_index < len(st.session_state.essay_history_log)
                else ""
            )
            with compare_left:
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <h4>📄 Your original</h4>
                        <div class="doc-card-body" style="white-space:pre-wrap;">{html.escape(display_original)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with compare_right:
                st.markdown(
                    f"""
                    <div class="doc-card" style="border-color: var(--gold);">
                        <h4>✨ Band 9.0 sample version</h4>
                        <div class="doc-card-body" style="white-space:pre-wrap;">{html.escape(st.session_state.band9_rewrite)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("💡 Click **Generate Band 9 Sample Version** above to see the side-by-side comparison.")
    else:
        st.info("👆 Paste an essay above and submit it to receive your full band breakdown.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — SPEAKING SIMULATOR
# ──────────────────────────────────────────────────────────────────────────
with tab_speaking:
    st.markdown(
        '<div class="section-eyebrow">Examiner question</div>',
        unsafe_allow_html=True,
    )

    prompt_col, reroll_col = st.columns([4, 1])
    with prompt_col:
        active_prompt: SpeakingPrompt = st.session_state.active_speaking_prompt
        st.markdown(
            f"""
            <div class="doc-card" style="margin-bottom:0.8rem;">
                <h4>{active_prompt.part} · {html.escape(active_prompt.topic)}</h4>
                <div class="doc-card-body" style="font-size:1.05rem; color:#F7F5F0;">
                    {html.escape(active_prompt.prompt)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reroll_col:
        st.markdown("<div style='padding-top:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 New question", use_container_width=True, key="reroll_speaking"):
            st.session_state.active_speaking_prompt = random.choice(SPEAKING_PROMPT_BANK)
            st.rerun()

    st.markdown(
        '<div class="section-eyebrow" style="margin-top:1rem;">Record your response</div>',
        unsafe_allow_html=True,
    )

    if groq_whisper.is_configured:
        st.caption(
            "🎙️ Record your spoken answer below. It's transcribed automatically with "
            "Groq Whisper, then graded exactly like a typed response."
        )
        audio_clip = st.audio_input("Record your spoken response", label_visibility="collapsed", key="speaking_audio_input")

        if audio_clip is not None:
            current_audio_id = f"{active_prompt.prompt}:{len(audio_clip.getvalue())}"
            if st.session_state.get("transcribed_audio_id") != current_audio_id:
                with st.spinner("🎧 Transcribing your response..."):
                    try:
                        transcript = groq_whisper.transcribe(audio_clip.getvalue())
                        st.session_state.speaking_transcript_box = transcript
                        st.session_state.transcribed_audio_id = current_audio_id
                    except GroqAPIError as exc:
                        st.error(f"❌ Transcription failed: {exc}")
    else:
        st.caption(
            "🎙️ Voice recording needs a Groq API key to transcribe audio. Add GROQ_API_KEY "
            "in .streamlit/secrets.toml to enable the microphone — typing still works below either way."
        )

    st.markdown(
        '<div class="section-eyebrow" style="margin-top:1rem;">Transcript (edit if needed)</div>',
        unsafe_allow_html=True,
    )
    speaking_response = st.text_area(
        "Speaking response",
        height=160,
        placeholder="Your transcribed response will appear here — or type/paste a response directly.",
        label_visibility="collapsed",
        key="speaking_transcript_box",
    )

    if st.button("🚀 Submit for grading", key="analyze_speaking"):
        if not speaking_response or not speaking_response.strip():
            st.warning("⚠️ Please record or type a response before requesting feedback.")
        elif not groq.is_configured:
            st.error("🔑 No Groq API key configured. Set GROQ_API_KEY in .streamlit/secrets.toml.")
        else:
            with st.spinner("🧐 The examiner is reviewing your response..."):
                try:
                    user_prompt = (
                        f"Speaking Part: {active_prompt.part}\n"
                        f"Question/Cue card: {active_prompt.prompt}\n\n"
                        f"Candidate's transcribed response:\n\"\"\"\n{speaking_response}\n\"\"\""
                    )
                    raw_response = groq.generate(SPEAKING_EXAMINER_PROMPT, user_prompt)
                    speaking_submission, mistakes = parse_speaking_response(
                        raw_response, active_prompt.part, active_prompt.prompt, speaking_response
                    )
                    speaking_submission.mistakes = mistakes  # type: ignore[attr-defined]
                    profile.speaking_history.append(speaking_submission)
                    ErrorFingerprintEngine.record(profile, mistakes)
                    st.session_state.last_speaking_submission = speaking_submission
                    st.success("✅ Feedback ready.")
                except (GroqAPIError, GroqParsingError) as exc:
                    st.error(f"❌ Evaluation failed: {exc}")

    if st.session_state.last_speaking_submission:
        sp = st.session_state.last_speaking_submission
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Overall Band", f"{sp.overall_band:.1f}")
        with kpi2:
            st.metric("Fluency", f"{sp.fluency:.1f}")
        with kpi3:
            st.metric("Vocabulary", f"{sp.vocabulary:.1f}")
        with kpi4:
            st.metric("Grammar", f"{sp.grammar:.1f}")

        st.markdown(
            f"""
            <div class="doc-card">
                <h4>Examiner feedback</h4>
                <div class="doc-card-body">{render_markdown_body(sp.feedback)}</div>
                <p style="margin-top:1rem; font-size:0.82rem; color:#8A8775; font-style:italic;">
                    🔊 Pronunciation note: {html.escape(sp.pronunciation_note) if sp.pronunciation_note else "Not available from text alone."}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — VOCABULARY UPGRADES
# ──────────────────────────────────────────────────────────────────────────
with tab_vocab:
    st.markdown('<div class="section-eyebrow">Lexical resource upgrade table</div>', unsafe_allow_html=True)
    st.caption("Weak or repetitive words from your most recent essay, mapped to Band 9.0 alternatives.")

    if st.session_state.last_writing_submission and st.session_state.last_writing_submission.upgrades_table_md:
        st.markdown(st.session_state.last_writing_submission.upgrades_table_md)
    else:
        st.info("💡 Submit an essay in **Writing Evaluation** to generate your personalised upgrade table.")

    st.markdown("---")
    st.markdown('<div class="section-eyebrow">Word bank</div>', unsafe_allow_html=True)
    word_cols = st.columns(2)
    for idx, vocab in enumerate(WORD_OF_THE_DAY_BANK):
        with word_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="doc-card" style="padding:1.0rem 1.2rem; margin-bottom:0.7rem;">
                    <div style="display:flex; justify-content:space-between; align-items:baseline;">
                        <span style="font-family:'Source Serif Pro', serif; font-size:1.1rem; font-weight:700; color:#F7F5F0;">{html.escape(vocab.word)}</span>
                        <span class="badge badge-gold">{html.escape(vocab.band_level)}</span>
                    </div>
                    <p style="color:#8A8775; font-size:0.85rem; margin:0.4rem 0 0.3rem 0;">{html.escape(vocab.definition)}</p>
                    <p style="color:#C9C6BC; font-size:0.85rem; font-style:italic; margin:0;">"{html.escape(vocab.example)}"</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — ERROR FINGERPRINT (signature feature)
# ──────────────────────────────────────────────────────────────────────────
with tab_fingerprint:
    st.markdown('<div class="section-eyebrow">Your recurring mistake fingerprint</div>', unsafe_allow_html=True)
    st.caption(
        "Every writing and speaking submission feeds this profile. Unlike one-off feedback, "
        "this tracks which specific errors keep recurring across your entire history — "
        "and whether they're improving or getting worse."
    )

    fingerprint_entries = ErrorFingerprintEngine.top_recurring(profile, limit=8)

    col_fp_chart, col_fp_explain = st.columns([1.3, 1], gap="large")
    with col_fp_chart:
        st.markdown(
            f'<div class="doc-card"><h4>Top recurring categories</h4>{render_fingerprint_bars_html(fingerprint_entries)}</div>',
            unsafe_allow_html=True,
        )
    with col_fp_explain:
        focus_text = ErrorFingerprintEngine.focus_recommendation(profile)
        st.markdown(
            f"""
            <div class="doc-card">
                <h4>Coaching priority</h4>
                <div class="doc-card-body">
                    {focus_text if focus_text else "No data yet — submit at least one essay or speaking response."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="doc-card">
                <h4>How this works</h4>
                <div class="doc-card-body" style="font-size:0.88rem;">
                    Each graded submission is tagged with the specific error categories
                    the examiner found. This tab accumulates those tags across every
                    submission in your session and compares your earlier attempts to
                    your most recent ones — so a category trending toward
                    <b style="color:#4ADE80;">improving</b> means you're actively fixing
                    it, while <b style="color:#F87171;">worsening</b> flags something
                    new creeping in.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — PROGRESS HISTORY
# ──────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section-eyebrow">Session history</div>', unsafe_allow_html=True)

    all_submissions = sorted(
        [("Writing", s.timestamp, s.score.overall, s.task_type) for s in profile.writing_history]
        + [("Speaking", s.timestamp, s.overall_band, s.part) for s in profile.speaking_history],
        key=lambda row: row[1],
        reverse=True,
    )

    if not all_submissions:
        st.info("📭 No submissions yet this session. Complete a Writing or Speaking task to start tracking progress.")
    else:
        all_bands = [row[2] for row in all_submissions]
        avg_band = sum(all_bands) / len(all_bands)
        best_band = max(all_bands)

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.metric("Total Submissions", len(all_submissions))
        with stat2:
            st.metric("Average Band", f"{avg_band:.1f}")
        with stat3:
            st.metric("Best Band", f"{best_band:.1f}")

        st.markdown("---")
        for module, timestamp, band, label in all_submissions:
            icon = "📝" if module == "Writing" else "🗣️"
            st.markdown(
                f"""
                <div class="history-row">
                    <div>
                        <div style="font-weight:600; color:#F7F5F0;">{icon} {module} — {html.escape(label)}</div>
                        <div class="history-meta">{html.escape(timestamp)}</div>
                    </div>
                    <div class="history-score">{band:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🗑️ Clear session history", type="secondary"):
            profile.writing_history.clear()
            profile.speaking_history.clear()
            profile.mistake_counter.clear()
            st.session_state.last_writing_submission = None
            st.session_state.last_speaking_submission = None
            st.session_state.essay_history_log = []
            st.session_state.selected_history_index = None
            st.session_state.band9_rewrite = None
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# TAB 6 — BAND SCORE CALCULATOR (pure-code, no API call)
# ──────────────────────────────────────────────────────────────────────────
with tab_calculator:
    st.markdown('<div class="section-eyebrow">Diagnostic band score calculator</div>', unsafe_allow_html=True)
    st.caption(
        "Pure arithmetic against the official IELTS conversion tables — instant, free, and works "
        "even without a Groq API key. Move the sliders to see your live overall band estimate."
    )

    calc_col_inputs, calc_col_dial = st.columns([1.3, 1], gap="large")

    with calc_col_inputs:
        st.markdown('<div class="doc-card"><h4>Listening &amp; Reading</h4></div>', unsafe_allow_html=True)
        listening_correct = st.slider("Listening — correct answers (0–40)", min_value=0, max_value=40, value=30, key="calc_listening")
        reading_correct = st.slider("Reading — correct answers (0–40)", min_value=0, max_value=40, value=30, key="calc_reading")

        listening_score = listening_band(listening_correct)
        reading_score = reading_band_academic(reading_correct)

        lcol1, lcol2 = st.columns(2)
        with lcol1:
            st.metric("Listening Band", f"{listening_score:.1f}")
        with lcol2:
            st.metric("Reading Band", f"{reading_score:.1f}")

        st.markdown("---")
        st.markdown('<div class="doc-card"><h4>Writing &amp; Speaking (self-estimate)</h4></div>', unsafe_allow_html=True)
        st.caption("No official raw-score conversion exists for these two — examiners assign the band directly. Estimate your own, or use your latest AI-graded band from the Writing tab.")
        default_writing_band = (
            st.session_state.last_writing_submission.score.overall
            if st.session_state.last_writing_submission else 6.5
        )
        writing_band_input = st.slider("Writing — estimated band", min_value=0.0, max_value=9.0, value=float(default_writing_band), step=0.5, key="calc_writing")
        speaking_band_input = st.slider("Speaking — estimated band", min_value=0.0, max_value=9.0, value=6.5, step=0.5, key="calc_speaking")

    with calc_col_dial:
        overall_estimate = overall_band_from_four(listening_score, reading_score, writing_band_input, speaking_band_input)
        st.markdown('<div class="score-dial-wrap">', unsafe_allow_html=True)
        st.markdown(render_band_dial_svg(overall_estimate, label="ESTIMATED OVERALL BAND"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="doc-card" style="text-align:center;">
                <div class="doc-card-body" style="font-size:0.85rem;">
                    Listening <b style="color:#F7F5F0;">{listening_score:.1f}</b> ·
                    Reading <b style="color:#F7F5F0;">{reading_score:.1f}</b> ·
                    Writing <b style="color:#F7F5F0;">{writing_band_input:.1f}</b> ·
                    Speaking <b style="color:#F7F5F0;">{speaking_band_input:.1f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Rounded per official IELTS rule: .25 rounds up to the next .5; .75 rounds up to the next whole band.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 7 — ABOUT
# ──────────────────────────────────────────────────────────────────────────
with tab_about:
    about_col1, about_col2 = st.columns([1.1, 1], gap="large")

    with about_col1:
        st.markdown(
            """
            <div class="doc-card">
                <h4>About this platform</h4>
                <div class="doc-card-body">
                    <p><b style="color:#F7F5F0;">IELTS AI Tutor</b> exists because real examiner feedback is
                    expensive, slow, and usually capped at one or two mock essays before a test date. This
                    platform gives students unlimited, examiner-grade feedback against the actual IELTS band
                    descriptors — for free, in their own language, available at 2am the night before an exam.</p>
                    <p>Every feature here is built to solve a real bottleneck in IELTS prep: the
                    <b style="color:var(--gold);">Error Fingerprint</b> tracks recurring mistakes across
                    sessions instead of grading each essay in isolation, the
                    <b style="color:var(--gold);">Band 9 Re-Write</b> shows students exactly what their own
                    ideas look like at the top band instead of a generic sample, and the
                    <b style="color:var(--gold);">Score Calculator</b> gives an instant, API-free diagnostic
                    using the same conversion tables the real test uses.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="doc-card">
                <h4>Tech stack</h4>
                <div class="doc-card-body" style="font-size:0.9rem;">
                    Built single-file in Python with Streamlit for the UI layer, Groq's free-tier API
                    (an open-weight chat model plus Whisper) for AI grading and speech transcription, and
                    hand-built SVG for every chart and certificate — no charting library, no database, no
                    auth system. Every visual and number on this page is either real-time arithmetic or a
                    live model call; nothing is mocked.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with about_col2:
        st.markdown(
            """
            <div class="doc-card">
                <h4>Builder</h4>
                <div class="doc-card-body">
                    <p style="font-size:1.05rem; font-weight:700; color:#F7F5F0; margin-bottom:0.2rem;">Atif Al Azad</p>
                    <p style="color:var(--gold); font-size:0.85rem; margin-bottom:0.9rem;">
                        Grade 11 STEM Student (PCM-CS) · DPS Modern Indian School, Doha, Qatar
                    </p>
                    <p>A Grade 11 student specialising in Physics, Chemistry, Mathematics, and Computer
                    Science, with a 98% aggregate in the CBSE Class 10 boards and a Top 10 finish out of
                    360+ students. Certified in JavaScript Essentials (Cisco Networking Academy, top 0.1%
                    score) with a parallel focus on cybersecurity fundamentals and systems engineering.</p>
                    <p>Builds at the intersection of code and cognition — this app is one expression of
                    that, alongside an ongoing research paper on how digital presence and social media
                    algorithms affect the cognitive load of high school students.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="doc-card">
                <h4>Background</h4>
                <div class="doc-card-body" style="font-size:0.88rem;">
                    <p><b style="color:#F7F5F0;">Skills:</b> Python, JavaScript, OOP, algorithm design,
                    system architecture, data analysis, technical writing.</p>
                    <p><b style="color:#F7F5F0;">Languages:</b> English (99%), French (98%), Bangla, Hindi,
                    and Urdu — native or near-native across all five.</p>
                    <p><b style="color:#F7F5F0;">Active in:</b> Science Club and Mathematics Club at
                    DPS-Modern Indian School; volunteered 10 hours at the Qatar Educational Leadership
                    Expo 2026 coordinating registration and exhibitor logistics.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# ARCHITECTURAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
#
# This file is organised as 14 numbered layers, each with a single
# responsibility, in strict dependency order (config → models → exceptions
# → API client → prompts → parsing → content → domain engine → state → UI).
# Nothing above Layer 11 imports Streamlit. That boundary is the whole
# point: Layers 1-10 are pure Python and could be lifted verbatim into
# `domain/`, `services/`, and `infra/` packages the day this becomes a
# real backend, with zero logic rewrites — only import paths change.
#
# Phase B migration path:
#   Layer 1  (Config)      -> config.py, reading from st.secrets / env vars
#   Layer 2  (Models)      -> domain/models.py, becomes SQLAlchemy models
#   Layer 4  (GroqClient)  -> infra/ai_gateway.py, gains retry + caching
#   Layer 5  (Prompts)     -> infra/prompts/ as versioned template files
#   Layer 6  (Parsing)     -> domain/parsing.py, unit-testable in isolation
#   Layer 8  (Fingerprint) -> domain/fingerprint_engine.py, queries Postgres
#                              instead of an in-memory Counter — same API
#   Layer 9  (Session)     -> replaced by a real auth + session service;
#                              StudentProfile becomes a `students` table row
#   Layers 11-14 (UI)      -> React Native screens per the roadmap's
#                              Screen 01-20 build guide, calling the same
#                              domain layer via a thin FastAPI wrapper
#
# The Error Fingerprint Engine (Layer 8) is the one piece worth protecting
# as IP: it is the actual data asset the roadmap's Phase C "Advanced
# analytics... feeds back into better AI training" line is describing.
# Every mistake_tags event emitted today is already shaped like the
# training signal a future proprietary error-classifier would need.