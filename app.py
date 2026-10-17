import re
import json
import streamlit as st
import time
import uuid
from detector import sanitize
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

st.set_page_config(
    page_title="SafeTrace",
    page_icon="🔐",
    layout="wide",
)

# --------------------------------------------------
# Temporary session state
# --------------------------------------------------

import time
import uuid

SESSION_TIMEOUT_SECONDS = 15 * 60  # 15 minutes

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_activity = time.time()
    st.session_state.scan_result = None
    st.session_state.raw_logs = ""

# Expire inactive session data.
if time.time() - st.session_state.last_activity > SESSION_TIMEOUT_SECONDS:
    st.session_state.scan_result = None
    st.session_state.raw_logs = ""
    st.session_state.session_id = str(uuid.uuid4())

st.session_state.last_activity = time.time()


st.title("🔐 SafeTrace")
st.subheader("Debug smarter. Share safer.")
st.write(
    "Detect sensitive information in debugging logs "
    "and redact it before sharing."
)

st.info(
    "Detection runs locally. This version does not send "
    "your logs to an external AI service."
)


# --------------------------------------------------
# 1. Load the local NLP engine
# --------------------------------------------------


@st.cache_resource
def load_analyzer():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_lg",
                }
            ],
        }
    )

    return AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"],
    )

analyzer = load_analyzer()


# --------------------------------------------------
# 2. Custom secret recognizers
# --------------------------------------------------

SECRET_PATTERNS = [
    (
        "API_KEY",
        re.compile(
            r"""(?i)\b(?:api[_-]?key|client[_-]?secret|
            secret[_-]?key)\b\s*[:=]\s*
            (?P<quote>["']?)
            (?P<value>[^"' \t\r\n,;}&]+)
            (?(quote)(?P=quote))""",
            re.VERBOSE,
        ),
        "value",
    ),
    (
        "PASSWORD",
        re.compile(
            r"""(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*
            (?P<quote>["']?)
            (?P<value>[^"' \t\r\n,;}&]+)
            (?(quote)(?P=quote))""",
            re.VERBOSE,
        ),
        "value",
    ),
    (
        "ACCESS_TOKEN",
        re.compile(
            r"""(?i)\b(?:access[_-]?token|refresh[_-]?token|
            auth[_-]?token)\b\s*[:=]\s*
            (?P<quote>["']?)
            (?P<value>[^"' \t\r\n,;}&]+)
            (?(quote)(?P=quote))""",
            re.VERBOSE,
        ),
        "value",
    ),
    (
        "BEARER_TOKEN",
        re.compile(
            r"(?i)\bBearer\s+(?P<value>[A-Za-z0-9._~+/=-]+)"
        ),
        "value",
    ),
    (
        "DATABASE_PASSWORD",
        re.compile(
            r"://[^:/@\s]+:(?P<value>[^@/\s]+)(?=@)"
        ),
        "value",
    ),
    (
        "AWS_ACCESS_KEY",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        None,
    ),
    (
        "GITHUB_TOKEN",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        None,
    ),
    (
        "SLACK_TOKEN",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        None,
    ),
    (
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
            r".*?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        None,
    ),
    (
        "JWT",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{5,}\."
            r"[A-Za-z0-9_-]{5,}\."
            r"[A-Za-z0-9_-]{5,}\b"
        ),
        None,
    ),
]


# Only redact these NLP entity types.
# This avoids automatically redacting every possible entity type,
# such as ordinary dates or potentially misclassified identifiers.
PII_TYPES = {
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
}


# --------------------------------------------------
# 3. Find sensitive spans
# --------------------------------------------------

def find_sensitive_spans(text, analyzer):
    findings = []

    # Custom secret detection
    for entity, pattern, group in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if group:
                start, end = match.span(group)
            else:
                start, end = match.span()

            if start < end:
                findings.append({
                    "type": entity,
                    "start": start,
                    "end": end,
                    "value": text[start:end],
                    "score": 0.99,
                    "source": "Secret pattern",
                })

    # NLP-based PII detection
    results = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.50,
    )

    for result in results:
        if result.entity_type not in PII_TYPES:
            continue

        start = result.start
        end = result.end

        if start >= end:
            continue

        findings.append({
            "type": result.entity_type,
            "start": start,
            "end": end,
            "value": text[start:end],
            "score": round(result.score, 2),
            "source": "Presidio NLP",
        })

    return findings


# --------------------------------------------------
# 4. Resolve overlapping detections and redact
# --------------------------------------------------

def scan_and_redact(text, analyzer):
    findings = find_sensitive_spans(text, analyzer)

    # Prefer custom secret detections when spans overlap.
    findings.sort(
        key=lambda item: (
            item["start"],
            0 if item["source"] == "Secret pattern" else 1,
            -(item["end"] - item["start"]),
        )
    )

    accepted = []

    for item in findings:
        overlaps = any(
            item["start"] < existing["end"]
            and item["end"] > existing["start"]
            for existing in accepted
        )

        if not overlaps:
            accepted.append(item)

    # Consistent placeholders for repeated identical values.
    placeholders = {}
    counters = {}
    sanitized = text

    for item in sorted(
        accepted,
        key=lambda item: item["start"],
        reverse=True,
    ):
        key = (item["type"], item["value"])

        if key not in placeholders:
            entity = item["type"]
            counters[entity] = counters.get(entity, 0) + 1

            placeholders[key] = (
                f"[{entity}_{counters[entity]}]"
            )

        sanitized = (
            sanitized[:item["start"]]
            + placeholders[key]
            + sanitized[item["end"]:]
        )

    return accepted, sanitized


# --------------------------------------------------
# 5. Lightweight residual verification
# --------------------------------------------------

def verify_output(text, analyzer):
    remaining = find_sensitive_spans(text, analyzer)

    return remaining


# --------------------------------------------------
# 6. Streamlit interface
# --------------------------------------------------

try:
    analyzer = load_analyzer()
    st.success("Local detection engine is ready.")
except Exception as error:
    st.error(
        "Could not load the detection engine. "
        "Check the terminal for details."
    )
    st.exception(error)
    st.stop()


sample = '''2026-10-17 14:22:09 ERROR payment_service
POST /api/v1/payments failed with status 401
user_email=developer@example.com
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.demo.signature
api_key=sk_test_example123
TimeoutError: gateway did not respond after 5000ms'''


# --------------------------------------------------
# Privacy: end session and clear temporary data
# --------------------------------------------------

if st.button("🗑️ End session & delete data"):
    keys_to_clear = [
        "raw_logs_input",
        "raw_logs",
        "scan_result",
        "findings",
        "sanitized",
        "residuals",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["last_activity"] = time.time()

    st.rerun()

logs = st.text_area(
    "Paste your debugging logs",
    value="",
    placeholder=sample,
    height=280,
    key="raw_logs_input",
)


if st.button("🔍 Scan and redact", type="primary"):
    if not logs.strip():
        st.warning("Paste some logs first.")

    else:
        try:
            result = sanitize(logs, analyzer)
        except RuntimeError as exc:
            st.error(f"Scan incomplete: {exc}")
            st.warning(
                "Do not share these logs. "
                "SafeTrace could not complete the scan."
            )
            st.stop()

        findings = result["findings"]
        sanitized = result["sanitized"]

        st.divider()
        st.subheader("Scan results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Detected spans",
                len(findings),
            )

        with col2:
            st.metric(
                "Redacted spans",
                result["redacted_count"],
            )

        with col3:
            st.metric(
                "Needs review",
                result["review_count"],
            )

        if findings:
            st.dataframe(
                [
                    {
                        "Type": item["kind"],
                        "Risk score": item["score"],
                        "Decision": item["action"],
                        "Evidence": ", ".join(item["reasons"]),
                    }
                    for item in findings
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(
                "No supported sensitive patterns detected."
            )

        st.subheader("Sanitized output")

        st.code(
            sanitized,
            language="text",
        )

        st.download_button(
            "⬇️ Download sanitized logs",
            data=sanitized,
            file_name="sanitized.log",
            mime="text/plain",
        )

        st.subheader("Privacy verification")

        if result["residual_count"] > 0:
            st.warning(
                f"{result['residual_count']} possible residual "
                "finding(s) remain. Review before sharing."
            )

            if result["residuals"]:
                st.dataframe(
                    [
                        {
                            "Type": item["kind"],
                            "Risk score": item["score"],
                            "Decision": item["action"],
                        }
                        for item in result["residuals"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

        else:
            st.info(
                "The second scan found no supported patterns. "
                "This does not guarantee complete sanitization."
            )

        st.subheader("Log structure detected")

        st.dataframe(
            result["line_types"],
            hide_index=True,
            use_container_width=True,
        )

        if result["review_count"] > 0:
            st.warning(
                "Some findings need manual review before "
                "you share the logs."
            )

        st.warning(
            "Review the sanitized output before sharing. "
            "Unknown secrets and false positives are possible."
        )
