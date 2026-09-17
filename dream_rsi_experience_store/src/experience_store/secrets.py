"""Conservative redaction for normalized text; raw source files remain untouched."""
import re

PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(\s*[:=]\s*)(['\"]?)([^\s,'\"}]{8,})"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{16,})\b"),
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,})\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----.*?-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", re.S),
]


def redact(text: str | None) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    detected = False
    out = text
    for pattern in PATTERNS:
        def repl(match):
            nonlocal detected
            detected = True
            if match.lastindex and match.lastindex >= 4:
                return f"{match.group(1)}{match.group(2)}[REDACTED_SECRET]"
            return "[REDACTED_SECRET]"
        out = pattern.sub(repl, out)
    return out, detected

