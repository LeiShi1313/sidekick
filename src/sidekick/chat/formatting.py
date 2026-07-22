from __future__ import annotations

import html
import re


AGENT_MARKDOWN_FORMAT_GUIDE = """Response format: portable Markdown-lite.
- Return only the answer. Never discuss formatting rules or decisions.
- Use plain text by default. If formatting is uncertain, keep it as plain text.
- When useful, use **bold**, *italic*, ~~strikethrough~~, `inline code`, fenced code blocks without language labels, hyphen or numbered lists, and [link text](https://example.com).
- Do not emit HTML, # headings, blockquotes, pipe tables, or any other formatting syntax."""

PORTABLE_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https://[^\s)\n]+)\)")

_ANY_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]*)\)")
_FENCED_CODE_RE = re.compile(r"```[^\n`]*\n(.*?)(?:\n)?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_ESCAPED_MARKER_RE = re.compile(r"\\([\\`*_[\]()~])")
_BOLD_RE = re.compile(
    r"(?<!\*)\*\*(?=\S)([^\n]*?\S)\*\*(?!\*)"
)
_ITALIC_RE = re.compile(
    r"(?<!\*)\*(?=\S)([^*\n]*?\S)\*(?!\*)"
)
_STRIKETHROUGH_RE = re.compile(
    r"(?<!~)~~(?=\S)([^\n]*?\S)~~(?!~)"
)
_STREAM_MARKER_ONLY_RE = re.compile(
    r"(?:\*+|_+|~+|`+|\[|#+|>+|[-+]|\d+[.)])"
)
_STREAM_LIST_OPENER_RE = re.compile(r"\s*(?:[-+*]|\d+[.)])\s+")
_STREAM_TABLE_SEPARATOR_RE = re.compile(r"\|[\s:|-]*\|")


class _PlaceholderStore:
    def __init__(self, source: str) -> None:
        marker = "\ue000"
        while marker in source:
            marker += "\ue001"
        self._marker = marker
        self._values: list[str] = []

    def protect(self, value: str) -> str:
        self._values.append(value)
        return f"{self._marker}{len(self._values) - 1}{self._marker}"

    def restore(self, text: str) -> str:
        token_re = re.compile(
            re.escape(self._marker) + r"(\d+)" + re.escape(self._marker)
        )
        return token_re.sub(
            lambda match: self._values[int(match.group(1))],
            text,
        )


def agent_system_prompt(base_prompt: str) -> str:
    return f"{base_prompt.rstrip()}\n\n{AGENT_MARKDOWN_FORMAT_GUIDE}".lstrip()


def markdown_to_plain_text(source: str) -> str:
    placeholders = _PlaceholderStore(source)
    text = _FENCED_CODE_RE.sub(
        lambda match: placeholders.protect(match.group(1)),
        source,
    )
    text = _INLINE_CODE_RE.sub(
        lambda match: placeholders.protect(match.group(1)),
        text,
    )
    text = _ESCAPED_MARKER_RE.sub(
        lambda match: placeholders.protect(match.group(1)),
        text,
    )
    text = _ANY_LINK_RE.sub(
        lambda match: (
            _plain_link(match)
            if PORTABLE_LINK_RE.fullmatch(match.group(0))
            else placeholders.protect(match.group(0))
        ),
        text,
    )
    text = _BOLD_RE.sub(r"\1", text)
    text = _STRIKETHROUGH_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)

    return placeholders.restore(text)


def sanitize_rich_markdown(source: str) -> str:
    placeholders = _PlaceholderStore(source)
    text = _FENCED_CODE_RE.sub(
        lambda match: placeholders.protect(match.group(0)),
        source,
    )
    text = _INLINE_CODE_RE.sub(
        lambda match: placeholders.protect(match.group(0)),
        text,
    )
    text = _ANY_LINK_RE.sub(
        lambda match: (
            match.group(0)
            if PORTABLE_LINK_RE.fullmatch(match.group(0))
            else f"{placeholders.protect('&#91;')}{match.group(0)[1:]}"
        ),
        text,
    )
    return placeholders.restore(html.escape(text, quote=False))


def has_streamable_markdown_content(source: str) -> bool:
    stripped = source.strip()
    if not stripped:
        return False
    if _STREAM_MARKER_ONLY_RE.fullmatch(stripped):
        return False
    if _STREAM_LIST_OPENER_RE.fullmatch(source):
        return False
    return _STREAM_TABLE_SEPARATOR_RE.fullmatch(stripped) is None


def _plain_link(match: re.Match[str]) -> str:
    label, url = match.groups()
    return label if label == url else f"{label} ({url})"
