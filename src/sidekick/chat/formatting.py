from __future__ import annotations

import re


AGENT_MARKDOWN_FORMAT_GUIDE = """Response format: portable Markdown-lite.
- Return only the answer. Never discuss formatting rules or decisions.
- Use plain text by default. If formatting is uncertain, keep it as plain text.
- When useful, use **bold**, *italic*, ~~strikethrough~~, `inline code`, fenced code blocks without language labels, hyphen or numbered lists, and [link text](https://example.com).
- Do not emit HTML, # headings, blockquotes, pipe tables, or any other formatting syntax."""

_FENCED_CODE_RE = re.compile(r"```[^\n`]*\n(.*?)(?:\n)?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)\n]+)\)")
_ESCAPED_MARKER_RE = re.compile(r"\\([\\`*_[\]()~])")
_BOLD_RE = re.compile(r"\*\*([^\n]+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_STRIKETHROUGH_RE = re.compile(r"~~([^\n]+?)~~")


def agent_system_prompt(base_prompt: str) -> str:
    return f"{base_prompt.rstrip()}\n\n{AGENT_MARKDOWN_FORMAT_GUIDE}".lstrip()


def markdown_to_plain_text(source: str) -> str:
    protected: list[str] = []
    marker = "\ue000"
    while marker in source:
        marker += "\ue001"

    def protect(value: str) -> str:
        protected.append(value)
        return f"{marker}{len(protected) - 1}{marker}"

    text = _ESCAPED_MARKER_RE.sub(lambda match: protect(match.group(1)), source)
    text = _FENCED_CODE_RE.sub(lambda match: protect(match.group(1)), text)
    text = _INLINE_CODE_RE.sub(lambda match: protect(match.group(1)), text)
    text = _LINK_RE.sub(_plain_link, text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _STRIKETHROUGH_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)

    token_re = re.compile(re.escape(marker) + r"(\d+)" + re.escape(marker))
    return token_re.sub(lambda match: protected[int(match.group(1))], text)


def _plain_link(match: re.Match[str]) -> str:
    label, url = match.groups()
    return label if label == url else f"{label} ({url})"
