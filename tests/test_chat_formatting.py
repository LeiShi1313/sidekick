from sidekick.chat.formatting import (
    agent_system_prompt,
    has_streamable_markdown_content,
    markdown_to_plain_text,
    sanitize_rich_markdown,
)
from sidekick.chat.output_policy import MAINLAND_MESSAGING_POLICY_ID


def test_agent_prompt_requests_one_portable_content_only_format():
    prompt = agent_system_prompt("Keep answers factual.")

    assert prompt.startswith("Keep answers factual.")
    assert "portable Markdown-lite" in prompt
    assert "Use plain text by default" in prompt
    assert "**bold**" in prompt
    assert "*italic*" in prompt
    assert "`inline code`" in prompt
    assert "[link text](https://example.com)" in prompt
    assert "Do not emit HTML" in prompt
    assert "Never discuss formatting rules or decisions" in prompt
    assert "Telegram" not in prompt
    assert "QQ" not in prompt
    assert MAINLAND_MESSAGING_POLICY_ID not in prompt


def test_markdown_to_plain_text_preserves_content_and_link_destinations():
    source = (
        "**Result**\n"
        "*Estimate* and ~~obsolete~~\n"
        "Use `x < y` and [the docs](https://example.com/docs).\n"
        "```\nprint('ok')\n```"
    )

    assert markdown_to_plain_text(source) == (
        "Result\n"
        "Estimate and obsolete\n"
        "Use x < y and the docs (https://example.com/docs).\n"
        "print('ok')"
    )


def test_markdown_to_plain_text_leaves_unfinished_or_literal_markers_visible():
    source = "Use 2 * 3 and unfinished **bold plus `code"

    assert markdown_to_plain_text(source) == source


def test_markdown_to_plain_text_preserves_paired_literal_markers():
    source = "2 * 3 * 4 = 24 and 2 ** 3 ** 4 = 48"

    assert markdown_to_plain_text(source) == source


def test_markdown_to_plain_text_flattens_cjk_adjacent_emphasis():
    source = "这是**重点**内容，也是*斜体*内容。"

    assert markdown_to_plain_text(source) == "这是重点内容，也是斜体内容。"


def test_markdown_to_plain_text_preserves_escapes_inside_code():
    source = "Use `\\*` here.\n```\n\\* stays literal\n```"

    assert markdown_to_plain_text(source) == (
        "Use \\* here.\n\\* stays literal"
    )


def test_markdown_to_plain_text_only_flattens_https_links():
    source = (
        "[safe](https://example.com/docs) "
        "[legacy](http://example.com) "
        "[unsafe](javascript:alert(1))"
    )

    assert markdown_to_plain_text(source) == (
        "safe (https://example.com/docs) "
        "[legacy](http://example.com) "
        "[unsafe](javascript:alert(1))"
    )


def test_rich_markdown_sanitizer_escapes_html_and_unsafe_links_outside_code():
    source = (
        "Use <strong>x & y</strong>, `a < b && c > d`, "
        "[safe](https://example.com?a=1&b=2), and "
        "[unsafe](javascript:alert(1)).\n"
        "```\na < b && c > d\n```"
    )

    assert sanitize_rich_markdown(source) == (
        "Use &lt;strong&gt;x &amp; y&lt;/strong&gt;, `a < b && c > d`, "
        "[safe](https://example.com?a=1&amp;b=2), and "
        "&#91;unsafe](javascript:alert(1)).\n"
        "```\na < b && c > d\n```"
    )


def test_streamable_markdown_content_holds_marker_only_fragments():
    for source in (
        "",
        "**",
        "[",
        "- ",
        "+",
        "1. ",
        "```",
        "|---|",
    ):
        assert has_streamable_markdown_content(source) is False

    for source in (
        "Result",
        "[Result",
        "- Result",
        "1",
        ":-)",
        "[]",
        "✅",
    ):
        assert has_streamable_markdown_content(source) is True
