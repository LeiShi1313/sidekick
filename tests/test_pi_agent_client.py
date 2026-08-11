from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO

import pytest
from aiohttp import web
from PIL import Image

from sidekick.ai import (
    AgentContext,
    AgentIdentityAnchor,
    AgentMemoryTarget,
    AgentModelCatalog,
    AgentParticipantAccess,
    AgentRequestIdentity,
    AgentRunOrigin,
    AgentRunRequest,
    PiAgentGateway,
)
from sidekick.ai_attachments import AttachmentAnalysisRequest
from sidekick.chat.attachments import (
    MAX_OUTBOUND_ATTACHMENT_BYTES,
    ModelInputImage,
    OutboundAttachment,
)


def model_image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(output, format="JPEG")
    return output.getvalue()


def test_pi_gateway_rejects_weak_credentials() -> None:
    with pytest.raises(ValueError, match="at least 24"):
        PiAgentGateway("http://agent.test:8790", token="short")


async def serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def run_request(
    *,
    model: str | None = None,
    images: tuple[ModelInputImage, ...] = (),
    tool_policy: str = "delegated",
) -> AgentRunRequest:
    return AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt="Calculate 6 * 7",
        context=(AgentContext(kind="reference", text="Prior conversation"),),
        system_prompt="Answer directly.",
        tool_policy=tool_policy,
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor(
                identity="telegram:user:40",
                label="Alice",
            ),
            anchors=(
                AgentIdentityAnchor(
                    identity="telegram:user:40",
                    label="Alice",
                ),
            ),
            requester_can_customize=True,
        ),
        model=model,
        images=images,
        origin=AgentRunOrigin(
            scope_id="telegram:chat:-1001",
            adapter_instance_id="telegram-default",
        ),
        memory=AgentMemoryTarget(
            primary_bank_id="telegram:chat:-1001",
            requester_is_owner=False,
            granted_bank_ids=("qq:group:686743769",),
            participants=(
                AgentParticipantAccess(
                    identity="telegram:user:41",
                    label="Bob",
                    allowed=True,
                    bank_ids=("telegram:chat:-1002",),
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_pi_gateway_streams_validated_ndjson_events() -> None:
    received = None

    async def runs(request: web.Request) -> web.StreamResponse:
        nonlocal received
        assert request.headers["Authorization"] == (
            "Bearer test-agent-token-that-is-long-enough"
        )
        received = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        await response.write(
            b'{"type":"run_started","runId":"11111111-1111-4111-8111-111111111111",'
        )
        await response.write(b'"sessionId":"session-1"}\n')
        await response.write(
            b'{"type":"tool_snapshot","phase":"completed","tool":"code_exec",'
            b'"summary":"Calculation result: 42"}\n'
        )
        await response.write(b'{"type":"text_delta","delta":"42","reset":true}\n')
        await response.write(
            b'{"type":"run_completed","sessionId":"session-1",'
            b'"entryId":"entry-1","answer":"42"}\n'
        )
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/runs", runs)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        events = [
            event
            async for event in gateway.run(run_request(model="gpt-5.4-mini"))
        ]
    finally:
        await gateway.close()
        await runner.cleanup()

    assert received == {
        "runId": "11111111-1111-4111-8111-111111111111",
        "sessionId": None,
        "parentEntryId": None,
        "prompt": "Calculate 6 * 7",
        "context": [{"kind": "reference", "text": "Prior conversation"}],
        "systemPrompt": "Answer directly.",
        "toolPolicy": "delegated",
        "identity": {
            "requester": {"id": "telegram:user:40", "label": "Alice"},
            "anchors": [{"id": "telegram:user:40", "label": "Alice"}],
            "requesterCanCustomize": True,
        },
        "model": "gpt-5.4-mini",
        "origin": {
            "scopeId": "telegram:chat:-1001",
            "adapterInstanceId": "telegram-default",
        },
        "memory": {
            "primaryBankId": "telegram:chat:-1001",
            "requesterIsOwner": False,
            "grantedBankIds": ["qq:group:686743769"],
            "participants": [
                {
                    "id": "telegram:user:41",
                    "label": "Bob",
                    "allowed": True,
                    "bankIds": ["telegram:chat:-1002"],
                }
            ],
        },
    }
    assert [event.type for event in events] == [
        "run_started",
        "tool_snapshot",
        "text_delta",
        "run_completed",
    ]
    assert events[1].summary == "Calculation result: 42"
    assert events[-1].session_id == "session-1"
    assert events[-1].entry_id == "entry-1"


@pytest.mark.asyncio
async def test_pi_gateway_removes_only_the_owner_run_deadline() -> None:
    async def runs(request: web.Request) -> web.Response:
        await request.json()
        await asyncio.sleep(0.15)
        return web.Response(
            text=(
                '{"type":"run_completed","sessionId":"session-1",'
                '"entryId":"entry-1","answer":"done"}\n'
            ),
            content_type="application/x-ndjson",
        )

    app = web.Application()
    app.router.add_post("/v1/runs", runs)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=0.05
    )
    try:
        owner_events = [
            event
            async for event in gateway.run(run_request(tool_policy="owner"))
        ]
        assert owner_events[-1].answer == "done"

        with pytest.raises(TimeoutError):
            async for _ in gateway.run(run_request()):
                pass
    finally:
        await gateway.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_pi_gateway_sends_one_model_image_as_base64() -> None:
    received = None

    async def runs(request: web.Request) -> web.StreamResponse:
        nonlocal received
        received = await request.json()
        return web.Response(
            text=(
                '{"type":"run_completed","sessionId":"session-1",'
                '"entryId":"entry-1","answer":"a fox"}\n'
            ),
            content_type="application/x-ndjson",
        )

    app = web.Application()
    app.router.add_post("/v1/runs", runs)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    image = ModelInputImage(
        mime_type="image/jpeg",
        data=model_image_bytes(),
    )
    try:
        events = [
            event
            async for event in gateway.run(run_request(images=(image,)))
        ]
    finally:
        await gateway.close()
        await runner.cleanup()

    assert events[-1].answer == "a fox"
    assert received is not None
    assert received["images"] == [
        {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(model_image_bytes()).decode("ascii"),
        }
    ]


@pytest.mark.asyncio
async def test_pi_gateway_streams_one_validated_attachment(make_jpeg) -> None:
    image = make_jpeg()

    async def runs(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        for event in (
            {
                "type": "run_started",
                "runId": run_request().run_id,
                "sessionId": "session-1",
            },
            {
                "type": "attachment",
                "filename": "generated-image.jpg",
                "mimeType": "image/jpeg",
                "displayAs": "image",
                "data": base64.b64encode(image).decode("ascii"),
            },
            {
                "type": "run_completed",
                "sessionId": "session-1",
                "entryId": "entry-1",
                "answer": "Here is the generated image.",
            },
        ):
            await response.write(json.dumps(event).encode() + b"\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/runs", runs)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        events = [event async for event in gateway.run(run_request())]
    finally:
        await gateway.close()
        await runner.cleanup()

    assert [event.type for event in events] == [
        "run_started",
        "attachment",
        "run_completed",
    ]
    assert events[1].attachment == OutboundAttachment(
        data=image,
        filename="generated-image.jpg",
        mime_type="image/jpeg",
        display_as="image",
    )


@pytest.mark.asyncio
async def test_pi_gateway_accepts_a_maximum_attachment_next_to_another_event() -> None:
    attachment_data = b"x" * MAX_OUTBOUND_ATTACHMENT_BYTES

    async def runs(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        events = (
            {
                "type": "text_delta",
                "delta": "p" * 1_300,
                "reset": True,
            },
            {
                "type": "attachment",
                "filename": "generated.bin",
                "mimeType": "application/octet-stream",
                "displayAs": "file",
                "data": base64.b64encode(attachment_data).decode("ascii"),
            },
            {
                "type": "text_delta",
                "delta": "p" * 4_000,
                "reset": True,
            },
            {
                "type": "run_completed",
                "sessionId": "session-1",
                "entryId": "entry-1",
                "answer": "Attached.",
            },
        )
        payload = b"".join(json.dumps(event).encode() + b"\n" for event in events)
        await response.write(payload)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/runs", runs)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        events = [event async for event in gateway.run(run_request())]
    finally:
        await gateway.close()
        await runner.cleanup()

    assert [event.type for event in events] == [
        "text_delta",
        "attachment",
        "text_delta",
        "run_completed",
    ]
    assert events[1].attachment is not None
    assert len(events[1].attachment.data) == MAX_OUTBOUND_ATTACHMENT_BYTES
    assert events[3].answer == "Attached."


@pytest.mark.asyncio
async def test_pi_gateway_lists_a_validated_model_catalog() -> None:
    async def models(request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == (
            "Bearer test-agent-token-that-is-long-enough"
        )
        return web.json_response(
            {
                "defaultModel": "gpt-5.6-sol",
                "models": ["gpt-5.4-mini", "gpt-5.6-sol"],
            }
        )

    app = web.Application()
    app.router.add_get("/v1/models", models)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        catalog = await gateway.list_models()
    finally:
        await gateway.close()
        await runner.cleanup()

    assert catalog == AgentModelCatalog(
        default_model="gpt-5.6-sol",
        models=("gpt-5.4-mini", "gpt-5.6-sol"),
    )


@pytest.mark.asyncio
async def test_pi_gateway_rejects_a_malformed_model_catalog() -> None:
    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "defaultModel": "gpt-5.6-sol",
                "models": ["gpt-5.6-sol", 42],
            }
        )

    app = web.Application()
    app.router.add_get("/v1/models", models)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        with pytest.raises(RuntimeError, match="catalog is malformed"):
            await gateway.list_models()
    finally:
        await gateway.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_pi_gateway_rejects_a_malformed_or_incomplete_stream() -> None:
    async def runs(request: web.Request) -> web.Response:
        return web.Response(
            text='{"type":"text_delta","delta":42,"reset":true}\n',
            content_type="application/x-ndjson",
        )

    app = web.Application()
    app.router.add_post("/v1/runs", runs)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        with pytest.raises(RuntimeError, match="invalid event"):
            async for _ in gateway.run(run_request()):
                pass
    finally:
        await gateway.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_pi_gateway_cancels_by_run_id() -> None:
    cancelled = None

    async def cancel(request: web.Request) -> web.Response:
        nonlocal cancelled
        assert request.headers["Authorization"] == (
            "Bearer test-agent-token-that-is-long-enough"
        )
        cancelled = request.match_info["run_id"]
        return web.json_response({"cancelled": True})

    app = web.Application()
    app.router.add_post("/v1/runs/{run_id}/cancel", cancel)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        assert await gateway.cancel(run_request().run_id) is True
    finally:
        await gateway.close()
        await runner.cleanup()

    assert cancelled == run_request().run_id


@pytest.mark.asyncio
async def test_pi_gateway_sends_bounded_attachment_for_description() -> None:
    received = None

    async def describe(request: web.Request) -> web.Response:
        nonlocal received
        assert request.headers["Authorization"] == (
            "Bearer test-agent-token-that-is-long-enough"
        )
        received = await request.json()
        return web.json_response(
            {"description": "Description: a diagram.\nVisible text: API"}
        )

    app = web.Application()
    app.router.add_post("/v1/attachments/describe", describe)
    runner, base_url = await serve(app)
    gateway = PiAgentGateway(
        base_url, token="test-agent-token-that-is-long-enough", timeout=5
    )
    try:
        result = await gateway.describe_attachment(
            AttachmentAnalysisRequest(
                kind="image",
                mime_type="image/jpeg",
                filename="diagram.jpg",
                data=b"image-bytes",
            )
        )
    finally:
        await gateway.close()
        await runner.cleanup()

    assert result == "Description: a diagram.\nVisible text: API"
    assert received == {
        "kind": "image",
        "mimeType": "image/jpeg",
        "filename": "diagram.jpg",
        "data": "aW1hZ2UtYnl0ZXM=",
    }
