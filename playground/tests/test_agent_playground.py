from __future__ import annotations

import json

import aiohttp
from aiohttp import web
import pytest

from agent_playground.app import (
    PlaygroundSettings,
    UpstreamUnavailable,
    _parse_pi_event,
    _parse_run_audit,
    create_app,
)

PI_TOKEN = "private-pi-token-that-is-long-enough"
MEMORY_TOKEN = "private-memory-token-that-is-long-enough"
CHANNEL_TOKEN = "private-channel-token-that-is-long-enough"
PNG_DATA = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


async def start(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def dependencies(
    *, bank_id: str = "chat:engineering"
) -> tuple[list[web.AppRunner], str, str, dict]:
    received = {
        "recalls": [],
        "runs": [],
        "cancelled": [],
        "session_queries": [],
        "audit_queries": [],
    }

    memory = web.Application()

    async def memory_health(_):
        return web.json_response({"status": "healthy"})

    async def banks(request):
        assert request.headers["Authorization"] == f"Bearer {MEMORY_TOKEN}"
        return web.json_response(
            {
                "banks": [
                    {
                        "bank_id": bank_id,
                        "name": "Engineering",
                        "fact_count": 12,
                    }
                ]
            }
        )

    async def recall(request):
        assert request.headers["Authorization"] == f"Bearer {MEMORY_TOKEN}"
        received["recalls"].append(await request.json())
        return web.json_response(
            {
                "results": [
                    {
                        "id": "memory-1",
                        "text": "Alice maintains the deployment pipeline.",
                        "type": "world",
                        "entities": ["Alice", "actor:alice"],
                        "occurred_start": "2026-07-13T12:00:00Z",
                        "mentioned_at": "2026-07-13T12:01:00Z",
                        "document_id": "conversation:41",
                        "chunk_id": "chunk-1",
                    }
                ]
            }
        )

    memory.router.add_get("/health", memory_health)
    memory.router.add_get("/v1/default/banks", banks)
    memory.router.add_post("/v1/default/banks/{bank_id}/memories/recall", recall)
    memory_runner, memory_url = await start(memory)

    pi = web.Application()

    async def pi_health(_):
        return web.json_response({"status": "ok"})

    async def run(request):
        assert request.headers["Authorization"] == f"Bearer {PI_TOKEN}"
        payload = await request.json()
        received["runs"].append(payload)
        response = web.StreamResponse(
            headers={"Content-Type": "application/x-ndjson; charset=utf-8"}
        )
        await response.prepare(request)
        events = (
            {
                "type": "memory_snapshot",
                "primaryBankId": payload["memory"]["primaryBankId"],
                "queries": ["Who owns deploys?"],
                "memories": [
                    {
                        "id": "memory-1",
                        "text": "Alice maintains the deployment pipeline.",
                        "type": "world",
                        "entities": ["Alice", "actor:alice"],
                        "occurredStart": "2026-07-13T12:00:00Z",
                        "occurredEnd": None,
                        "mentionedAt": "2026-07-13T12:01:00Z",
                        "documentId": "conversation:41",
                        "chunkId": "chunk-1",
                    }
                ],
            },
            {
                "type": "run_started",
                "runId": payload["runId"],
                "sessionId": "session-1",
            },
            {
                "type": "tool_snapshot",
                "phase": "completed",
                "tool": "memory_reflect",
                "summary": "Memory reflection completed",
            },
            {
                "type": "attachment",
                "filename": "generated-image.png",
                "mimeType": "image/png",
                "displayAs": "image",
                "data": PNG_DATA,
            },
            {"type": "text_delta", "delta": "Alice owns it.", "reset": True},
            {
                "type": "run_completed",
                "sessionId": "session-1",
                "entryId": "entry-1",
                "answer": "Alice owns it.",
            },
        )
        for event in events:
            await response.write(json.dumps(event).encode() + b"\n")
        await response.write_eof()
        return response

    async def cancel(request):
        assert request.headers["Authorization"] == f"Bearer {PI_TOKEN}"
        received["cancelled"].append(request.match_info["run_id"])
        return web.json_response({"cancelled": True})

    async def sessions(request):
        assert request.headers["Authorization"] == f"Bearer {PI_TOKEN}"
        received["session_queries"].append(dict(request.query))
        return web.json_response(
            {
                "items": [
                    {
                        "id": "session-1",
                        "name": "Deployment ownership",
                        "createdAt": "2026-07-13T12:00:00.000Z",
                        "modifiedAt": "2026-07-13T12:05:00.000Z",
                        "messageCount": 4,
                        "firstMessage": "Who owns deployment?",
                    }
                ],
                "total": 1,
                "nextCursor": None,
            }
        )

    async def session_detail(request):
        assert request.headers["Authorization"] == f"Bearer {PI_TOKEN}"
        assert request.match_info["session_id"] == "session-1"
        return web.json_response(
            {
                "id": "session-1",
                "name": "Deployment ownership",
                "createdAt": "2026-07-13T12:00:00.000Z",
                "modifiedAt": "2026-07-13T12:05:00.000Z",
                "messageCount": 4,
                "firstMessage": "Who owns deployment?",
                "header": {"version": 3, "id": "session-1"},
                "leafId": "entry-2",
                "entries": [
                    {
                        "type": "message",
                        "id": "entry-1",
                        "parentId": None,
                        "timestamp": "2026-07-13T12:00:00.000Z",
                        "message": {
                            "role": "user",
                            "content": "Who owns deployment?",
                        },
                    },
                    {
                        "type": "message",
                        "id": "entry-2",
                        "parentId": "entry-1",
                        "timestamp": "2026-07-13T12:00:01.000Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Alice owns it."}],
                            "usage": {"input": 20, "output": 5},
                        },
                    },
                ],
            }
        )

    async def audits(request):
        assert request.headers["Authorization"] == f"Bearer {PI_TOKEN}"
        if request.query.get("status") == "active":
            return web.json_response({"items": [], "total": 0})
        received["audit_queries"].append(dict(request.query))
        return web.json_response(
            {
                "items": [
                    {
                        "runId": "11111111-1111-4111-8111-111111111111",
                        "sessionId": "session-1",
                        "entryId": "entry-2",
                        "status": "completed",
                        "startedAt": "2026-07-13T12:00:00.000Z",
                        "finishedAt": "2026-07-13T12:00:01.000Z",
                        "prompt": "",
                        "memoryEnabled": True,
                        "memoryScopeId": None,
                        "eventCount": 11,
                    }
                ],
                "total": 1,
                "nextCursor": None,
            }
        )

    async def audit_detail(request):
        assert request.headers["Authorization"] == f"Bearer {PI_TOKEN}"
        run_id = request.match_info["run_id"]
        return web.json_response(
            {
                "runId": run_id,
                "summary": {
                    "status": "completed",
                    "startedAt": "2026-07-13T12:00:00.000Z",
                    "finishedAt": "2026-07-13T12:00:01.000Z",
                    "durationMs": 1_000,
                    "prompt": "Who owns deployment?",
                    "eventCount": 11,
                    "session": {
                        "kind": "root",
                        "id": "session-1",
                        "parentEntryId": None,
                        "entryId": "entry-2",
                    },
                    "model": {
                        "id": "gpt-5",
                        "provider": "openai",
                        "thinkingLevel": "medium",
                    },
                    "memory": {
                        "enabled": True,
                        "primaryBankId": None,
                        "route": "cross_bank_queried",
                        "initialRecall": {
                            "status": "completed",
                            "queries": [],
                            "queryCount": 1,
                            "memoryCount": 1,
                            "eventSequence": 3,
                        },
                        "directory": {
                            "status": "available",
                            "query": None,
                            "sourceCount": 2,
                            "eventSequence": 4,
                        },
                    },
                    "tools": [
                        {
                            "callId": "call-1",
                            "name": "memory_reflect",
                            "status": "completed",
                            "durationMs": 42,
                            "query": None,
                            "source": None,
                            "eventSequence": 2,
                        },
                        {
                            "callId": "call-find-1",
                            "name": "memory_find_sources",
                            "status": "completed",
                            "durationMs": 18,
                            "query": None,
                            "source": None,
                            "eventSequence": 5,
                        },
                        {
                            "callId": "call-source-1",
                            "name": "memory_query_source",
                            "status": "completed",
                            "durationMs": 31,
                            "query": None,
                            "source": {
                                "handle": "source_2",
                                "displayName": None,
                                "bankId": None,
                            },
                            "eventSequence": 7,
                        },
                    ],
                    "warnings": [
                        {
                            "kind": "memory_access",
                            "unavailableBankCount": 1,
                            "eventSequence": 10,
                        }
                    ],
                    "failure": None,
                },
                "events": [
                    {
                        "version": 2,
                        "sequence": 1,
                        "timestamp": "2026-07-13T12:00:00.000Z",
                        "runId": run_id,
                        "type": "memory.http.request",
                        "data": {
                            "exchangeId": "exchange-1",
                            "operation": "recall",
                            "variant": "initial",
                            "toolCallId": None,
                            "method": "POST",
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 2,
                        "timestamp": "2026-07-13T12:00:00.100Z",
                        "runId": run_id,
                        "type": "tool.completed",
                        "data": {
                            "toolCallId": "call-1",
                            "toolName": "memory_reflect",
                            "isError": False,
                            "unavailable": False,
                            "durationMs": 42,
                            "sourceHandle": None,
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 3,
                        "timestamp": "2026-07-13T12:00:00.200Z",
                        "runId": run_id,
                        "type": "memory.context",
                        "data": {
                            "memoryEnabled": True,
                            "queryCount": 1,
                            "memoryCount": 1,
                            "recall": {"status": "completed"},
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 4,
                        "timestamp": "2026-07-13T12:00:00.250Z",
                        "runId": run_id,
                        "type": "memory.directory.result",
                        "data": {"status": "available", "referenceCount": 0},
                    },
                    {
                        "version": 2,
                        "sequence": 5,
                        "timestamp": "2026-07-13T12:00:00.300Z",
                        "runId": run_id,
                        "type": "tool.started",
                        "data": {
                            "toolCallId": "call-find-1",
                            "toolName": "memory_find_sources",
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 6,
                        "timestamp": "2026-07-13T12:00:00.400Z",
                        "runId": run_id,
                        "type": "tool.completed",
                        "data": {
                            "toolCallId": "call-find-1",
                            "toolName": "memory_find_sources",
                            "isError": False,
                            "unavailable": False,
                            "durationMs": 18,
                            "sourceHandle": None,
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 7,
                        "timestamp": "2026-07-13T12:00:00.500Z",
                        "runId": run_id,
                        "type": "tool.started",
                        "data": {
                            "toolCallId": "call-source-1",
                            "toolName": "memory_query_source",
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 8,
                        "timestamp": "2026-07-13T12:00:00.650Z",
                        "runId": run_id,
                        "type": "memory.http.response",
                        "data": {
                            "toolCallId": "call-source-1",
                            "operation": "source.recall",
                            "status": 200,
                            "ok": True,
                            "usable": True,
                            "failureReason": None,
                            "durationMs": 27,
                            "bodyBytes": 100,
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 9,
                        "timestamp": "2026-07-13T12:00:00.700Z",
                        "runId": run_id,
                        "type": "tool.completed",
                        "data": {
                            "toolCallId": "call-source-1",
                            "toolName": "memory_query_source",
                            "isError": False,
                            "unavailable": False,
                            "durationMs": 31,
                            "sourceHandle": "source_2",
                        },
                    },
                    {
                        "version": 2,
                        "sequence": 10,
                        "timestamp": "2026-07-13T12:00:00.800Z",
                        "runId": run_id,
                        "type": "memory.access.warning",
                        "data": {"unavailableBankCount": 1},
                    },
                    {
                        "version": 2,
                        "sequence": 11,
                        "timestamp": "2026-07-13T12:00:01.000Z",
                        "runId": run_id,
                        "type": "run.completed",
                        "data": {
                            "sessionId": "session-1",
                            "entryId": "entry-2",
                            "answerChars": 14,
                        },
                    },
                ],
            }
        )

    pi.router.add_get("/health", pi_health)
    pi.router.add_post("/v1/runs", run)
    pi.router.add_post("/v1/runs/{run_id}/cancel", cancel)
    pi.router.add_get("/v1/sessions", sessions)
    pi.router.add_get("/v1/sessions/{session_id}", session_detail)
    pi.router.add_get("/v1/runs", audits)
    pi.router.add_get("/v1/runs/{run_id}/audit", audit_detail)
    pi_runner, pi_url = await start(pi)
    return [memory_runner, pi_runner], memory_url, pi_url, received


async def request_app(memory_url: str, pi_url: str) -> tuple[web.AppRunner, str]:
    return await start(
        create_app(
            PlaygroundSettings(
                memory_url=memory_url,
                pi_url=pi_url,
                pi_token=PI_TOKEN,
                memory_token=MEMORY_TOKEN,
                channel_token=CHANNEL_TOKEN,
                system_prompt="Use evidence carefully.",
            )
        )
    )


def test_settings_use_generic_environment_names(monkeypatch):
    monkeypatch.setenv("PI_AGENT_TOKEN", PI_TOKEN)
    monkeypatch.setenv("MEMORY_API_TOKEN", MEMORY_TOKEN)
    monkeypatch.setenv("PLAYGROUND_CHANNEL_TOKEN", CHANNEL_TOKEN)
    monkeypatch.setenv("MEMORY_API_URL", "http://memory.internal:8888/")
    monkeypatch.setenv("PI_AGENT_URL", "http://pi.internal:8790/")

    settings = PlaygroundSettings.from_env()

    assert settings.memory_url == "http://memory.internal:8888"
    assert settings.pi_url == "http://pi.internal:8790"
    assert settings.pi_token == PI_TOKEN
    assert settings.memory_token == MEMORY_TOKEN
    assert settings.channel_token == CHANNEL_TOKEN
    assert settings.system_prompt.startswith("You are a helpful assistant")


@pytest.mark.asyncio
async def test_banks_accept_percent_escaped_hindsight_ids():
    bank_id = "wechat:account:wxid_v11uy95lmdjh22:chat:49277108357%40chatroom"
    dependency_runners, memory_url, pi_url, _ = await dependencies(bank_id=bank_id)
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{playground_url}/api/banks") as response,
        ):
            payload = await response.json()

        assert response.status == 200
        assert payload["items"][0]["bank_id"] == bank_id
    finally:
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_empty_system_prompt_uses_the_configured_default():
    dependency_runners, memory_url, pi_url, received = await dependencies()
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{playground_url}/api/runs",
                json={
                    "mode": "agent",
                    "prompt": "Who owns deploys?",
                    "bankId": "chat:engineering",
                    "memoryQuery": None,
                    "recallContext": "",
                    "context": "",
                    "systemPrompt": "",
                    "sessionId": None,
                    "parentEntryId": None,
                },
            ) as response,
        ):
            events = [json.loads(line) async for line in response.content]

        assert response.status == 200
        assert events[-1]["type"] == "run_completed"
        assert received["runs"][0]["systemPrompt"] == "Use evidence carefully."
    finally:
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_agent_run_accepts_proxy_origin_and_streams_pi_events():
    dependency_runners, memory_url, pi_url, received = await dependencies()
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{playground_url}/health") as response:
                assert response.status == 200
                assert await response.json() == {"status": "ok"}

            async with session.get(f"{playground_url}/api/banks") as response:
                banks = await response.json()
            assert banks["items"][0]["name"] == "Engineering"

            async with session.post(
                f"{playground_url}/api/recall",
                json={"bankId": "chat:engineering", "query": "Who owns deploys?"},
            ) as response:
                preview = await response.json()
                assert response.status == 200
            assert preview["memories"][0]["id"] == "memory-1"
            assert "Alice maintains" in preview["context"]

            async with session.post(
                f"{playground_url}/api/runs",
                headers={"Origin": "http://playground.sidekick.localhost:18865"},
                json={
                    "mode": "agent",
                    "prompt": "Who owns deploys?",
                    "bankId": "chat:engineering",
                    "recallContext": "Earlier we discussed deployment ownership.",
                    "context": "A release is scheduled tomorrow.",
                    "sessionId": None,
                    "parentEntryId": None,
                },
            ) as response:
                assert response.status == 200
                assert response.headers["Content-Type"].startswith(
                    "application/x-ndjson"
                )
                events = [json.loads(line) async for line in response.content]

        prepared = events[0]
        assert prepared["type"] == "run_prepared"
        assert prepared["mode"] == "agent"
        assert prepared["toolPolicy"] == "owner"
        assert prepared["memory"] == {
            "bankId": "chat:engineering",
            "query": "Automatic from request and references",
            "memories": [],
            "managedBy": "agent",
            "status": "pending",
        }
        assert prepared["request"]["systemPrompt"] == "Use evidence carefully."
        assert events[-1]["type"] == "run_completed"
        assert events[-1]["answer"] == "Alice owns it."
        assert events[1]["type"] == "memory_snapshot"
        assert events[1]["memories"][0]["id"] == "memory-1"
        attachment = next(event for event in events if event["type"] == "attachment")
        assert attachment == {
            "type": "attachment",
            "filename": "generated-image.png",
            "mimeType": "image/png",
            "displayAs": "image",
            "data": PNG_DATA,
        }

        assert len(received["recalls"]) == 1
        pi_request = received["runs"][0]
        assert pi_request["toolPolicy"] == "owner"
        assert pi_request["memory"] == {
            "primaryBankId": "chat:engineering",
            "requesterIsOwner": True,
            "grantedBankIds": [],
            "participants": [],
        }
        assert pi_request["identity"] == {
            "requester": {
                "id": "playground:user:owner",
                "label": "Playground owner",
            },
            "anchors": [
                {
                    "id": "playground:user:owner",
                    "label": "Playground owner",
                }
            ],
        }
        assert pi_request["origin"] == {
            "scopeId": "playground:owner",
            "adapterInstanceId": "playground",
        }
        assert pi_request["includeMemorySnapshot"] is True
        assert [item["kind"] for item in pi_request["context"]] == [
            "reference",
            "reference",
        ]
        assert "Earlier we discussed" in pi_request["context"][0]["text"]
        assert PI_TOKEN not in json.dumps(events)
    finally:
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_llm_mode_disables_tools_and_supports_cancellation():
    dependency_runners, memory_url, pi_url, received = await dependencies()
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{playground_url}/api/runs",
                json={
                    "mode": "llm",
                    "prompt": "Continue",
                    "bankId": None,
                    "sessionId": "session-1",
                    "parentEntryId": "entry-1",
                },
            ) as response:
                events = [json.loads(line) async for line in response.content]
            run_id = events[0]["runId"]
            async with session.post(
                f"{playground_url}/api/runs/{run_id}/cancel"
            ) as response:
                assert response.status == 200
                assert await response.json() == {"cancelled": True}

        assert received["runs"][0]["toolPolicy"] == "none"
        assert received["runs"][0]["sessionId"] == "session-1"
        assert received["runs"][0]["parentEntryId"] == "entry-1"
        assert received["cancelled"] == [run_id]
    finally:
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_session_history_and_run_audits_are_proxied_without_exposing_token():
    dependency_runners, memory_url, pi_url, received = await dependencies()
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    run_id = "11111111-1111-4111-8111-111111111111"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{playground_url}/api/sessions",
                params={"limit": "20", "q": "deploy"},
            ) as response:
                sessions = await response.json()
                assert response.status == 200
            async with session.get(
                f"{playground_url}/api/sessions/session-1"
            ) as response:
                detail = await response.json()
                assert response.status == 200
            async with session.get(
                f"{playground_url}/api/audits",
                params={"limit": "10", "sessionId": "session-1"},
            ) as response:
                audits = await response.json()
                assert response.status == 200
            async with session.get(f"{playground_url}/api/audits/{run_id}") as response:
                audit = await response.json()
                assert response.status == 200

        assert sessions["items"][0]["id"] == "session-1"
        assert detail["leafId"] == "entry-2"
        assert detail["entries"][1]["message"]["usage"]["input"] == 20
        assert audits["items"][0]["eventCount"] == 11
        assert audit["events"][0]["data"]["method"] == "POST"
        assert audit["events"][1]["data"]["durationMs"] == 42
        assert audit["summary"]["memory"]["route"] == "cross_bank_queried"
        assert audit["summary"]["memory"]["initialRecall"]["status"] == "completed"
        assert audit["summary"]["tools"][0]["name"] == "memory_reflect"
        assert received["session_queries"] == [{"limit": "20", "q": "deploy"}]
        assert received["audit_queries"] == [{"limit": "10", "sessionId": "session-1"}]
        assert PI_TOKEN not in json.dumps(
            {"sessions": sessions, "detail": detail, "audits": audits, "audit": audit}
        )
    finally:
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_history_proxy_rejects_malformed_identifiers_and_queries():
    dependency_runners, memory_url, pi_url, received = await dependencies()
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    try:
        async with aiohttp.ClientSession() as session:
            for path in (
                "/api/sessions?limit=0",
                "/api/sessions?limit=101",
                f"/api/sessions?q={'x' * 201}",
                "/api/sessions/%2e%2e%2fsecret",
                "/api/audits?sessionId=../../secret",
                "/api/audits/not-a-run-id",
            ):
                async with session.get(f"{playground_url}{path}") as response:
                    assert response.status in {400, 404}

        assert received["session_queries"] == []
        assert received["audit_queries"] == []
    finally:
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_playground_rejects_invalid_input_and_untrusted_hosts():
    dependency_runners, memory_url, pi_url, received = await dependencies()
    playground_runner, playground_url = await request_app(memory_url, pi_url)
    try:
        async with aiohttp.ClientSession() as session:
            for payload in (
                {"mode": "raw", "prompt": "hello"},
                {"mode": "agent", "prompt": ""},
                {
                    "mode": "agent",
                    "prompt": "hello",
                    "bankId": "../../internal",
                },
                {
                    "mode": "agent",
                    "prompt": "hello",
                    "sessionId": "session-only",
                    "parentEntryId": None,
                },
            ):
                async with session.post(
                    f"{playground_url}/api/runs", json=payload
                ) as response:
                    assert response.status == 400
                    body = await response.json()
                    assert body["error"]["code"] == "INVALID_REQUEST"

            for host in (
                "localhost",
                "playground.sidekick.localhost",
                "playground.sidekick.localhost:18865",
            ):
                async with session.get(
                    f"{playground_url}/", headers={"Host": host}
                ) as response:
                    assert response.status == 200

            for host in (
                "example.com",
                "localhost.example.com",
                "playground.sidekick.localhost.example.com",
                "-invalid.localhost",
                "invalid-.localhost",
            ):
                async with session.get(
                    f"{playground_url}/", headers={"Host": host}
                ) as response:
                    assert response.status == 400

            async with session.get(f"{playground_url}/") as response:
                markup = await response.text()
                assert response.status == 200
                assert "img-src 'self' data:" in response.headers[
                    "Content-Security-Policy"
                ]
                assert '<section id="audit-summary"' in markup
                assert 'aria-live="polite"' in markup

            async with session.get(f"{playground_url}/app.js") as response:
                script = await response.text()
                assert "innerHTML" not in script
                assert "textContent" in script
                assert 'if (event.type === "run_started") return;' in script
                assert 'if (event.type === "memory_snapshot")' in script
                assert 'if (event.type === "attachment")' in script
                assert (
                    "state.sessionId = event.sessionId || state.sessionId" not in script
                )
                assert "elements.newChat.disabled = running" in script
                assert 'event.type === "memory.access.warning"' in script
                assert "content omitted" in script
                assert "function renderAuditDiagnosis" in script
                assert "Decision trail" in script
                assert "Inspect event metadata" in script
                assert "current_bank_only" in script

            async with session.get(f"{playground_url}/styles.css") as response:
                styles = await response.text()
                assert response.status == 200
                assert ".trace-facts" in styles
                assert ".trace-step" in styles
                assert ".generated-attachment" in styles
    finally:
        assert received["runs"] == []
        await playground_runner.cleanup()
        for runner in dependency_runners:
            await runner.cleanup()


@pytest.mark.parametrize(
    "event",
    (
        {},
        {
            "type": "memory_snapshot",
            "primaryBankId": "chat:engineering",
            "queries": ["Who owns deploys?"],
            "memories": [
                {
                    "id": "memory-1",
                    "text": "Alice maintains the deployment pipeline.",
                    "entities": ["Alice"],
                    "documentId": "not a valid source id",
                }
            ],
        },
    ),
)
def test_playground_rejects_malformed_pi_memory_events(event):
    with pytest.raises(UpstreamUnavailable, match="malformed events"):
        _parse_pi_event(json.dumps(event).encode())


@pytest.mark.parametrize(
    "event",
    (
        {
            "type": "attachment",
            "filename": "generated-image.png",
            "mimeType": "image/png",
            "displayAs": "image",
            "data": "not-base64",
        },
        {
            "type": "attachment",
            "filename": "../generated-image.png",
            "mimeType": "image/png",
            "displayAs": "image",
            "data": PNG_DATA,
        },
        {
            "type": "attachment",
            "filename": "generated-image.png",
            "mimeType": "text/html",
            "displayAs": "image",
            "data": PNG_DATA,
        },
    ),
)
def test_playground_rejects_malformed_pi_attachments(event):
    with pytest.raises(UpstreamUnavailable, match="malformed events"):
        _parse_pi_event(json.dumps(event).encode())


def test_playground_accepts_wechat_memory_snapshot_source_ids():
    bank_id = "wechat:account:wxid_v11uy95lmdjh22:chat:49277108357%40chatroom"
    document_id = (
        "wechat:memory-session:49277108357@chatroom:20260731T103229Z:670640216917235054"
    )
    chunk_id = f"{bank_id}_{document_id}_0"
    event = {
        "type": "memory_snapshot",
        "primaryBankId": bank_id,
        "queries": ["What happened today?"],
        "memories": [
            {
                "id": "d94960ea-e1a2-4739-8435-4fca079e8870",
                "text": "A remembered fact.",
                "entities": [],
                "documentId": document_id,
                "chunkId": chunk_id,
            }
        ],
    }

    parsed = _parse_pi_event(json.dumps(event).encode())

    assert parsed["memories"][0]["documentId"] == document_id
    assert parsed["memories"][0]["chunkId"] == chunk_id


def test_run_audit_requires_a_bounded_diagnostic_summary():
    run_id = "11111111-1111-4111-8111-111111111111"
    for summary in (
        None,
        {"status": "mysterious"},
        {
            "status": "completed",
            "startedAt": "2026-07-13T12:00:00.000Z",
            "finishedAt": "2026-07-13T12:00:01.000Z",
            "durationMs": 1_000,
            "prompt": "hello",
            "eventCount": 0,
            "session": {
                "kind": "root",
                "id": None,
                "parentEntryId": None,
                "entryId": None,
            },
            "model": None,
            "memory": {
                "primaryBankId": None,
                "route": "read_every_bank",
                "initialRecall": None,
                "directory": None,
            },
            "tools": [],
            "warnings": [],
            "failure": None,
        },
    ):
        with pytest.raises(UpstreamUnavailable, match="malformed audit summary"):
            _parse_run_audit(
                {"runId": run_id, "summary": summary, "events": []},
                run_id,
            )
