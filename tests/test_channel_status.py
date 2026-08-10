from __future__ import annotations

import time

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from sidekick.channel_status import (
    ActiveAIRun,
    AdapterRuntimeState,
    CachedChannelInventory,
    ChannelInventoryItem,
    ChannelOpsServer,
    ChannelOpsSettings,
    ChannelSnapshotService,
    AgentRunOrigin,
    StoredChannelState,
)


class StateReader:
    def __init__(self, *states: StoredChannelState):
        self.states = states

    async def list_channel_operational_states(
        self,
    ) -> tuple[StoredChannelState, ...]:
        return self.states


def test_ops_settings_default_to_loopback() -> None:
    settings = ChannelOpsSettings.from_env(
        default_instance_id="telegram-default",
        environ={"SIDEKICK_OPS_TOKEN": "channel-ops-token-that-is-long-enough"},
    )

    assert settings.host == "127.0.0.1"
    assert settings.port == 8781


def test_adapter_instance_ids_follow_the_128_character_pi_contract() -> None:
    valid = "a" * 128

    assert ChannelOpsSettings.from_env(
        default_instance_id="default",
        environ={
            "SIDEKICK_ADAPTER_INSTANCE_ID": valid,
            "SIDEKICK_OPS_TOKEN": "channel-ops-token-that-is-long-enough",
        },
    ).instance_id == valid
    assert AgentRunOrigin("qq:group:700", valid).adapter_instance_id == valid

    with pytest.raises(ValueError, match="1 to 128"):
        ChannelOpsSettings.from_env(
            default_instance_id="default",
            environ={
                "SIDEKICK_ADAPTER_INSTANCE_ID": "a" * 129,
                "SIDEKICK_OPS_TOKEN": "channel-ops-token-that-is-long-enough",
            },
        )
    with pytest.raises(ValueError, match="adapter instance"):
        AgentRunOrigin("qq:group:700", "a" * 129)


@pytest.mark.asyncio
async def test_snapshot_merges_live_inventory_with_nested_operational_state() -> None:
    scope_id = "wechat:account:wxid%40example.com:chat:group%40chatroom"
    state = StoredChannelState(
        scope_id=scope_id,
        access_open=True,
        model_override="openai/gpt-5",
        continuous_enabled=True,
        continuous_cursor_message_id="msg-42",
        continuous_scanned_until_at=1_800_000_005,
        continuous_last_attempt_at=1_800_000_010,
        continuous_last_success_at=1_800_000_000,
        continuous_last_error="token=should-not-leak upstream failed",
        retained_document_count=4,
        pending_count=1,
        retrying_count=1,
        dead_letter_count=1,
        next_retry_at=1_800_000_040,
        outbox_last_error="authorization=should-not-leak retain failed",
        outbox_last_error_at=1_800_000_015,
        outbox_last_dead_lettered_at=1_800_000_015,
        last_ingested_at=1_800_000_000,
        last_retained_source_at=1_799_999_900,
        last_retained_at=1_800_000_002,
        active_runs=(
            ActiveAIRun(
                run_id="run-1",
                status="RUNNING",
                session_id="session-1",
                started_at=1_800_000_020,
                updated_at=1_800_000_030,
            ),
        ),
        updated_at=1_800_000_030,
    )
    adapter = AdapterRuntimeState(
        id="wechat-peer",
        platform="wechat",
        account_id="wxid@example.com",
        connected=True,
        observed_at=1_800_000_000,
        indeterminate_outbound_probe=lambda: 3,
    )

    async def inventory() -> tuple[ChannelInventoryItem, ...]:
        return (
            ChannelInventoryItem(
                scope_id=scope_id,
                display_name="Example group",
                chat_kind="GROUP",
                last_observed_at=1_800_000_025,
            ),
        )

    before = time.time()
    snapshot = await ChannelSnapshotService(
        state_reader=StateReader(state),
        inventory_loader=inventory,
        adapter=adapter,
        memory_available=True,
    ).snapshot()

    assert snapshot["adapter"]["accountId"] == "wxid@example.com"
    assert snapshot["adapter"]["indeterminateOutboundCount"] == 3
    assert adapter.observed_at is not None and adapter.observed_at >= before
    assert len(snapshot["items"]) == 1
    row = snapshot["items"][0]
    assert row["scopeId"] == scope_id
    assert row["accessMode"] == "OPEN"
    assert row["modelOverride"] == "openai/gpt-5"
    assert row["lastObservedAt"] == "2027-01-15T08:00:25Z"
    assert row["updatedAt"] == "2027-01-15T08:00:30Z"
    assert row["memory"] == {
        "continuousEnabled": True,
        "dreamEnabled": False,
        "effectiveMode": "CONTINUOUS",
        "continuousLastAttemptAt": "2027-01-15T08:00:10Z",
        "continuousLastSuccessAt": "2027-01-15T08:00:00Z",
        "continuousLastError": "token=[redacted] upstream failed",
        "dreamLastAttemptAt": None,
        "dreamLastSuccessAt": None,
        "dreamLastError": None,
        "pendingDocumentCount": 1,
        "retryingDocumentCount": 1,
        "deadLetterDocumentCount": 1,
        "nextRetryAt": "2027-01-15T08:00:40Z",
        "scanCursor": "msg-42",
        "scanWatermarkAt": "2027-01-15T08:00:05Z",
        "retainWatermarkAt": "2027-01-15T08:00:02Z",
        "retainedSourceAt": "2027-01-15T07:58:20Z",
        "lastIngestedAt": "2027-01-15T08:00:00Z",
        "hindsightBankId": scope_id,
        "factCount": None,
        "retainedDocumentCount": 4,
    }
    assert row["activeRuns"][0]["scopeId"] == scope_id
    assert row["errors"] == [
        {
            "component": "CONTINUOUS_MEMORY",
            "code": "MEMORY_INGESTION_ERROR",
            "message": "token=[redacted] upstream failed",
            "occurredAt": "2027-01-15T08:00:10Z",
            "runId": None,
        },
        {
            "component": "MEMORY_OUTBOX",
            "code": "MEMORY_DELIVERY_DEAD_LETTER",
            "message": "authorization=[redacted] retain failed",
            "occurredAt": "2027-01-15T08:00:15Z",
            "runId": None,
        },
    ]


@pytest.mark.asyncio
async def test_ops_routes_require_the_dedicated_channel_bearer_token() -> None:
    async def inventory() -> tuple[ChannelInventoryItem, ...]:
        return ()

    adapter = AdapterRuntimeState(
        id="qq-default",
        platform="qq",
        account_id="42",
        connected=True,
    )
    ops = ChannelOpsServer(
        snapshot_service=ChannelSnapshotService(
            state_reader=StateReader(),
            inventory_loader=inventory,
            adapter=adapter,
            memory_available=True,
        ),
        settings=ChannelOpsSettings(
            instance_id="qq-default",
            token="channel-ops-token-that-is-long-enough",
        ),
    )
    server = TestServer(ops.application)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as client:
            unauthorized = await client.get(server.make_url("/v1/channels"))
            assert unauthorized.status == 401
            assert unauthorized.headers["Cache-Control"] == "no-store"
            health_unauthorized = await client.get(server.make_url("/health"))
            assert health_unauthorized.status == 401

            authorized = await client.get(
                server.make_url("/v1/channels"),
                headers={
                    "Authorization": "Bearer channel-ops-token-that-is-long-enough"
                },
            )
            assert authorized.status == 200
            payload = await authorized.json()
            observed_at = payload["adapter"].pop("observedAt")
            assert isinstance(observed_at, str) and observed_at.endswith("Z")
            assert payload == {
                "adapter": {
                    "id": "qq-default",
                    "platform": "qq",
                    "accountId": "42",
                    "connected": True,
                    "indeterminateOutboundCount": None,
                },
                "items": [],
            }
            health = await client.get(
                server.make_url("/health"),
                headers={
                    "Authorization": "Bearer channel-ops-token-that-is-long-enough"
                },
            )
            assert health.status == 200
            assert (await health.json())["ok"] is True
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_cached_inventory_avoids_reloading_on_every_dashboard_poll() -> None:
    now = [100.0]
    calls = 0

    async def load() -> tuple[ChannelInventoryItem, ...]:
        nonlocal calls
        calls += 1
        return (
            ChannelInventoryItem(
                scope_id="telegram:chat:1",
                display_name="Saved Messages",
                chat_kind="DIRECT",
            ),
        )

    cached = CachedChannelInventory(
        load,
        ttl_seconds=30,
        clock=lambda: now[0],
    )

    assert await cached.list_channels() == await cached.list_channels()
    assert calls == 1
    now[0] = 131.0
    await cached.list_channels()
    assert calls == 2


@pytest.mark.asyncio
async def test_cached_inventory_backs_off_after_a_refresh_failure() -> None:
    now = [100.0]
    calls = 0

    async def load() -> tuple[ChannelInventoryItem, ...]:
        nonlocal calls
        calls += 1
        raise RuntimeError("directory offline")

    cached = CachedChannelInventory(load, ttl_seconds=30, clock=lambda: now[0])

    with pytest.raises(RuntimeError, match="directory offline"):
        await cached.list_channels()
    now[0] = 101.0
    with pytest.raises(RuntimeError, match="backoff"):
        await cached.list_channels()
    assert calls == 1

    now[0] = 131.0
    with pytest.raises(RuntimeError, match="directory offline"):
        await cached.list_channels()
    assert calls == 2


@pytest.mark.asyncio
async def test_snapshot_retains_last_good_inventory_after_transient_failure() -> None:
    attempts = 0

    async def inventory() -> tuple[ChannelInventoryItem, ...]:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("temporary directory failure")
        return (
            ChannelInventoryItem(
                scope_id="qq:group:700",
                display_name="Example group",
                chat_kind="GROUP",
                last_observed_at=1_800_000_000,
            ),
        )

    service = ChannelSnapshotService(
        state_reader=StateReader(),
        inventory_loader=inventory,
        adapter=AdapterRuntimeState(
            id="qq-default",
            platform="qq",
            account_id="42",
            connected=True,
        ),
        memory_available=True,
    )

    first = await service.snapshot()
    degraded = await service.snapshot()

    assert [row["scopeId"] for row in first["items"]] == ["qq:group:700"]
    assert [row["scopeId"] for row in degraded["items"]] == ["qq:group:700"]
    assert degraded["adapter"]["error"] == (
        "Channel inventory is temporarily unavailable."
    )


@pytest.mark.asyncio
async def test_unknown_legacy_row_timestamp_stays_null_across_snapshots() -> None:
    async def inventory() -> tuple[ChannelInventoryItem, ...]:
        return ()

    service = ChannelSnapshotService(
        state_reader=StateReader(StoredChannelState(scope_id="telegram:chat:1")),
        inventory_loader=inventory,
        adapter=AdapterRuntimeState(
            id="telegram-default",
            platform="telegram",
            account_id="1",
            connected=True,
        ),
        memory_available=True,
    )

    first = await service.snapshot()
    second = await service.snapshot()

    assert first["items"][0]["updatedAt"] is None
    assert second["items"][0]["updatedAt"] is None
