from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, TypeAlias


MAX_MEMORY_BACKFILL_DAYS = 30
MAX_MEMORY_BACKFILL_MESSAGES = 5_000
MAX_AI_COOLDOWN_SECONDS = 86_400
DEFAULT_AI_COMMAND_PREFIX = "/ai"
MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
AI_COMMAND_PREFIX_RE = re.compile(r"[^\w\s][A-Za-z][A-Za-z0-9_]{0,30}")


@dataclass(frozen=True, slots=True)
class AIAskCommand:
    prompt: str
    recent_messages: int | None = None


@dataclass(frozen=True, slots=True)
class AICancelCommand:
    pass


@dataclass(frozen=True, slots=True)
class AILimitCommand:
    action: Literal["show", "set", "reset"]
    cooldown_seconds: int | None = None

    def __post_init__(self) -> None:
        if (self.action == "set") != (self.cooldown_seconds is not None):
            raise ValueError("Only a limit-setting command accepts a cooldown")
        if self.cooldown_seconds is not None and not (
            0 <= self.cooldown_seconds <= MAX_AI_COOLDOWN_SECONDS
        ):
            raise ValueError("AI cooldown is outside supported bounds")


@dataclass(frozen=True, slots=True)
class AIModelCommand:
    action: Literal["show", "set", "reset"]
    model: str | None = None

    def __post_init__(self) -> None:
        if (self.action == "set") != (self.model is not None):
            raise ValueError("Only a model-selection command accepts a model")


@dataclass(frozen=True, slots=True)
class AIPrefixCommand:
    action: Literal["show", "set", "reset"]
    prefix: str | None = None

    def __post_init__(self) -> None:
        if (self.action == "set") != (self.prefix is not None):
            raise ValueError("Only a prefix-setting command accepts a prefix")
        if self.prefix is not None and normalize_ai_command_prefix(self.prefix) != (
            self.prefix
        ):
            raise ValueError("AI command prefixes must be normalized")


@dataclass(frozen=True, slots=True)
class AccessCommand:
    allowed: bool

    @property
    def name(self) -> str:
        return "/ai_allow" if self.allowed else "/ai_deny"


@dataclass(frozen=True, slots=True)
class ChatAccessCommand:
    action: Literal["open", "restricted", "status"]


@dataclass(frozen=True, slots=True)
class DirectoryPublishCommand:
    arguments: str


@dataclass(frozen=True, slots=True)
class BankGrantCommand:
    allowed: bool
    source: str

    @property
    def name(self) -> str:
        return "/ai_bank_allow" if self.allowed else "/ai_bank_deny"


@dataclass(frozen=True, slots=True)
class MemoryRememberCommand:
    instruction: str


@dataclass(frozen=True, slots=True)
class MemoryBackfillCommand:
    mode: Literal["days", "messages"]
    value: int

    def __post_init__(self) -> None:
        maximum = (
            MAX_MEMORY_BACKFILL_DAYS
            if self.mode == "days"
            else MAX_MEMORY_BACKFILL_MESSAGES
        )
        if self.mode not in {"days", "messages"} or not 1 <= self.value <= maximum:
            raise ValueError("Memory backfill request is outside supported bounds")


@dataclass(frozen=True, slots=True)
class MemoryModeCommand:
    mode: Literal["continuous", "dream"]
    enabled: bool
    target: str | None = None

    @property
    def name(self) -> str:
        prefix = "/ai_memory" if self.mode == "continuous" else "/ai_dream"
        suffix = "enable" if self.enabled else "disable"
        return f"{prefix}_{suffix}"


@dataclass(frozen=True, slots=True)
class MemoryStatusCommand:
    pass


@dataclass(frozen=True, slots=True)
class MemoryListCommand:
    pass


@dataclass(frozen=True, slots=True)
class MemoryDreamCommand:
    pass


@dataclass(frozen=True, slots=True)
class InvalidCommand:
    name: str


ChatCommand: TypeAlias = (
    AIAskCommand
    | AICancelCommand
    | AILimitCommand
    | AIModelCommand
    | AIPrefixCommand
    | AccessCommand
    | ChatAccessCommand
    | DirectoryPublishCommand
    | BankGrantCommand
    | MemoryRememberCommand
    | MemoryBackfillCommand
    | MemoryModeCommand
    | MemoryStatusCommand
    | MemoryListCommand
    | MemoryDreamCommand
    | InvalidCommand
)


def normalize_ai_command_prefix(prefix: str) -> str:
    if AI_COMMAND_PREFIX_RE.fullmatch(prefix) is None:
        raise ValueError(
            "AI command prefix must start with a punctuation character followed by letters"
        )
    normalized = prefix.casefold()
    if normalized.startswith("/ai_"):
        raise ValueError("AI command prefix conflicts with the control namespace")
    return normalized


def parse_chat_command(
    text: str | None,
    *,
    ai_prefix: str = DEFAULT_AI_COMMAND_PREFIX,
) -> ChatCommand | None:
    if text is None:
        return None
    ai_prefix = normalize_ai_command_prefix(ai_prefix)
    is_slash_command = text.startswith("/")
    is_ai_command = text.casefold().startswith(ai_prefix)
    
    if not is_slash_command and not is_ai_command:
        return None
        
    if is_slash_command:
        prefix = _parse_ai_prefix_control(text.strip())
        if prefix is not None:
            return prefix

        directory = _parse_directory_control(text)
        if directory is not None:
            return directory

        ai_limit = _parse_ai_limit_control(text.strip())
        if ai_limit is not None:
            return ai_limit

        model = _parse_model_control(text.strip())
        if model is not None:
            return model

        chat_access = _parse_chat_access_control(text.strip())
        if chat_access is not None:
            return chat_access

    ai = _parse_ai(text, ai_prefix)
    if ai is not None:
        return ai

    if is_slash_command:
        memory_revision = _parse_memory_revision(text)
        if memory_revision is not None:
            return memory_revision

        control = text.strip()
        if control == "/ai_cancel":
            return AICancelCommand()
        if control == "/ai_allow":
            return AccessCommand(allowed=True)
        if control == "/ai_deny":
            return AccessCommand(allowed=False)

        memory = _parse_memory_control(control)
        if memory is not None:
            return memory

    return None


def _parse_ai_prefix_control(text: str) -> AIPrefixCommand | InvalidCommand | None:
    parts = text.split()
    if not parts or parts[0] != "/ai_prefix":
        return None
    if len(parts) == 1:
        return AIPrefixCommand(action="show")
    if len(parts) != 2:
        return InvalidCommand(name="/ai_prefix")
    value = parts[1]
    if value.casefold() == "default":
        return AIPrefixCommand(action="reset")
    try:
        prefix = normalize_ai_command_prefix(value)
    except ValueError:
        return InvalidCommand(name="/ai_prefix")
    if prefix == DEFAULT_AI_COMMAND_PREFIX:
        return AIPrefixCommand(action="reset")
    return AIPrefixCommand(action="set", prefix=prefix)


def _parse_chat_access_control(
    text: str,
) -> ChatAccessCommand | InvalidCommand | None:
    parts = text.split()
    if not parts or parts[0] != "/ai_access":
        return None
    if len(parts) != 2 or parts[1] not in {"open", "restricted", "status"}:
        return InvalidCommand(name="/ai_access")
    return ChatAccessCommand(action=parts[1])


def _parse_ai_limit_control(text: str) -> AILimitCommand | InvalidCommand | None:
    parts = text.split()
    if not parts or parts[0] != "/ai_limit":
        return None
    if len(parts) == 1:
        return AILimitCommand(action="show")
    if len(parts) != 2:
        return InvalidCommand(name="/ai_limit")
    value = parts[1]
    if value.casefold() == "default":
        return AILimitCommand(action="reset")
    if not value.isascii() or not value.isdigit():
        return InvalidCommand(name="/ai_limit")
    cooldown_seconds = int(value)
    if not 0 <= cooldown_seconds <= MAX_AI_COOLDOWN_SECONDS:
        return InvalidCommand(name="/ai_limit")
    return AILimitCommand(action="set", cooldown_seconds=cooldown_seconds)


def _parse_model_control(text: str) -> AIModelCommand | InvalidCommand | None:
    parts = text.split()
    if not parts or parts[0] != "/ai_model":
        return None
    if len(parts) == 1:
        return AIModelCommand(action="show")
    if len(parts) != 2:
        return InvalidCommand(name="/ai_model")
    if parts[1].casefold() == "default":
        return AIModelCommand(action="reset")
    if MODEL_ID_RE.fullmatch(parts[1]) is None:
        return InvalidCommand(name="/ai_model")
    return AIModelCommand(action="set", model=parts[1])


def _parse_directory_control(text: str) -> ChatCommand | None:
    commands: tuple[tuple[str, type[DirectoryPublishCommand] | bool], ...] = (
        ("/ai_directory", DirectoryPublishCommand),
        ("/ai_bank_allow", True),
        ("/ai_bank_deny", False),
    )
    for name, command in commands:
        if text == name:
            arguments = ""
        elif text.startswith((f"{name} ", f"{name}\n", f"{name}\t")):
            arguments = text[len(name) :].strip()
        else:
            continue
        if command is DirectoryPublishCommand:
            return DirectoryPublishCommand(arguments=arguments)
        return BankGrantCommand(allowed=bool(command), source=arguments)
    return None


def _parse_ai(text: str, prefix: str) -> AIAskCommand | None:
    if not text.casefold().startswith(prefix):
        return None
    cursor = len(prefix)
    digit_start = cursor
    while cursor < len(text) and text[cursor].isascii() and text[cursor].isdigit():
        cursor += 1
    recent_messages = int(text[digit_start:cursor]) if cursor > digit_start else None
    if cursor < len(text) and text[cursor] == "@":
        cursor += 1
        mention_start = cursor
        while cursor < len(text) and text[cursor] not in " \n\t\r":
            cursor += 1
        if cursor == mention_start:
            return None
    if cursor == len(text):
        return AIAskCommand(prompt="", recent_messages=recent_messages)
    if text[cursor] not in " \n\t\r":
        return None
    return AIAskCommand(
        prompt=text[cursor:].strip(),
        recent_messages=recent_messages,
    )


def _parse_memory_revision(text: str) -> MemoryRememberCommand | None:
    if text == "/ai_memory":
        return MemoryRememberCommand(instruction="")
    if text.startswith(("/ai_memory ", "/ai_memory\n", "/ai_memory\t")):
        return MemoryRememberCommand(instruction=text[len("/ai_memory") :].strip())
    return None


def _parse_memory_control(text: str) -> ChatCommand | None:
    parts = text.split()
    if not parts:
        return None
    name = parts[0]
    if name == "/ai_memory_backfill":
        if len(parts) != 3 or parts[1] not in {"days", "messages"}:
            return InvalidCommand(name=name)
        try:
            value = int(parts[2])
        except ValueError:
            return InvalidCommand(name=name)
        maximum = (
            MAX_MEMORY_BACKFILL_DAYS
            if parts[1] == "days"
            else MAX_MEMORY_BACKFILL_MESSAGES
        )
        if not 1 <= value <= maximum:
            return InvalidCommand(name=name)
        return MemoryBackfillCommand(mode=parts[1], value=value)

    mode_commands = {
        "/ai_memory_enable": ("continuous", True),
        "/ai_memory_disable": ("continuous", False),
        "/ai_dream_enable": ("dream", True),
        "/ai_dream_disable": ("dream", False),
    }
    if name in mode_commands:
        if len(parts) > 2:
            return InvalidCommand(name=name)
        mode, enabled = mode_commands[name]
        return MemoryModeCommand(
            mode=mode,
            enabled=enabled,
            target=parts[1] if len(parts) == 2 else None,
        )
    if text == "/ai_memory_status":
        return MemoryStatusCommand()
    if text == "/ai_memory_list":
        return MemoryListCommand()
    if text == "/ai_memory_dream":
        return MemoryDreamCommand()
    return None
