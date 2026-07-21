import fire

from sidekick.config import apply_config
from sidekick.plugins import load_plugins
from sidekick.plugins.base import command_registry


def main() -> None:
    """Run Sidekick's chat-platform adapters and administrative commands."""
    load_plugins()
    apply_config()
    fire.Fire(command_registry.as_fire_commands())


if __name__ == "__main__":
    main()
