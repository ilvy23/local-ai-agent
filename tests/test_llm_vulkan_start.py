"""The Vulkan server start command: a persistent user unit wins over the wrapper script (found live: after `systemctl --user stop ollama-vk`
the wrapper's `systemd-run --unit=ollama-vk` failed with 'already loaded or has a fragment file' and the agent fell back to the default server)."""

from agent import llm


def test_a_persistent_unit_is_started_with_systemctl(monkeypatch):
    monkeypatch.delenv("AGENT_VULKAN_START_CMD", raising=False)
    monkeypatch.setattr(llm, "_vulkan_unit_exists", lambda: True)
    assert llm._vulkan_start_command() == ["systemctl", "--user", "start", "ollama-vk.service"]


def test_without_a_unit_the_wrapper_script_is_used(monkeypatch):
    monkeypatch.delenv("AGENT_VULKAN_START_CMD", raising=False)
    monkeypatch.setattr(llm, "_vulkan_unit_exists", lambda: False)
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/x/ollama-vk" if name == "ollama-vk" else None)
    assert llm._vulkan_start_command() == ["/x/ollama-vk", "list"]


def test_the_custom_command_still_wins(monkeypatch):
    monkeypatch.setenv("AGENT_VULKAN_START_CMD", "my-start --now")
    monkeypatch.setattr(llm, "_vulkan_unit_exists", lambda: True)
    assert llm._vulkan_start_command() == ["my-start", "--now"]


def test_unit_detection_survives_missing_systemctl(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    assert llm._vulkan_unit_exists() is False
