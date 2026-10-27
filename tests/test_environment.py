from __future__ import annotations

from agent.memory.environment import system_facts
from agent.memory.recall import _build_system_prompt


def test_system_facts_includes_os_and_home():
    lines = system_facts({})
    joined = "\n".join(lines)
    assert any(line.startswith("OS:") for line in lines)
    assert any("Home directory:" in line for line in lines)
    assert "Agent project directory:" in joined


def test_environment_block_in_system_prompt():
    prompt = _build_system_prompt(
        "persona", facts=["likes trains"], memories=[], tool_names=None,
        environment=["OS: Ubuntu", "Home directory: /home/user"],
    )
    assert "System environment" in prompt
    assert "/home/user" in prompt
    assert "likes trains" in prompt
