"""The smallest plugin: one tool the model can call."""


def register(api):
    api.tool(
        "say_hello",
        "Greet someone by name. Use when the user asks you to say hello to somebody.",
        {"type": "object", "properties": {"name": {"type": "string", "description": "who to greet"}}, "required": ["name"]},
        lambda name: f"Hello, {name}!",
        risk="safe",            # it only builds a string, so nothing needs asking
    )
