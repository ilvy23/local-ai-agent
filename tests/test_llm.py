import json

import httpx
import pytest

from companion.llm import OllamaClient


def _ndjson_response(request: httpx.Request) -> httpx.Response:
    chunks = [
        '{"message": {"role": "assistant", "content": "Hel"}, "done": false}',
        '{"message": {"role": "assistant", "content": "lo!"}, "done": false}',
        '{"message": {"role": "assistant", "content": ""}, "done": true}',
    ]
    body = "\n".join(chunks) + "\n"
    return httpx.Response(200, content=body, request=request)


def test_chat_streams_content_tokens_in_order():
    transport = httpx.MockTransport(_ndjson_response)
    client = OllamaClient(transport=transport)

    tokens = list(client.chat(messages=[{"role": "user", "content": "hi"}], model="mixtral:8x7b"))

    assert tokens == ["Hel", "lo!"]


def test_chat_sends_expected_request_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = httpx.Request(request.method, request.url, content=request.content).content
        return _ndjson_response(request)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    list(client.chat(messages=messages, model="mixtral:8x7b"))

    import json

    payload = json.loads(captured["json"])
    assert payload["model"] == "mixtral:8x7b"
    assert payload["messages"] == messages
    assert payload["stream"] is True


def test_chat_raises_connection_error_when_ollama_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    with pytest.raises(httpx.ConnectError):
        list(client.chat(messages=[{"role": "user", "content": "hi"}], model="mixtral:8x7b"))


def test_embed_returns_one_vector_per_input():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))

    vectors = client.embed(["a", "b"], model="nomic-embed-text")

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_truncates_oversized_input_to_char_limit():
    from companion.llm import EMBED_MAX_CHARS

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.content
        return httpx.Response(200, json={"embeddings": [[1.0]]}, request=request)

    client = OllamaClient(transport=httpx.MockTransport(handler))
    client.embed(["x" * (EMBED_MAX_CHARS + 5000)], model="nomic-embed-text")

    import json

    sent = json.loads(captured["json"])["input"][0]
    assert len(sent) == EMBED_MAX_CHARS


def test_embed_sends_expected_request_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.content
        return httpx.Response(200, json={"embeddings": [[1.0]]}, request=request)

    client = OllamaClient(transport=httpx.MockTransport(handler))
    client.embed(["only"], model="nomic-embed-text")

    import json

    payload = json.loads(captured["json"])
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == ["only"]


def test_chat_with_tools_returns_message_with_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "current_time", "arguments": {}}}
                    ],
                },
                "done": True,
            },
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))

    message = client.chat_with_tools(
        messages=[{"role": "user", "content": "what time is it?"}],
        model="mixtral:8x7b",
        tools=[{"type": "function", "function": {"name": "current_time"}}],
    )

    assert message["tool_calls"] == [{"function": {"name": "current_time", "arguments": {}}}]


def test_chat_with_tools_returns_plain_text_message_when_no_tool_call():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello!"}, "done": True},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))

    message = client.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        model="mixtral:8x7b",
        tools=[],
    )

    assert message["content"] == "Hello!"
    assert message.get("tool_calls") is None


def test_chat_with_tools_sends_non_streaming_request_with_tools_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.content
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}, "done": True},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    tools_payload = [{"type": "function", "function": {"name": "current_time"}}]
    client.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}], model="mixtral:8x7b", tools=tools_payload
    )

    import json

    payload = json.loads(captured["json"])
    assert payload["stream"] is False
    assert payload["tools"] == tools_payload


def test_chat_with_tools_retries_without_tools_when_model_rejects_them():
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests_seen.append(body)
        if "tools" in body:
            return httpx.Response(
                400,
                json={"error": "mixtral:8x7b does not support tools"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "plain text reply"}, "done": True},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    tools_payload = [{"type": "function", "function": {"name": "current_time"}}]

    message = client.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}], model="mixtral:8x7b", tools=tools_payload
    )

    assert message["content"] == "plain text reply"
    assert len(requests_seen) == 2
    assert "tools" in requests_seen[0]
    assert "tools" not in requests_seen[1]


def test_chat_with_tools_caches_unsupported_model_and_skips_tools_on_next_call():
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests_seen.append(body)
        if "tools" in body:
            return httpx.Response(
                400,
                json={"error": "mixtral:8x7b does not support tools"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "reply"}, "done": True},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    tools_payload = [{"type": "function", "function": {"name": "current_time"}}]

    client.chat_with_tools(
        messages=[{"role": "user", "content": "hi"}], model="mixtral:8x7b", tools=tools_payload
    )
    requests_seen.clear()

    client.chat_with_tools(
        messages=[{"role": "user", "content": "again"}], model="mixtral:8x7b", tools=tools_payload
    )

    assert len(requests_seen) == 1
    assert "tools" not in requests_seen[0]


def test_chat_with_tools_raises_other_http_status_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal server error"}, request=request)

    client = OllamaClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        client.chat_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            model="mixtral:8x7b",
            tools=[{"type": "function", "function": {"name": "current_time"}}],
        )
