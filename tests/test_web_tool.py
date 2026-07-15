"""Web tool: URL-unwrap, tag-strip, and the /web trigger parse. No network."""

from companion.tools.web import _detag, _unwrap
from companion.tui import _split_web_trigger


def test_unwrap_ddg_redirect():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX&rut=abc"
    assert _unwrap(wrapped) == "https://en.wikipedia.org/wiki/X"
    # a bare protocol-relative link gets https:
    assert _unwrap("//example.com/p").startswith("https://example.com")
    # a normal absolute url is untouched
    assert _unwrap("https://example.com/p") == "https://example.com/p"


def test_detag():
    assert _detag("<b>hi</b> &amp; bye") == "hi & bye"


def test_web_trigger():
    assert _split_web_trigger("latest news /web") == ("latest news", True)
    assert _split_web_trigger("price of gold /search") == ("price of gold", True)
    assert _split_web_trigger("just talking") == ("just talking", False)
    # a path that merely contains a slash must NOT trigger
    assert _split_web_trigger("cat /etc/hosts") == ("cat /etc/hosts", False)


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    print("ok — unwrap, detag, /web trigger")
