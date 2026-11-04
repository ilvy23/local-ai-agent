from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_REAL_CONFIG = pytest.StashKey[tuple]()


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def pytest_sessionstart(session):
    """Remember the developer's real config.yaml, so a test that writes it fails the run (see pytest_sessionfinish)."""
    from agent import config as cfgmod

    path = Path(cfgmod.DEFAULT_CONFIG_PATH)
    session.config.stash[_REAL_CONFIG] = (path, _digest(path))


def pytest_sessionfinish(session, exitstatus):
    try:
        path, before = session.config.stash[_REAL_CONFIG]
    except KeyError:
        return
    if _digest(path) != before:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(f"a test changed the real {path} - tests must never write the user's config", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def _isolated_config_file(monkeypatch, tmp_path_factory):
    """Every config writer (settings, /model, modelpick, the web routes, coding/models, WhatsApp toggles) targets a
    throw-away config.yaml: the module constant `DEFAULT_CONFIG_PATH` is redirected in agent.config and in every loaded
    agent module holding its own copy of it, and `config.LAST_LOAD` starts fresh. A test that wants a particular file
    still patches the path itself, as before."""
    from agent import config as cfgmod

    real = cfgmod.DEFAULT_CONFIG_PATH
    fake = tmp_path_factory.mktemp("nightage-cfg") / "config.yaml"      # its own folder: tests that list tmp_path must not see it
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "agent" or name.startswith("agent.")):
            continue
        try:
            current = getattr(mod, "DEFAULT_CONFIG_PATH", None)
            # the real path, or an earlier test's redirect that a module imported DURING that test copied for good
            same = current == real or (isinstance(current, Path) and current.parent.name.startswith("nightage-cfg"))
        except Exception:  # noqa: BLE001 - an odd module attribute must not break every test
            same = False
        if same:
            monkeypatch.setattr(mod, "DEFAULT_CONFIG_PATH", fake)
    monkeypatch.setattr(cfgmod, "LAST_LOAD", {"path": "", "user_models": [], "profile": "default", "existed": False})
    yield fake


@pytest.fixture(autouse=True)
def _fresh_model_protection():
    """agent.llm's protected models (the chat / code model of a session) are process-wide: every test starts without any."""
    from agent import llm
    from agent.memory import recall

    llm.reset_protection()
    llm._EMBED_CACHE.clear()
    recall.reset_heads()
    yield
    llm.reset_protection()
    llm._EMBED_CACHE.clear()
    recall.reset_heads()


def pytest_collection_modifyitems(config, items):
    """Tests marked `posix_only` need bash / a pty / fcntl / /proc and are skipped on Windows (the CI matrix runs a subset there)."""
    if sys.platform != "win32":
        return
    skip = pytest.mark.skip(reason="needs a POSIX system")
    for item in items:
        if "posix_only" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_ollama_session(monkeypatch):
    """Entry points wrap their run in agent.ollama_local.session (autostart + exit cleanup + signal handlers): never in tests."""
    monkeypatch.setenv("NIGHTAGE_OLLAMA_SESSION", "off")
    monkeypatch.setenv("NIGHTAGE_NO_AUTOSTART", "1")


@pytest.fixture(autouse=True)
def _no_embedding_backoff(monkeypatch):
    """Don't make the suite wait out the embedding retry's backoff.

    `embed_with_retry` sleeps between attempts so a busy GPU has a moment to free
    up — real and wanted in production, pure dead time in tests (it took the suite
    from 2s to 96s). The retry logic itself still runs; only the waiting is gone.
    """
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _isolated_gpu_choice(monkeypatch, tmp_path):
    """Tests never see the developer's real GPU choice and never start a real server.

    Without this, a CLI test runs the start-of-run GPU question non-interactively, which
    reuses the *real* last answer from ~/.local/state/agent/gpu_mode (e.g. "dual") for the
    rest of the process - and later tests would then try to start the real Vulkan Ollama
    through `ollama-vk` (a 90 s wait, and a real server on a real GPU).

    Also neutralises real GPU detection for the no-args production call (`detect_gpus()`,
    from `auto_mode`/`sysinfo`/`ctxbudget`): on a machine with ≥2 real Vulkan-capable GPUs it
    would otherwise find them for real and route *every* model to Vulkan regardless of
    AGENT_VULKAN_MODELS - exactly the "developer's real GPU choice" this fixture exists to
    hide. `detect_gpus(drm_root=...)` - how the dedicated detection tests call it - still
    runs for real against their fake sysfs; only a test that patches `detect_gpus` itself
    (to exercise the real auto_mode/current_mode chain against chosen fake GPUs) overrides
    this default, same as patching anything else in setup."""
    from agent import gpumode

    real_detect_gpus = gpumode.detect_gpus
    monkeypatch.setattr(gpumode, "detect_gpus",
                         lambda drm_root=None, **kw: real_detect_gpus(drm_root, **kw) if drm_root is not None else [])
    for var in ("AGENT_GPU_MODE", "AGENT_CHAT_CTX", "AGENT_SINGLE_KV", "AGENT_DUAL_KV",
                "AGENT_VULKAN_MODELS", "AGENT_VULKAN_OLLAMA_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_VULKAN_MODELS", r"qwen[^\s]*27b")      # tests keep the narrow routing; the real default (every model) is tested explicitly
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_VULKAN_START_CMD", "/bin/false")   # tests may never start a real server
    gpumode.reset()
    yield
    gpumode.reset()


@pytest.fixture(autouse=True)
def _isolated_sysinfo(monkeypatch):
    """`agent.sysinfo.current()` caches a real `detect()` for the process - so whichever test
    runs first (e.g. modelpick recommending a chat model from total VRAM) leaks this machine's
    real hardware (via the real detect_gpus() call `detect()` makes) into every test after it,
    unless a test seeds `sysinfo._cached` itself (see test_profiles.py) - which this must not
    override. Clearing the cache per test is enough: a fresh `detect()` still goes through
    `gpumode.detect_gpus()`, already neutralised to [] by default in `_isolated_gpu_choice`."""
    from agent import sysinfo

    monkeypatch.setattr(sysinfo, "_cached", None)


@pytest.fixture(autouse=True)
def _isolated_code_policy(monkeypatch, tmp_path):
    """No test ever reads or writes the developer's real ~/.config/agent/code-policy.json.

    Found live: the suite had added ~12,500 `/tmp/pytest-...` repo entries to the real file (every test that grants a mode
    persists it). The policy path honours AGENT_CODE_POLICY, so every test gets its own file."""
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "code-policy.json"))


@pytest.fixture(autouse=True)
def _research_stays_offline(monkeypatch, tmp_path):
    """No test reaches the real network through the research tools or the developer's own trust rules.

    `connectors.lookup` runs on every web_search ("what is X" asks Wikipedia), the search-engine variables
    would route a test to a real SearXNG/Brave, and ~/.config/agent/source-trust.yaml would change the ratings.
    A test that wants a connector or an engine patches it again inside the test."""
    def offline(*args, **kwargs):
        raise RuntimeError("network is switched off in tests")

    monkeypatch.setattr("agent.research.connectors._get_json", offline)
    monkeypatch.setattr("agent.research.connectors._get_text", offline)
    for var in ("AGENT_SEARXNG_URL", "BRAVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.research.trust.RULES_PATH", tmp_path / "no-source-trust.yaml")


@pytest.fixture(autouse=True)
def _no_real_plugins(monkeypatch, tmp_path):
    """Tests never load the developer's real plugins (the WhatsApp plugin reads private message data)."""
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path / "no-plugins"))


@pytest.fixture(autouse=True)
def _fresh_notices():
    """Once-per-session notices (agent.friendly.once) start fresh in every test; the model-list cache too."""
    from agent import friendly, modelpick

    friendly.forget_notices()
    modelpick.forget_cache()
    yield
