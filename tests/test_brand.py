from agent import brand


def test_brand_basics():
    assert brand.NAME == "Nightage" and brand.CMD == "nightage"
    assert brand.wordmark() == "nightage" and "\n" not in brand.wordmark()
    assert set(brand.LEGACY_CMDS) == {"agent", "lantern"}
    assert brand.REPO_URL.startswith("https://") and brand.ACCENT and brand.TAGLINE


def test_pyproject_scripts_and_dep():
    import pathlib
    import re

    txt = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert 'nightage = "agent.main:app"' in txt
    for c in brand.LEGACY_CMDS:
        assert re.search(rf'^{c} = "agent.main:app"', txt, re.M)
    assert "prompt_toolkit>=3.0.43" in txt
