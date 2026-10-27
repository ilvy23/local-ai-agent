"""Frontend smoke checks that need no browser: every module parses, imports resolve, CSS tokens exist, view data is well formed."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "agent" / "web" / "static"
JS = sorted((STATIC / "js").rglob("*.js"))
node = shutil.which("node")


def test_relative_imports_resolve():
    for f in JS:
        code = "\n".join(l for l in f.read_text().splitlines() if not l.lstrip().startswith("//"))
        for spec in re.findall(r'(?:from|import)\s*\(?\s*["\'](\.[^"\']+)["\']', code):
            assert (f.parent / spec).resolve().exists(), f"{f.relative_to(STATIC)} imports missing {spec}"


def test_every_css_variable_is_defined():
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", "".join(p.read_text() for p in (STATIC / "css").glob("*.css"))))
    # set inline by scripts or locally scoped inside components
    local = {"--pad", "--press-nx", "--press-ny", "--cmyk-num-ground"}
    used = set(re.findall(r"var\((--[a-z0-9-]+)", "".join(p.read_text() for p in list((STATIC / "css").glob("*.css")) + JS)))
    used = {u for u in used if not re.match(r"--(ok|info|warn|danger|block|neutral)$|--\$", u)}
    missing = {u for u in used - defined - local if not u.startswith("--${") and not u.endswith("-")}
    assert not missing, f"undefined CSS variables: {sorted(missing)}"


def test_info_colour_is_not_purple():
    css = (STATIC / "css" / "tokens.css").read_text()
    for m in re.finditer(r"--info:\s*(#[0-9a-fA-F]{6})", css):
        r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (1, 3, 5))
        assert g >= r and b >= r, f"--info {m.group(1)} reads purple/red"


@pytest.mark.skipif(node is None, reason="node not installed")
def test_js_syntax_and_view_data():
    for f in JS:
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.relative_to(STATIC)}: {r.stderr}"
    script = """
    import fs from 'node:fs';
    const dir = process.argv[1]; const out = {};
    for (const f of fs.readdirSync(dir).filter(n => n.endsWith('.js') && !n.startsWith('_'))) {
      const m = (await import(dir + '/' + f)).default; out[f] = { tabs: m.tabs.length, kinds: [...new Set(m.tabs.flatMap(t => t.s.map(s => s.k)))], keys: m.tabs.map(t => t.k) };
    }
    console.log(JSON.stringify(out));
    """
    r = subprocess.run([node, "--input-type=module", "-e", script, str(STATIC / "js" / "views-data")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    known = set(re.findall(r"^  (\w+)\((?:s|)[,)]", (STATIC / "js" / "viewkit.js").read_text(), re.M))
    for name, d in data.items():
        assert d["tabs"] and len(set(d["keys"])) == d["tabs"], name
        assert set(d["kinds"]) <= known, f"{name}: unknown section kinds {set(d['kinds']) - known}"
    assert {"sessions", "apps", "knowledge", "research", "tools", "ops"} <= {n[:-3] for n in data}


def test_every_routed_screen_has_a_module():
    main = (STATIC / "js" / "main.js").read_text()
    names = re.search(r"new Set\(\[(.*?)\]\)", main, re.S).group(1)
    for n in re.findall(r'"(\w+)"', names):
        assert (STATIC / "js" / "screens" / f"{n}.js").exists(), n
