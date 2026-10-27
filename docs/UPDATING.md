# Updating nightage and its plugins

```
nightage update --check      # look, print what is new, change nothing
nightage update              # show the changelog and the exact commands, ask, update, then "restart nightage"
nightage update --yes        # same without the question (scripts)
nightage plugin update --check | <name> | --all [--yes]
```

## What is checked, and when

* **The framework:** GitHub's public releases API for the project repository (`update.repo: owner/name` in config.yaml, else the
  checkout's `origin`, else the project page in `agent/brand.py`; if none is known, nothing is checked). A newer release (semantic
  version compare, pre-releases ordered correctly, never from an unparseable tag) is an update. For a git checkout the branch is also
  compared with its remote using `git ls-remote`, which changes nothing on disk.
* **Plugins:** every plugin installed from a git URL (`nightage plugin install <url>`) has an install record `.source.json` (url, ref,
  commit, time); the remote is compared with `git ls-remote`. Plugins installed from a local path say "no update source".
  A plugin may name `repository` / `update_url` in its `plugin.toml`.
* **When:** at most once every 24 hours (`update.check_interval_hours`), in a background thread, with a 3 s HTTP timeout. Startup never waits.
  Offline or failing checks are silent and retried after an hour. The result is cached in `~/.local/state/nightage/update-check.json`.
* **What is sent:** one plain GET per request to the releases API (and a git ls-remote per plugin). No identifier, no version, no cookies;
  GitHub sees your IP address like for any web request. The only header besides HTTP's own is a fixed `User-Agent: nightage-update-check`
  that GitHub's API requires.
* **Opt out:** `update: {check: false}` in config.yaml, or `NIGHTAGE_NO_UPDATE_CHECK=1`. Then nothing runs by itself; the commands above still work when you type them.

## Where you see it

One calm line in the home screen status block, in `nightage doctor` (group "update") and under `nightage --version`:
`Update available: 0.2.0 (you have 0.1.0) - run nightage update`, and `N plugin updates - nightage plugin update`. It disappears as soon as
you have updated. Nothing is shown in chat or code sessions.

## How each install kind updates

Everything runs as you. Never sudo. The commands are printed before you are asked.

| install | `nightage update` runs |
|---|---|
| git checkout (with remote) | `git fetch`, then `git pull --ff-only <remote> <branch>`; if `pyproject.toml` / `uv.lock` changed: `uv sync` (or `pip install -e .` for an editable install; otherwise a note). **Refused** (nothing changed, with the reason): uncommitted changes to tracked files, diverged history, detached HEAD, no remote, offline. |
| `uv tool` | `uv tool upgrade agent` |
| pip | `python -m pip install --upgrade git+<repo>@<release tag>` |
| editable copy without git / unknown | explains what to do by hand; there is nothing safe to automate |

Afterwards a small doctor subset (python, packages, config, database) runs in a fresh process with the new code, and you are told to
restart nightage. If a step fails it stops and leaves the install as it was.

## Plugin updates and the trust model

A plugin is ordinary code that runs with your permissions, so an update is new code you have not looked at. `nightage plugin update <name>`:

1. clones the source into a temporary folder (your plugin folder is not touched),
2. shows the changelog (`git log --oneline old..new`) and a **file-level summary** (added / removed / changed files with line counts),
3. asks. On yes the old folder moves to `<plugin folder>/.backups/` and the new files take its place,
4. switches the plugin **off**. If it was enabled before, the normal `plugin enable` question follows: new fingerprint, the full warning, a
   confirmation (default no). Decline and it stays off. `--yes` skips only the update question, **never** the enable question: the plugin
   stays off and you are told to run `nightage plugin enable <name>` after reading the code.

Refused: a source that is no longer a plugin, or that now has a different plugin name. The same holds for `plugin install <url> --force`:
replacing an enabled plugin switches it off.
