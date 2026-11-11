"""The night's view of the project map (agent/codeagent/projectmap.py does the work, for every coding run): the map lives in `.night/` so the model can add notes to it, and
`MAP_PROMPT` is the optional first model pass of a night that does exactly that."""
from __future__ import annotations

from pathlib import Path

from agent.codeagent.projectmap import CODE_EXT, NOTES_MARK, SKIP_DIRS, FileInfo, render, scan  # noqa: F401 - the old import names stay valid
from agent.codeagent import projectmap


def refresh(root: Path, budget: int = 8000) -> list[str]:
    """Refresh the map stored in <root>/.night/ and return the paths that changed."""
    (Path(root) / ".night").mkdir(exist_ok=True)
    return projectmap.refresh(root, Path(root) / ".night", budget).changed


MAP_PROMPT = ("Read .night/MAP.md (the harness wrote it from the code). Then, for the parts of this project that matter for the tasks of this night, read the code and append "
              "to the section '## Notes' at the end of .night/MAP.md (create the heading if it is missing; never edit the lines above it): how a feature is added here, "
              "where state is stored, what each tricky function really does, gotchas. Short, factual, no guessing; at most 6000 characters. Save after every 2-3 files you "
              "read. Do not change any project file other than .night/MAP.md.")
