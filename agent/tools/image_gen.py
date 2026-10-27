"""generate_image: runs the locally-installed HiDream-O1-Image model as a
genuinely backgrounded job - a single generation takes several minutes on
this hardware (CPU-offloaded, since the model doesn't fit the GPU's VRAM on
its own), far too long to block a chat turn on. The tool call itself
returns almost immediately with a job id and an ETA; the actual image
shows up in chat later, once the background thread finishes.

Delivery is done directly by the background thread (see _run_job), not by
the frontend's status poll: the poll route used to be the one creating the
file-share token and persisting the file_card row, on its first "done"
response - which meant an image only ever got delivered if some tab
happened to still be polling when it finished. A generation takes ~8
minutes; a backgrounded/reloaded/closed tab (all routine on a phone) meant
the token and DB row simply never got created, and the finished PNG sat
orphaned on disk forever with the model having already told the user
"it'll appear automatically". Writing the file_card row here instead means
delivery doesn't depend on a browser being present at all - a reload just
re-reads history like any other message.

Deliberately does NOT report live percent-complete during generation (the
same repeated-callback-into-asyncio pattern that made the app completely
unresponsive during a video transcode earlier tonight applies here too,
maybe worse - a 6+ minute generation calling back on every diffusion step
would be the same bug at a different call site). One notification when the
job starts, nothing in between - completion is a plain DB write, not an
SSE event.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_DATA_DIR
from agent.textio import human_size
from agent.tools.registry import Tool

IMAGE_GEN_REPO = Path.home() / "HiDream-O1-Image"
IMAGE_GEN_CHECKPOINT = IMAGE_GEN_REPO / "checkpoints"
IMAGE_GEN_PYTHON = Path("/usr/bin/python3")
IMAGE_OUTPUT_DIR = DEFAULT_DATA_DIR / "generated-images"
DURATION_HISTORY_FILE = DEFAULT_DATA_DIR / "image_gen_durations.json"
# The one real sample from tonight's first successful end-to-end run was
# 502s - close to but a bit over the original 400s guess, which was never
# actually measured. Median (not mean) below so one contended/slow-VRAM-
# wait outlier can't drag the estimate around once there's real history.
_DEFAULT_ETA_S = 500.0
_MAX_HISTORY = 10
_GEN_TIMEOUT_S = 1800
_VRAM_WAIT_TIMEOUT_S = 60.0

# job_id -> {"status": "running"|"done"|"failed", "prompt": str,
#            "started_at": float, "eta_s": float, "path": Path | None,
#            "error": str | None, "url"/"filename"/"size_str" once done}
# In-memory only: a job that was mid-flight when the process restarts is
# simply lost (the model won't know to report on it) - acceptable for a
# single-user local tool where a restart is a rare, deliberate action, not
# worth the complexity of persisting + resuming a live subprocess handle.
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_DURATION_LOCK = threading.Lock()

# This machine's 8GB card can't fit the image model alongside the chat
# model even sequentially-queued (both would try to claim VRAM at once) -
# only one generation at a time, full stop. A non-blocking acquire means a
# second request gets told to wait instead of silently OOMing both jobs.
_GEN_SEMAPHORE = threading.Semaphore(1)


def _load_duration_history() -> list[float]:
    try:
        return json.loads(DURATION_HISTORY_FILE.read_text())
    except (OSError, ValueError):
        return []


def _record_duration(seconds: float) -> None:
    with _DURATION_LOCK:
        history = _load_duration_history()
        history.append(seconds)
        history = history[-_MAX_HISTORY:]
        DURATION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        DURATION_HISTORY_FILE.write_text(json.dumps(history))


def estimate_eta_s() -> float:
    """The "algorithm" behind the ETA the model reports: the median of
    real recent generation times on THIS hardware, falling back to the
    one observed duration (~500s) until there's enough history to make an
    average worth trusting."""
    history = _load_duration_history()
    if not history:
        return _DEFAULT_ETA_S
    return statistics.median(history)


def get_job_status(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _wait_for_vram_free(client: Any, timeout_s: float = _VRAM_WAIT_TIMEOUT_S) -> bool:
    """Unloading from the tool handler isn't enough: the SAME chat turn
    keeps generating its closing narration ("Your image is generating...")
    right after the tool call returns, and that reloads the chat model
    into VRAM again before this background thread even gets going -
    proven by a real OOM that happened despite an unload_all() call
    earlier in the request. So the unload has to happen here, right
    before the subprocess actually needs the VRAM, and it has to
    confirm the model is really gone rather than trust one API call -
    Ollama's own /api/ps was seen lagging behind reality by more than
    just a couple seconds in testing.

    Checks `loaded_models(vram_only=True)`, not just loaded_models(): the
    embedding model runs with num_gpu:0 specifically so it never touches
    VRAM, but it still shows up in plain /api/ps, which previously made
    this wait uselessly unload/re-check a model that was never occupying
    the VRAM this is actually waiting on - and chat.py's own post-reply
    embedding step runs concurrently with this loop, so that false signal
    was near-guaranteed on every generation.

    Deadline-based, not attempt-count-based: Ollama's single generation
    slot (-np 1 on this box) means each unload call can itself block for
    seconds behind whatever's still finishing, so a fixed attempt count
    doesn't reliably map to a fixed time budget. Returns False (never
    raises) if VRAM still isn't free when the deadline passes - the
    caller decides what to do with that rather than launching into a
    near-certain OOM.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not client.loaded_models(vram_only=True):
            return True
        client.unload_all()
        time.sleep(1.0)
    return not client.loaded_models(vram_only=True)


def _deliver(store: Any, job_id: str, job: dict[str, Any]) -> None:
    """Persists the finished image as a real file-share + file_card
    message, the same durable delivery every other sent file gets -
    done here, unconditionally, the moment generation succeeds, so
    delivery never depends on a browser tab still being open/polling."""
    path = job["path"]
    filename = (job["prompt"][:40].strip() or "generated-image") + ".png"
    token = store.add_file_share(
        job["session_id"], str(path), disposition="inline", display_name=filename
    )
    size_str = human_size(path.stat().st_size)
    event = {
        "type": "file_ready", "kind": "image", "url": f"/api/files/{token}",
        "filename": filename, "size": size_str,
    }
    store.add_message(job["session_id"], "file_card", json.dumps(event))
    with _JOBS_LOCK:
        _JOBS[job_id]["url"] = event["url"]
        _JOBS[job_id]["filename"] = filename
        _JOBS[job_id]["size_str"] = size_str


def _fail(job_id: str, message: str) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "failed"
        _JOBS[job_id]["error"] = message[:800]


def _run_job(
    job_id: str, prompt: str, height: int, width: int, client: Any, store: Any
) -> None:
    # Every exit path below must release this - a single try/finally around
    # the whole body instead of a release() call scattered at each return,
    # so an exception nobody anticipated (a disk-full JSON write, a bug in
    # _deliver) can't leak the semaphore and permanently wedge "a generation
    # is already running" until the process is restarted.
    try:
        started = time.monotonic()
        output_path = IMAGE_OUTPUT_DIR / f"{job_id}.png"
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not _wait_for_vram_free(client):
            _fail(
                job_id,
                "Couldn't free enough VRAM to start generation (the chat "
                "model wouldn't unload in time) - try again in a moment.",
            )
            return

        try:
            subprocess.run(
                [
                    str(IMAGE_GEN_PYTHON), str(IMAGE_GEN_REPO / "inference.py"),
                    "--model_path", str(IMAGE_GEN_CHECKPOINT),
                    "--model_type", "full",
                    "--prompt", prompt,
                    "--output_image", str(output_path),
                    "--height", str(height),
                    "--width", str(width),
                ],
                cwd=str(IMAGE_GEN_REPO),
                capture_output=True, timeout=_GEN_TIMEOUT_S, check=True,
            )
            if not output_path.exists():
                raise RuntimeError("inference.py exited cleanly but wrote no output file")
        except subprocess.CalledProcessError as exc:
            # exc.stderr is the actual traceback (OOM, missing weights,
            # etc.) - discarding it in favor of exc's own str() (just
            # "returned non-zero exit status 1") was exactly the kind of
            # silent-failure pattern that hid the original VRAM race
            # condition tonight.
            stderr_tail = (exc.stderr or b"").decode(errors="replace")[-800:]
            _fail(job_id, stderr_tail or str(exc))
            return
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            _fail(job_id, str(exc))
            return

        elapsed = time.monotonic() - started
        try:
            _record_duration(elapsed)
        except OSError:
            pass  # ETA history is a nice-to-have, never worth failing a real generation over

        # Deliberately NOT marked "done" until _deliver actually succeeds:
        # a poller that sees "done" with no url would otherwise sit
        # showing nothing until its own give-up deadline, for an image
        # that generated fine but failed to hand off.
        try:
            _deliver(store, job_id, {"path": output_path, "prompt": prompt, "session_id": _JOBS[job_id]["session_id"]})
        except Exception as exc:  # noqa: BLE001 - genuinely any failure here must still fail the job, not leak the semaphore
            _fail(job_id, f"Generated the image but couldn't deliver it: {exc}")
            return
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["path"] = output_path
    except Exception as exc:  # noqa: BLE001 - last-resort net: anything truly
        # unanticipated (e.g. IMAGE_OUTPUT_DIR.mkdir() hitting a permissions
        # error, which sits outside every specific handler above) must still
        # mark the job failed - otherwise it's stuck at "running" forever
        # and the poller just sits there until its own give-up deadline
        # with a generic timeout message instead of the real cause.
        _fail(job_id, str(exc))
    finally:
        _GEN_SEMAPHORE.release()


def start_generation(
    prompt: str, session_id: int, client: Any, store: Any, height: int = 1024, width: int = 1024
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:16]
    eta_s = estimate_eta_s()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running", "prompt": prompt, "session_id": session_id,
            "started_at": time.time(), "eta_s": eta_s, "path": None, "error": None,
        }
    thread = threading.Thread(
        target=_run_job, args=(job_id, prompt, height, width, client, store), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "eta_s": eta_s}


def make_generate_image_tool(store: Any, session_id: int, console: Any, client: Any) -> Tool:
    def handler(prompt: str = "", **_kwargs: Any) -> str:
        if not prompt:
            return "Error: no prompt given."
        if not IMAGE_GEN_CHECKPOINT.is_dir():
            return (
                "Error: the image generation model isn't installed at "
                f"{IMAGE_GEN_CHECKPOINT} - it needs to be set up first."
            )
        if not _GEN_SEMAPHORE.acquire(blocking=False):
            return (
                "A generation is already running - this hardware can only run one "
                "at a time (the model doesn't fit VRAM alongside a second copy of "
                "itself). Tell the user to wait for the current one to finish "
                "before asking for another."
            )
        # The actual VRAM-freeing wait happens inside the background job
        # (_wait_for_vram_free), not here - this same chat turn still has
        # a closing reply left to generate after this handler returns,
        # which would just reload the model again if we unloaded now.
        try:
            result = start_generation(prompt, session_id, client, store)
        except Exception:
            # _run_job's own finally is what normally releases this - but
            # that only exists once the thread is actually running. If
            # thread.start() itself fails (OS resource exhaustion, however
            # rare), nothing downstream ever gets a chance to release the
            # semaphore, wedging every future generation - so this is the
            # one place that has to catch it instead.
            _GEN_SEMAPHORE.release()
            raise
        eta_min = round(result["eta_s"] / 60, 1)
        console.emit_event(
            "image_job", job_id=result["job_id"], eta_s=result["eta_s"], prompt=prompt,
        )
        return (
            f"Image generation started in the background (job {result['job_id']}), "
            f"estimated to take about {eta_min} minutes based on recent generations on "
            "this hardware. It will appear in the chat automatically the moment it's "
            "done - do NOT say the image is ready now, and do NOT make up a different "
            "time estimate; just tell the user generation has started and roughly how "
            "long it usually takes, in one short sentence. Also mention that replies "
            "will be slower than usual until it finishes, since this GPU can't run "
            "the chat model and image generation at full speed at the same time."
        )

    return Tool(
        name="generate_image",
        description=(
            "Generate an image locally from a text description, using the "
            "HiDream-O1-Image model installed on this machine - NOT a web "
            "search or an external API, a real local generation. Takes "
            "several minutes (this model doesn't fit entirely in VRAM on "
            "this hardware) and only one can run at a time, so this starts "
            "the job in the background and returns immediately with an "
            "estimated completion time - the actual image appears in the "
            "chat automatically once it's done, the same way a sent "
            "video/audio file does. Never claim the image is ready in "
            "your reply to this call; it isn't yet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A detailed description of the image to generate - the more "
                        "specific (style, subject, lighting, composition), the better "
                        "the result."
                    ),
                },
            },
            "required": ["prompt"],
        },
        handler=handler,
        risk="safe",
    )
