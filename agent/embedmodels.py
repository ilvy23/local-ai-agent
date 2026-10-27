"""The embedding models Nightage knows: what each costs and who it is for.

The memory model ("embedder") turns text into vectors so a question finds the facts and chat lines that MEAN the same
thing. It runs on the CPU (llm.OllamaClient.embed passes num_gpu=0), so it competes for RAM, not VRAM; a smaller one
leaves more memory for the chat model.

Where the numbers come from (checked 2026-10-27, nothing here is from memory):
  * download size and context window: the tag pages on ollama.com/library/<name>/tags;
  * dimensions and RAM when loaded: measured on this project's own test machine (Ollama 0.40, CPU only) by embedding
    a few sentences and reading /api/ps; they vary a little with the Ollama version. `None` = not measured;
  * languages: the model's ollama.com page for granite-embedding (30m English only, 278m multilingual),
    embeddinggemma ("100+ spoken languages"), qwen3-embedding ("over 100 languages") and bge-m3 ("more than 100
    working languages"). all-minilm and nomic-embed-text are English models according to their upstream model cards
    (all-MiniLM, nomic-embed-text-v1.5); the ollama.com pages do not say so: marked `languages_source = "upstream"`.
  * quality was NOT benchmarked here. A 4-sentence probe in English, German and French (see docs/MEMORY.md) matched the
    labels but is no retrieval benchmark; the trade-off lines are guidance, not measurements.

Changing the embedder changes the vector size, so the stored index must be rebuilt: `nightage embed use <tag>` does it
(agent/memory/reembed.py). Never mix sizes in one index (VectorIndex.add refuses)."""

from __future__ import annotations

from dataclasses import dataclass, replace

DEFAULT_TAG = "bge-m3"
ENGLISH, MULTILINGUAL, UNKNOWN = "english", "multilingual", "unknown"


@dataclass(frozen=True)
class EmbedModel:
    tag: str                      # what `ollama pull` and models.embed take
    download_mb: int | None       # size on disk
    ram_mb: int | None            # resident memory when loaded (measured), None = not measured
    dim: int | None               # vector size, None = not measured
    languages: str                # english | multilingual | unknown
    context: int | None           # max input tokens
    label: str                    # one line a user reads
    tradeoff: str                 # one line: what you give up
    languages_source: str = "ollama"      # ollama = the ollama.com page says so; upstream = the model card does
    curated: bool = True          # offered by installers and `nightage embed list` by default
    exact: bool = True            # False: describe() matched the family, not this exact variant

    @property
    def multilingual(self) -> bool:
        return self.languages == MULTILINGUAL

    def line(self) -> str:
        size = f"{self.download_mb} MB" if self.download_mb else "size unknown"
        return f"{self.tag:<30} {size:>8}  {self.label}"


CATALOGUE: tuple[EmbedModel, ...] = (
    EmbedModel("all-minilm", 46, 47, 384, ENGLISH, 512,
               "English only - smallest, fastest",
               "Short 512-token input; weakest at long or subtle sentences; no real German/French understanding.",
               languages_source="upstream"),
    EmbedModel("nomic-embed-text", 274, 371, 768, ENGLISH, 2048,
               "English only - small, good all-rounder",
               "English only; 6 to 8 times the memory of all-minilm.",
               languages_source="upstream"),
    EmbedModel("embeddinggemma:300m-qat-q4_0", 239, 295, 768, MULTILINGUAL, 2048,
               "Multilingual (100+ languages, e.g. German, French, Spanish) - small",
               "Needs Ollama 0.11.10 or newer; the 4-bit variant trades a little precision for size.",
               ),
    EmbedModel("granite-embedding:278m", 563, 546, 768, MULTILINGUAL, 512,
               "Multilingual - mid size, Apache-2.0",
               "Short 512-token input; about twice the memory of embeddinggemma."),
    EmbedModel("qwen3-embedding:0.6b", 639, 1180, 1024, MULTILINGUAL, 32768,
               "Multilingual (100+ languages) - long inputs",
               "Loads to about 1.2 GB; no smaller than bge-m3 in memory, mainly worth it for very long texts."),
    EmbedModel("bge-m3", 1200, 1186, 1024, MULTILINGUAL, 8192,
               "Multilingual (100+ languages, e.g. German, French) - the default, best for non-English chats",
               "Biggest of the list: 1.2 GB download and about 1.2 GB of RAM while loaded."),
)

# Known, not offered by default (not measured here or no reason to prefer them over the list above).
OTHERS: tuple[EmbedModel, ...] = (
    EmbedModel("granite-embedding:30m", 63, 66, 384, ENGLISH, 512, "English only - tiny",
               "Probe showed weak separation of unrelated sentences; prefer all-minilm.", curated=False),
    EmbedModel("mxbai-embed-large", 670, None, None, ENGLISH, 512, "English only - large, strong on English",
               "670 MB for an English-only model.", languages_source="upstream", curated=False),
    EmbedModel("snowflake-arctic-embed2", 1200, None, None, MULTILINGUAL, 8192, "Multilingual - large",
               "About as big as bge-m3.", curated=False),
    EmbedModel("paraphrase-multilingual", 563, None, None, MULTILINGUAL, 512, "Multilingual - older sentence model",
               "Older model; granite-embedding:278m is the same size.", languages_source="upstream", curated=False),
)

_ALL = CATALOGUE + OTHERS


def _base(tag: str) -> str:
    return tag.split(":", 1)[0].strip().lower()


def _canon(tag: str) -> str:
    t = (tag or "").strip().lower()
    return t[: -len(":latest")] if t.endswith(":latest") else t


def describe(tag: str) -> EmbedModel:
    """The catalogue entry for `tag`. A variant of a known family (e.g. `bge-m3:567m`, `all-minilm:l12`) returns the
    family entry with `exact=False`; an unknown tag is a generic 'custom embedding model'."""
    want = _canon(tag)
    for m in _ALL:
        if _canon(m.tag) == want:
            return m
    for m in _ALL:
        if _base(m.tag) == _base(tag):
            return replace(m, exact=False, tag=tag)
    return EmbedModel(tag, None, None, None, UNKNOWN, None, "custom embedding model",
                      "Not in Nightage's list: size, languages and memory use are unknown.", curated=False, exact=False)


def curated() -> list[EmbedModel]:
    return [m for m in CATALOGUE if m.curated]


def recommend(ram_mb: int | None, vram_mb: int | None = None, languages_hint: str | None = None) -> EmbedModel:
    """The embedder to start with.

    `ram_mb` is the machine's total RAM (0 or None = unknown: be careful). The embedder runs on the CPU, so `vram_mb`
    does not change the answer; it is accepted so callers can pass what they know. `languages_hint`: 'english',
    'multilingual' (also a locale like 'de_DE.UTF-8' or 'de'), or None when unknown.

      * English chats: all-minilm below 4 GB RAM, nomic-embed-text from 4 GB.
      * other languages: embeddinggemma (small) below 16 GB RAM, bge-m3 from 16 GB.
      * unknown language: treated like English below 4 GB, else the small multilingual model - a multilingual embedder
        also handles English, the other way round does not work."""
    ram = int(ram_mb or 0)
    hint = language_hint(languages_hint)
    by_tag = {m.tag: m for m in CATALOGUE}
    if hint == ENGLISH:
        return by_tag["all-minilm"] if ram < 4096 else by_tag["nomic-embed-text"]
    if ram < 4096:
        return by_tag["all-minilm"] if hint == UNKNOWN else by_tag["embeddinggemma:300m-qat-q4_0"]
    if hint == MULTILINGUAL and ram >= 16384:
        return by_tag["bge-m3"]
    return by_tag["embeddinggemma:300m-qat-q4_0"]


def language_hint(value: str | None) -> str:
    """'english' | 'multilingual' | 'unknown' from a word or a locale name (LANG / LC_ALL)."""
    v = (value or "").strip().lower()
    if not v or v in ("unknown", "auto") or v.split(".")[0] in ("c", "posix"):
        return UNKNOWN
    if v in (ENGLISH, "en", "english-only"):
        return ENGLISH
    if v in (MULTILINGUAL, "other", "non-english"):
        return MULTILINGUAL
    return ENGLISH if v.startswith("en") else MULTILINGUAL


def warn_if_large(model: EmbedModel, ram_free_mb: int | None, ram_total_mb: int | None) -> str:
    """A sentence when the embedder takes a big share of the memory that is free, else ''."""
    need = model.ram_mb or model.download_mb
    if not need:
        return ""
    free, total = int(ram_free_mb or 0), int(ram_total_mb or 0)
    if free and need * 4 > free:
        return f"{model.tag} needs about {need} MB while loaded, a quarter or more of the {free} MB RAM that is free now"
    if not free and total and need * 8 > total:
        return f"{model.tag} needs about {need} MB while loaded, an eighth or more of this machine's {total} MB RAM"
    return ""
