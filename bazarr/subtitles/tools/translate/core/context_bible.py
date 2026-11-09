# coding=utf-8
"""Context Bible and scene-aware block segmentation for LLM subtitle translation.

Builds a per-title character, gender, formality, and glossary reference ("Bible")
by combining local library metadata, optional TMDB credits, and a single structured
LLM pass over sampled dialogue, then segments subtitles along scene pauses rather
than arbitrary fixed line counts.
"""

from dataclasses import dataclass, field
import base64
import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from app.database import TableEpisodes, TableMovies, TableShows, database, select
from utilities.binaries import get_binary
from ..core.translator_utils import _scope, is_movie_media_type

logger = logging.getLogger(__name__)

MIN_CUES_FOR_CONTEXT_BIBLE = 10
DEFAULT_SCENE_PAUSE_SECONDS = 3.0
DEFAULT_MIN_SCENE_BLOCK_CUES = 20
DEFAULT_MAX_SCENE_BLOCK_CUES = 75
DEFAULT_PREVIOUS_CONTEXT_LINES = 3
DEFAULT_MAX_CPS = 17.0
DEFAULT_MAX_CHARS = 84
DEFAULT_MIN_CHARS = 16
TMDB_TIMEOUT_SECONDS = 4
BIBLE_SAMPLE_MAX_LINES = 80

_BIBLE_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_BIBLE_CACHE_LOCK = threading.Lock()

_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")


def _clean_cue_text(text: str) -> str:
    cleaned = _TAG_RE.sub("", str(text or "")).strip()
    return " ".join(cleaned.split())


def _tmdb_gender_code(code: Any) -> str:
    if code == 1:
        return "F"
    if code == 2:
        return "M"
    return "?"


@dataclass
class SceneCue:
    index: str
    content: str
    start_seconds: float
    end_seconds: float
    max_chars: int
    pause_before: Optional[float] = None
    speaker_gender: Optional[str] = None

    def to_prompt_dict(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "index": self.index,
            "content": self.content,
        }
        if self.speaker_gender in ("F", "M"):
            entry["speaker_gender"] = self.speaker_gender
        return entry


@dataclass
class SceneBlock:
    block_index: int
    cues: List[SceneCue] = field(default_factory=list)
    previous_context: List[str] = field(default_factory=list)
    next_context: List[str] = field(default_factory=list)
    pause_before: Optional[float] = None


def compute_max_chars(
    start_seconds: float,
    end_seconds: float,
    max_cps: float = DEFAULT_MAX_CPS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> int:
    duration = max(0.5, float(end_seconds) - float(start_seconds))
    allowance = int(duration * max_cps * 1.2)
    return min(max_chars, max(min_chars, allowance))


def build_scene_blocks(
    subtitles: Sequence[Any],
    start_index: int = 0,
    max_block_size: int = DEFAULT_MAX_SCENE_BLOCK_CUES,
    min_block_size: int = DEFAULT_MIN_SCENE_BLOCK_CUES,
    scene_pause_seconds: float = DEFAULT_SCENE_PAUSE_SECONDS,
    context_lines: int = DEFAULT_PREVIOUS_CONTEXT_LINES,
) -> List[SceneBlock]:
    """Group parsed SRT subtitles into scene-coherent blocks based on silence gaps."""
    effective_max = max(1, int(max_block_size))
    effective_min = max(1, min(int(min_block_size), effective_max))
    blocks: List[SceneBlock] = []
    current_cues: List[SceneCue] = []
    block_pause_before: Optional[float] = None
    prev_end: Optional[float] = None

    for idx in range(max(0, start_index), len(subtitles)):
        sub = subtitles[idx]
        start_sec = sub.start.total_seconds() if hasattr(sub.start, "total_seconds") else 0.0
        end_sec = sub.end.total_seconds() if hasattr(sub.end, "total_seconds") else start_sec + 2.0
        pause = (start_sec - prev_end) if prev_end is not None and start_sec >= prev_end else None

        if current_cues:
            is_scene_break = (
                pause is not None
                and pause >= scene_pause_seconds
                and len(current_cues) >= effective_min
            )
            if is_scene_break or len(current_cues) >= effective_max:
                prev_ctx = (
                    [_clean_cue_text(c.content) for c in blocks[-1].cues[-context_lines:]]
                    if blocks and context_lines > 0
                    else []
                )
                blocks.append(
                    SceneBlock(
                        block_index=len(blocks),
                        cues=current_cues,
                        previous_context=[line for line in prev_ctx if line],
                        pause_before=block_pause_before,
                    )
                )
                block_pause_before = pause
                current_cues = []

        current_cues.append(
            SceneCue(
                index=str(idx),
                content=sub.content,
                start_seconds=start_sec,
                end_seconds=end_sec,
                max_chars=compute_max_chars(start_sec, end_sec),
                pause_before=pause if current_cues else None,
            )
        )
        prev_end = end_sec

    if current_cues:
        prev_ctx = (
            [_clean_cue_text(c.content) for c in blocks[-1].cues[-context_lines:]]
            if blocks and context_lines > 0
            else []
        )
        blocks.append(
            SceneBlock(
                block_index=len(blocks),
                cues=current_cues,
                previous_context=[line for line in prev_ctx if line],
                pause_before=block_pause_before,
            )
        )

    if context_lines > 0 and len(blocks) > 1:
        for idx in range(len(blocks) - 1):
            nxt = [_clean_cue_text(c.content) for c in blocks[idx + 1].cues[:context_lines]]
            blocks[idx].next_context = [line for line in nxt if line]

    return blocks


def sample_dialogue_for_bible(
    subtitles: Sequence[Any], max_samples: int = BIBLE_SAMPLE_MAX_LINES
) -> List[str]:
    """Select evenly spaced non-empty dialogue cues across the entire subtitle file."""
    non_empty: List[Tuple[int, str]] = []
    for idx, sub in enumerate(subtitles):
        cleaned = _clean_cue_text(getattr(sub, "content", ""))
        if len(cleaned) >= 2:
            non_empty.append((idx, cleaned))

    if not non_empty:
        return []

    if len(non_empty) <= max_samples:
        return [f"#{idx}: {text}" for idx, text in non_empty]

    step = len(non_empty) / float(max_samples)
    sampled = []
    seen = set()
    for i in range(max_samples):
        pos = min(len(non_empty) - 1, int(i * step))
        if pos not in seen:
            seen.add(pos)
            idx, text = non_empty[pos]
            sampled.append(f"#{idx}: {text}")
    return sampled


def _lookup_local_media_metadata(
    media_type: str,
    radarr_id: Optional[int],
    sonarr_series_id: Optional[int],
    sonarr_episode_id: Optional[int],
    arr_instance_id: Optional[int],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "title": "",
        "year": "",
        "imdb_id": "",
        "tmdb_id": "",
        "overview": "",
        "episode_title": "",
    }
    try:
        if is_movie_media_type(media_type) and radarr_id is not None:
            row = database.execute(
                _scope(
                    select(
                        TableMovies.title,
                        TableMovies.year,
                        TableMovies.imdbId,
                        TableMovies.tmdbId,
                        TableMovies.overview,
                    ).where(TableMovies.radarrId == radarr_id),
                    TableMovies,
                    arr_instance_id,
                )
            ).first()
            if row:
                info["title"] = str(getattr(row, "title", "") or "")
                info["year"] = str(getattr(row, "year", "") or "")
                info["imdb_id"] = str(getattr(row, "imdbId", "") or "")
                info["tmdb_id"] = str(getattr(row, "tmdbId", "") or "")
                info["overview"] = str(getattr(row, "overview", "") or "")
        elif sonarr_series_id is not None:
            row = database.execute(
                _scope(
                    select(
                        TableShows.title,
                        TableShows.year,
                        TableShows.imdbId,
                        TableShows.overview,
                    ).where(TableShows.sonarrSeriesId == sonarr_series_id),
                    TableShows,
                    arr_instance_id,
                )
            ).first()
            if row:
                info["title"] = str(getattr(row, "title", "") or "")
                info["year"] = str(getattr(row, "year", "") or "")
                info["imdb_id"] = str(getattr(row, "imdbId", "") or "")
                info["overview"] = str(getattr(row, "overview", "") or "")
            if sonarr_episode_id is not None:
                ep_row = database.execute(
                    _scope(
                        select(
                            TableEpisodes.season,
                            TableEpisodes.episode,
                            TableEpisodes.title,
                        ).where(TableEpisodes.sonarrEpisodeId == sonarr_episode_id),
                        TableEpisodes,
                        arr_instance_id,
                    )
                ).first()
                if ep_row and getattr(ep_row, "title", None):
                    info["episode_title"] = str(ep_row.title)
    except Exception:
        logger.debug("Could not read local media metadata for Context Bible", exc_info=True)
    return info


def fetch_tmdb_cast(
    media_type: str,
    tmdb_id: str = "",
    imdb_id: str = "",
    max_cast: int = 15,
) -> List[Dict[str, str]]:
    """Fetch top character names and actor genders from TMDB using the built-in key."""
    try:
        from app.tmdb import api_key

        key = api_key()
        if not key:
            return []

        is_movie = is_movie_media_type(media_type)
        resolved_id = str(tmdb_id or "").strip()
        imdb = str(imdb_id or "").strip()
        if imdb and not imdb.startswith("tt") and imdb.isdigit():
            imdb = f"tt{imdb}"

        if not resolved_id and imdb.startswith("tt"):
            find_resp = requests.get(
                f"https://api.themoviedb.org/3/find/{imdb}",
                params={"api_key": key, "external_source": "imdb_id"},
                timeout=TMDB_TIMEOUT_SECONDS,
            )
            if find_resp.status_code == 200:
                find_data = find_resp.json()
                if is_movie:
                    movies = find_data.get("movie_results") or []
                    if movies and movies[0].get("id"):
                        resolved_id = str(movies[0]["id"])
                else:
                    shows = find_data.get("tv_results") or []
                    if shows and shows[0].get("id"):
                        resolved_id = str(shows[0]["id"])
                    else:
                        eps = find_data.get("tv_episode_results") or []
                        if eps and eps[0].get("show_id"):
                            resolved_id = str(eps[0]["show_id"])

        if not resolved_id:
            return []

        endpoint = "movie" if is_movie else "tv"
        cred_resp = requests.get(
            f"https://api.themoviedb.org/3/{endpoint}/{resolved_id}/credits",
            params={"api_key": key},
            timeout=TMDB_TIMEOUT_SECONDS,
        )
        if cred_resp.status_code != 200:
            return []

        cast_list = cred_resp.json().get("cast") or []
        characters: List[Dict[str, str]] = []
        for entry in cast_list[:max_cast]:
            if not isinstance(entry, dict):
                continue
            char_name = str(entry.get("character") or "").strip()
            actor_name = str(entry.get("name") or "").strip()
            if not char_name:
                continue
            # Strip common suffixes like "(voice)" or "/ Self"
            char_name = re.sub(r"\s*\(voice\)\s*$", "", char_name, flags=re.IGNORECASE).strip()
            if not char_name:
                continue
            characters.append(
                {
                    "name": char_name,
                    "gender": _tmdb_gender_code(entry.get("gender")),
                    "actor": actor_name,
                }
            )
        return characters
    except Exception:
        logger.debug("TMDB cast lookup for Context Bible skipped", exc_info=True)
        return []


def _cache_key(
    media_type: str,
    arr_instance_id: Optional[int],
    radarr_id: Optional[int],
    sonarr_series_id: Optional[int],
    target_language: str,
) -> Optional[str]:
    inst = arr_instance_id or 0
    lang = str(target_language or "").strip().lower()
    if is_movie_media_type(media_type) and radarr_id is not None:
        return f"movie_{inst}_{radarr_id}_{lang}"
    if sonarr_series_id is not None:
        return f"series_{inst}_{sonarr_series_id}_{lang}"
    return None


def _disk_cache_path(key: str) -> Optional[Path]:
    config_root = Path("/config/cache")
    if not config_root.is_dir():
        return None
    folder = config_root / "context_bibles"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", key)
    return folder / f"{safe_key}.json"


def load_cached_bible(key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    with _BIBLE_CACHE_LOCK:
        if key in _BIBLE_MEMORY_CACHE:
            return dict(_BIBLE_MEMORY_CACHE[key])
    path = _disk_cache_path(key)
    if path and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("characters") is not None:
                with _BIBLE_CACHE_LOCK:
                    _BIBLE_MEMORY_CACHE[key] = data
                return dict(data)
        except Exception:
            logger.debug("Failed reading cached Context Bible at %s", path, exc_info=True)
    return None


def save_cached_bible(key: Optional[str], bible: Dict[str, Any]) -> None:
    if not key or not isinstance(bible, dict):
        return
    with _BIBLE_CACHE_LOCK:
        _BIBLE_MEMORY_CACHE[key] = dict(bible)
    path = _disk_cache_path(key)
    if path:
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.debug("Could not persist Context Bible to %s", path, exc_info=True)


def clear_bible_cache() -> None:
    with _BIBLE_CACHE_LOCK:
        _BIBLE_MEMORY_CACHE.clear()


def normalize_bible(
    raw_bible: Any,
    known_cast: Sequence[Dict[str, str]] = (),
    overview: str = "",
) -> Dict[str, Any]:
    """Sanitize LLM output and merge known TMDB cast genders with high priority."""
    data = raw_bible if isinstance(raw_bible, dict) else {}
    by_name: Dict[str, Dict[str, Any]] = {}

    for cast_item in known_cast:
        name = str(cast_item.get("name") or "").strip()
        if not name:
            continue
        by_name[name.lower()] = {
            "name": name,
            "gender": cast_item.get("gender") if cast_item.get("gender") in ("F", "M") else "?",
        }

    for char in data.get("characters") or []:
        if not isinstance(char, dict):
            continue
        name = str(char.get("name") or "").strip()
        if not name:
            continue
        gender = str(char.get("gender") or "?").upper()
        if gender not in ("F", "M", "?"):
            gender = "?"
        key = name.lower()
        if key in by_name:
            if by_name[key]["gender"] == "?" and gender in ("F", "M"):
                by_name[key]["gender"] = gender
            notes = str(char.get("notes") or "").strip()
            if notes:
                by_name[key]["notes"] = notes[:120]
        else:
            entry: Dict[str, Any] = {"name": name, "gender": gender}
            notes = str(char.get("notes") or "").strip()
            if notes:
                entry["notes"] = notes[:120]
            by_name[key] = entry

    relations: List[Dict[str, str]] = []
    for rel in (data.get("relations") or [])[:20]:
        if not isinstance(rel, dict):
            continue
        src = str(rel.get("from") or rel.get("a") or "").strip()
        dst = str(rel.get("to") or rel.get("b") or "").strip()
        formality = str(rel.get("formality") or rel.get("address") or "").strip()
        rel_type = str(rel.get("relation") or rel.get("type") or "").strip()
        if src and dst and (formality or rel_type):
            relations.append(
                {
                    "from": src,
                    "to": dst,
                    "relation": rel_type[:80],
                    "formality": formality[:60],
                }
            )

    glossary: List[Dict[str, str]] = []
    seen_terms = set()
    for item in (data.get("glossary") or [])[:30]:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or item.get("source") or "").strip()
        translation = str(item.get("translation") or item.get("target") or "").strip()
        if term and translation and term.lower() not in seen_terms:
            seen_terms.add(term.lower())
            glossary.append({"term": term[:80], "translation": translation[:80]})

    tone = str(data.get("tone") or "").strip()[:240]
    synopsis = str(data.get("synopsis") or overview or "").strip()[:400]

    return {
        "characters": list(by_name.values())[:25],
        "relations": relations,
        "glossary": glossary,
        "tone": tone,
        "synopsis": synopsis,
    }


def build_bible_prompt(
    target_language: str,
    title: str,
    year: str,
    overview: str,
    known_cast: Sequence[Dict[str, str]],
    dialogue_samples: Sequence[str],
) -> Tuple[str, str]:
    system_instruction = (
        f"You build a compact translation 'Context Bible' for translating a movie or episode into {target_language}. "
        "Extract ONLY facts supported by the provided metadata, cast list, and dialogue sample. "
        "Return EXCLUSIVELY a valid JSON object matching this exact structure:\n"
        "{\n"
        '  "characters": [{"name": "string", "gender": "M" | "F" | "?", "notes": "short role/trait"}],\n'
        '  "relations": [{"from": "Character A", "to": "Character B", "relation": "e.g. siblings / boss-employee", '
        f'"formality": "pronoun/address to use in {target_language}, e.g. você, tu, o senhor/a senhora"}}],\n'
        f'  "glossary": [{{"term": "recurring proper noun/place/jargon", "translation": "exact form to keep in {target_language}"}}],\n'
        '  "tone": "genre, era, setting, and register (formal, colloquial, slang)"\n'
        "}"
    )

    known_info = {
        "title": title,
        "year": year,
        "overview": overview,
        "known_cast": list(known_cast),
    }
    user_prompt = (
        f"Known metadata and cast (highest priority):\n{json.dumps(known_info, ensure_ascii=False)}\n\n"
        f"Sampled dialogue lines:\n" + "\n".join(dialogue_samples)
    )
    return system_instruction, user_prompt


def format_bible_for_prompt(bible: Optional[Dict[str, Any]]) -> str:
    if not bible or not isinstance(bible, dict):
        return ""
    compact = {
        k: v
        for k, v in bible.items()
        if v and k in ("characters", "relations", "glossary", "tone", "synopsis")
    }
    if not compact:
        return ""
    return json.dumps(compact, ensure_ascii=False)


_FIRST_PERSON_THANKS_RE = re.compile(
    r"\b(thank\s+you|thanks|much\s+obliged|i\s+appreciate\s+it)\b", re.IGNORECASE
)
_FIRST_PERSON_PRONOUN_RE = re.compile(
    r"\b(i['’]?m|i\s+am|i\s+was|i\s+feel|i\s+felt|i\s+got|i['’]?ll\s+be|i\s+will\s+be|"
    r"myself|on\s+my\s+own|left\s+me)\b",
    re.IGNORECASE,
)
_GENDERED_STATE_RE = re.compile(
    r"\b(tired|exhausted|ready|sure|certain|alone|lonely|worried|scared|afraid|"
    r"surprised|shocked|trapped|stuck|lost|confused|drunk|crazy|insane|mad|furious|"
    r"dead|alive|safe|sorry|wrong|right|done|finished|busy|occupied|invited|prepared|"
    r"grateful|thankful|born|raised|married|divorced|fired|hired|arrested|wounded|"
    r"hurt|healed|cured|blessed|cursed|saved|doomed|fucked|screwed)\b",
    re.IGNORECASE,
)
_SPEAKER_PREFIX_RE = re.compile(r"^\s*(?:\[([A-Za-z][A-Za-z0-9 ._-]{1,25})\]|([A-Z][A-Z0-9 ._-]{1,25}):)")


def is_first_person_gender_ambiguous(
    text: str, known_genders: Optional[Dict[str, str]] = None
) -> bool:
    """Return True when a cue has 1st-person expressions that inflect by speaker gender in Romance languages."""
    raw = str(text or "").strip()
    if not raw:
        return False

    if known_genders:
        m = _SPEAKER_PREFIX_RE.match(raw)
        if m:
            speaker_tag = (m.group(1) or m.group(2) or "").strip().lower()
            if speaker_tag and known_genders.get(speaker_tag) in ("F", "M"):
                return False

    cleaned = _clean_cue_text(raw)
    if not cleaned:
        return False

    if _FIRST_PERSON_THANKS_RE.search(cleaned):
        return True

    if _FIRST_PERSON_PRONOUN_RE.search(cleaned) and _GENDERED_STATE_RE.search(cleaned):
        return True

    return False


def find_gender_ambiguous_cues(
    blocks: Sequence[SceneBlock],
    bible: Optional[Dict[str, Any]] = None,
    max_probes: int = 30,
) -> List[SceneCue]:
    known_genders: Dict[str, str] = {}
    if isinstance(bible, dict):
        for ch in bible.get("characters") or []:
            if isinstance(ch, dict) and ch.get("name") and ch.get("gender") in ("F", "M"):
                name_low = str(ch["name"]).strip().lower()
                known_genders[name_low] = ch["gender"]
                for token in name_low.split():
                    if len(token) >= 3:
                        known_genders.setdefault(token, ch["gender"])

    ambiguous: List[SceneCue] = []
    for block in blocks:
        for cue in block.cues:
            if is_first_person_gender_ambiguous(cue.content, known_genders=known_genders):
                ambiguous.append(cue)
                if len(ambiguous) >= max_probes:
                    return ambiguous
    return ambiguous


def extract_audio_microclip_wav_b64(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    timeout_seconds: int = 12,
) -> Optional[str]:
    """Extract a short 12kHz mono WAV micro-clip around [start_seconds, end_seconds] via fast container seek."""
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        if os.path.getsize(video_path) < 4096:
            return None
    except OSError:
        return None

    ffmpeg_exe = get_binary("ffmpeg") or "ffmpeg"
    clip_start = max(0.0, float(start_seconds) - 0.15)
    clip_dur = min(3.2, max(1.0, (float(end_seconds) - float(start_seconds)) + 0.30))

    cmd = [
        ffmpeg_exe,
        "-v",
        "error",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{clip_dur:.3f}",
        "-i",
        video_path,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-af",
        "highpass=f=85,lowpass=f=4000",
        "-ar",
        "12000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_seconds, check=True)
        if proc.stdout and len(proc.stdout) > 512:
            return base64.b64encode(proc.stdout).decode("ascii")
    except Exception:
        logger.debug("Audio micro-clip extraction skipped for %s at %.2fs", video_path, start_seconds)
    return None


def build_audio_gender_probe_payload(
    cues_with_audio: Sequence[Tuple[SceneCue, str]],
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [
        {
            "text": (
                "For each short audio clip below from a movie or episode, listen to the voice "
                "and determine whether the speaker is Female ('F') or Male ('M'). "
                "Return EXCLUSIVELY a JSON array of objects: "
                "[{\"index\": \"<cue_index>\", \"speaker_gender\": \"F\" | \"M\"}]."
            )
        }
    ]
    for cue, wav_b64 in cues_with_audio:
        parts.append({"text": f"Cue index=\"{cue.index}\": \"{_clean_cue_text(cue.content)}\""})
        parts.append({"inlineData": {"mimeType": "audio/wav", "data": wav_b64}})

    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

