# coding=utf-8
"""Elastic non-linear subtitle synchronization against a reference or embedded subtitle.

Uses monotonic dynamic-programming alignment over language-invariant and lexical
landmarks (proper nouns, numbers, punctuation, duration/length ratios, and token
trigrams), followed by RANSAC global scale estimation and a rolling-median elastic
curve (ElasticFit) that tracks progressive drift and scene/commercial cuts.
"""

from dataclasses import dataclass
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pysubs2

from subtitles.tools.subsync_engines import MissingSyncEngineError, SyncEngineDeclinedError
from utilities.binaries import get_binary

logger = logging.getLogger(__name__)

TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
)
SUBTITLE_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt", ".sub")

DEFAULT_MIN_MATCHED_RATIO = 0.25
DEFAULT_MIN_INLIER_RATIO = 0.60
DEFAULT_MAX_MEAN_ERROR_SECONDS = 0.65
DEFAULT_MIN_SCALE = 0.90
DEFAULT_MAX_SCALE = 1.10
DEFAULT_INLIER_TOLERANCE_SECONDS = 1.0
DEFAULT_SNAP_TOLERANCE_SECONDS = 0.60
DEFAULT_ROLLING_WINDOW = 15
DEFAULT_CUT_JUMP_SECONDS = 2.0
DEFAULT_PAIR_MIN_WEIGHT = 0.28

_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class CueFeature:
    index: int
    start: float
    end: float
    duration: float
    text: str
    clean_text: str
    tokens: frozenset
    trigrams: frozenset
    capitalized: frozenset
    numbers: frozenset
    has_question: bool
    has_exclamation: bool
    is_speech: bool


@dataclass(frozen=True)
class AnchorMatch:
    target_idx: int
    ref_idx: int
    target_start: float
    ref_start: float
    weight: float


@dataclass(frozen=True)
class ElasticCurve:
    xs: np.ndarray
    ys: np.ndarray
    inliers: np.ndarray
    scale: float
    intercept: float
    residuals: np.ndarray

    def map_time(self, t: float) -> float:
        if len(self.xs) == 0:
            return float(self.scale * t + self.intercept)
        if t <= self.xs[0]:
            return float(self.ys[0] + self.scale * (t - self.xs[0]))
        if t >= self.xs[-1]:
            return float(self.ys[-1] + self.scale * (t - self.xs[-1]))
        return float(np.interp(t, self.xs, self.ys))

    def local_slope(self, t: float, span: float = 30.0) -> float:
        lo = self.map_time(t - span)
        hi = self.map_time(t + span)
        slope = (hi - lo) / (2.0 * span)
        return max(0.5, min(2.0, slope))


def _clean_text(raw: str) -> str:
    txt = _TAG_RE.sub(" ", str(raw or "").replace("\\N", " ").replace("\\n", " "))
    return " ".join(txt.split())


def _char_trigrams(text: str) -> frozenset:
    norm = f"  {text.lower()}  "
    if len(norm) < 5:
        return frozenset()
    return frozenset(norm[i : i + 3] for i in range(len(norm) - 2))


def _extract_features(subs: Sequence[pysubs2.SSAEvent]) -> List[CueFeature]:
    features: List[CueFeature] = []
    for idx, event in enumerate(subs):
        start = float(event.start) / 1000.0
        end = max(start + 0.05, float(event.end) / 1000.0)
        clean = _clean_text(event.plaintext or event.text)
        words = _WORD_RE.findall(clean)
        tokens = frozenset(w.lower() for w in words if len(w) >= 2)
        # Proper nouns: capitalized words after the first word, or any capitalized word >= 4 chars
        cap_words = {
            w.lower()[:4]
            for pos, w in enumerate(words)
            if len(w) >= 3 and w[0].isupper() and (pos > 0 or len(w) >= 4)
        }
        nums = frozenset(_NUMBER_RE.findall(clean))
        is_speech = bool(tokens) and not (clean.startswith("[") and clean.endswith("]"))
        features.append(
            CueFeature(
                index=idx,
                start=start,
                end=end,
                duration=end - start,
                text=event.text,
                clean_text=clean,
                tokens=tokens,
                trigrams=_char_trigrams(clean),
                capitalized=frozenset(cap_words),
                numbers=nums,
                has_question="?" in clean,
                has_exclamation="!" in clean,
                is_speech=is_speech,
            )
        )
    return features


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def build_similarity_matrix(
    target: Sequence[CueFeature],
    ref: Sequence[CueFeature],
    max_time_window_seconds: float = 600.0,
) -> np.ndarray:
    """Compute cue-to-cue similarity matrix supporting both same-language and cross-language pairs."""
    n = len(target)
    m = len(ref)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    weights = np.zeros((n, m), dtype=np.float64)
    ref_starts = np.array([r.start for r in ref], dtype=np.float64)
    ref_durations = np.array([r.duration for r in ref], dtype=np.float64)
    ref_lengths = np.array([max(1, len(r.clean_text)) for r in ref], dtype=np.float64)

    for i, tc in enumerate(target):
        if not tc.is_speech:
            continue
        # Restrict candidate columns to a generous window around tc.start
        dt = np.abs(ref_starts - tc.start)
        candidate_indices = np.nonzero(dt <= max_time_window_seconds)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(m)

        t_len = max(1.0, float(len(tc.clean_text)))
        for j in candidate_indices:
            rc = ref[int(j)]
            if not rc.is_speech:
                continue

            # 1. Lexical / trigram similarity (high when same language or shared names)
            token_sim = _jaccard(tc.tokens, rc.tokens)
            trigram_sim = _jaccard(tc.trigrams, rc.trigrams)
            lexical = max(token_sim, trigram_sim)

            # 2. Cross-language structural & anchor similarity
            dur_ratio = min(tc.duration, ref_durations[j]) / max(tc.duration, ref_durations[j])
            len_ratio = min(t_len, ref_lengths[j]) / max(t_len, ref_lengths[j])

            anchor_bonus = 0.0
            if tc.numbers and rc.numbers and (tc.numbers & rc.numbers):
                anchor_bonus += 0.35
            if tc.capitalized and rc.capitalized and (tc.capitalized & rc.capitalized):
                anchor_bonus += 0.25
            if tc.has_question and rc.has_question:
                anchor_bonus += 0.15
            if tc.has_exclamation and rc.has_exclamation:
                anchor_bonus += 0.12

            # Temporal proximity prior (soft decay over max_time_window_seconds)
            proximity = max(0.0, 1.0 - (abs(tc.start - rc.start) / max_time_window_seconds))

            structural = (0.45 * dur_ratio + 0.35 * len_ratio + 0.20 * proximity) + anchor_bonus
            score = max(lexical * 0.85 + structural * 0.35, structural * 0.75)
            if score >= DEFAULT_PAIR_MIN_WEIGHT:
                weights[i, j] = min(1.5, score)

    return weights


def monotonic_align(
    weights: np.ndarray, min_weight: float = DEFAULT_PAIR_MIN_WEIGHT
) -> List[Tuple[int, int]]:
    """Find pairs (i, j) with i strictly increasing and j strictly increasing maximizing weight sum."""
    n, m = weights.shape
    if n == 0 or m == 0:
        return []
    w = np.where(weights >= min_weight, weights, 0.0)
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)

    for i in range(1, n + 1):
        row_w = w[i - 1]
        prev_diag = dp[i - 1, :-1] + row_w
        prev_up = dp[i - 1, 1:]
        best_without_left = np.maximum(prev_diag, prev_up)
        # Prefix maximum handles transition from (i, j-1)
        dp[i, 1:] = np.maximum.accumulate(best_without_left)

    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i, j] == dp[i, j - 1]:
            j -= 1
        elif w[i - 1, j - 1] > 0 and math.isclose(
            dp[i, j], dp[i - 1, j - 1] + w[i - 1, j - 1], rel_tol=1e-7, abs_tol=1e-9
        ):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    return pairs


def ransac_line(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float = DEFAULT_INLIER_TOLERANCE_SECONDS,
    iters: int = 400,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """Robust linear fit y = scale * x + intercept using deterministic RANSAC."""
    n = len(x)
    if n < 2:
        offset = float(y[0] - x[0]) if n == 1 else 0.0
        return 1.0, offset, np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    best_score = -1.0

    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        dx = float(x[j] - x[i])
        if abs(dx) < 1.0:
            continue
        a = float(y[j] - y[i]) / dx
        if not (DEFAULT_MIN_SCALE <= a <= DEFAULT_MAX_SCALE):
            continue
        b = float(y[i]) - a * float(x[i])
        err = np.abs(y - (a * x + b))
        mask = err <= tolerance
        inliers_count = int(mask.sum())
        if inliers_count > best_score:
            best_score = inliers_count
            best_mask = mask

    if best_mask.sum() < 2:
        best_mask = np.ones(n, dtype=bool)

    a_fit, b_fit = np.polyfit(x[best_mask], y[best_mask], 1)
    final_mask = np.abs(y - (a_fit * x + b_fit)) <= tolerance
    if final_mask.sum() >= 2:
        a_fit, b_fit = np.polyfit(x[final_mask], y[final_mask], 1)
    return float(a_fit), float(b_fit), final_mask


def rolling_median(
    values: np.ndarray,
    window: int = DEFAULT_ROLLING_WINDOW,
    cut_jump: float = DEFAULT_CUT_JUMP_SECONDS,
) -> np.ndarray:
    """Compute rolling median within each continuous segment without smearing across cut jumps."""
    n = len(values)
    if n == 0:
        return values
    # Split into segments wherever adjacent residual jumps exceed cut_jump
    jumps = np.nonzero(np.abs(np.diff(values)) >= cut_jump)[0] + 1
    bounds = [0, *jumps.tolist(), n]
    out = np.empty(n, dtype=np.float64)
    for start_idx, end_idx in zip(bounds[:-1], bounds[1:]):
        seg = values[start_idx:end_idx]
        seg_len = len(seg)
        eff_win = min(window, max(3, (seg_len // 2) | 1))
        half = max(1, eff_win // 2)
        for i in range(seg_len):
            out[start_idx + i] = np.median(seg[max(0, i - half) : min(seg_len, i + half + 1)])
    return out


def fit_elastic_curve(
    x: np.ndarray,
    y: np.ndarray,
    inlier_tolerance: float = DEFAULT_INLIER_TOLERANCE_SECONDS,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> ElasticCurve:
    """Fit global RANSAC line plus piecewise rolling-median residual curve for drift and cuts."""
    scale, intercept, _ = ransac_line(x, y, tolerance=inlier_tolerance)
    raw_residuals = y - (scale * x + intercept)
    median_trend = rolling_median(raw_residuals, window=window)
    inlier_band = np.abs(raw_residuals - median_trend) <= inlier_tolerance

    if int(inlier_band.sum()) < 2:
        empty = np.array([], dtype=np.float64)
        return ElasticCurve(empty, empty, inlier_band, scale, intercept, empty)

    xi = x[inlier_band]
    smooth_resid = rolling_median(raw_residuals[inlier_band], window=window)
    # Enforce monotonic time progression
    yi = np.maximum.accumulate(scale * xi + intercept + smooth_resid)
    prelim_curve = ElasticCurve(xi, yi, inlier_band, scale, intercept, np.array([], dtype=np.float64))

    errors = np.array(
        [abs(float(yy) - prelim_curve.map_time(float(xx))) for xx, yy in zip(x, y)],
        dtype=np.float64,
    )
    final_inliers = errors <= inlier_tolerance
    return ElasticCurve(xi, yi, final_inliers, scale, intercept, errors[final_inliers])


def align_subtitles_elastic(
    target_subs: pysubs2.SSAFile,
    ref_subs: pysubs2.SSAFile,
    min_matched_ratio: float = DEFAULT_MIN_MATCHED_RATIO,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    max_mean_error: float = DEFAULT_MAX_MEAN_ERROR_SECONDS,
) -> Tuple[pysubs2.SSAFile, Dict[str, Any]]:
    """Align target_subs against ref_subs using elastic curve fitting."""
    target_features = _extract_features(target_subs)
    ref_features = _extract_features(ref_subs)

    n_speech = sum(1 for f in target_features if f.is_speech) or 1
    if not ref_features or len(target_subs) < 2:
        raise SyncEngineDeclinedError(
            "elasticsync",
            "elasticsync requires at least 2 subtitle cues in both source and reference.",
        )

    weights = build_similarity_matrix(target_features, ref_features)
    pairs = monotonic_align(weights)
    matches = [
        AnchorMatch(
            target_idx=i,
            ref_idx=j,
            target_start=target_features[i].start,
            ref_start=ref_features[j].start,
            weight=float(weights[i, j]),
        )
        for i, j in pairs
    ]

    matched_ratio = len(matches) / float(n_speech)
    if matched_ratio < min_matched_ratio or len(matches) < 2:
        raise SyncEngineDeclinedError(
            "elasticsync",
            f"elasticsync matched only {matched_ratio:.0%} of speech cues "
            f"(minimum {min_matched_ratio:.0%}).",
        )

    x = np.array([m.target_start for m in matches], dtype=np.float64)
    y = np.array([m.ref_start for m in matches], dtype=np.float64)
    curve = fit_elastic_curve(x, y)

    if not (DEFAULT_MIN_SCALE <= curve.scale <= DEFAULT_MAX_SCALE):
        raise SyncEngineDeclinedError(
            "elasticsync",
            f"elasticsync measured implausible framerate scale {curve.scale:.4f} "
            f"outside [{DEFAULT_MIN_SCALE:.2f}, {DEFAULT_MAX_SCALE:.2f}].",
        )

    inlier_ratio = float(curve.inliers.mean()) if len(curve.inliers) else 0.0
    mean_error = float(curve.residuals.mean()) if len(curve.residuals) else 0.0
    quality = max(0.0, min(1.0, 0.7 * inlier_ratio + 0.3 * max(0.0, 1.0 - mean_error / max_mean_error)))

    if inlier_ratio < min_inlier_ratio or mean_error > max_mean_error:
        raise SyncEngineDeclinedError(
            "elasticsync",
            f"elasticsync measured a quality of fit of {quality:.2f} "
            f"(inliers {inlier_ratio:.0%}, mean error {mean_error:.2f}s), "
            f"below its {min_inlier_ratio:.2f} threshold.",
        )

    snapped = {
        m.target_idx: m
        for m, is_inlier in zip(matches, curve.inliers)
        if is_inlier
    }

    aligned = pysubs2.SSAFile()
    aligned.info = dict(target_subs.info)
    aligned.styles = dict(target_subs.styles)

    prev_start_sec = -1.0
    shifts: List[float] = []

    for idx, event in enumerate(target_subs):
        orig_start = float(event.start) / 1000.0
        orig_end = max(orig_start + 0.1, float(event.end) / 1000.0)
        orig_dur = orig_end - orig_start

        mapped_start = curve.map_time(orig_start)
        match = snapped.get(idx)
        if (
            match is not None
            and abs(match.ref_start - mapped_start) <= DEFAULT_SNAP_TOLERANCE_SECONDS
            and match.ref_start > prev_start_sec
        ):
            mapped_start = match.ref_start

        if mapped_start <= prev_start_sec:
            mapped_start = prev_start_sec + 0.001

        local_scale = curve.local_slope(orig_start)
        mapped_end = mapped_start + max(0.2, orig_dur * local_scale)

        shifts.append(mapped_start - orig_start)
        prev_start_sec = mapped_start

        cloned = event.copy()
        cloned.start = max(0, int(round(mapped_start * 1000.0)))
        cloned.end = max(cloned.start + 100, int(round(mapped_end * 1000.0)))
        aligned.events.append(cloned)

    # Prevent overlaps with next cue by trimming end time (leaving at least 40ms gap)
    for k in range(len(aligned.events) - 1):
        cur_ev = aligned.events[k]
        next_ev = aligned.events[k + 1]
        if cur_ev.end > next_ev.start - 40:
            cur_ev.end = max(cur_ev.start + 100, next_ev.start - 40)

    median_offset = float(np.median(shifts)) if shifts else float(curve.intercept)
    drift = float(shifts[-1] - shifts[0]) if len(shifts) >= 2 else 0.0

    diagnostics = {
        "success": True,
        "sync_was_successful": True,
        "offset_seconds": round(median_offset, 3),
        "quality_of_fit": round(quality, 4),
        "inlier_ratio": round(inlier_ratio, 4),
        "matched_ratio": round(matched_ratio, 4),
        "mean_error_seconds": round(mean_error, 4),
        "skew": round(curve.scale, 6),
        "framerate_scale_factor": round(curve.scale, 6),
        "drift_seconds": round(drift, 3),
    }
    return aligned, diagnostics


def _extract_embedded_subtitle_track(
    video_path: str, stream_spec: Optional[str] = None
) -> Optional[str]:
    """Extract an embedded text subtitle track from video_path to a temporary .srt file."""
    if not video_path or not os.path.isfile(video_path):
        return None

    ffprobe_exe = get_binary("ffprobe") or "ffprobe"
    ffmpeg_exe = get_binary("ffmpeg") or "ffmpeg"

    chosen_spec = stream_spec if (stream_spec and str(stream_spec).startswith("s:")) else None
    if not chosen_spec:
        probe_cmd = [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name:stream_tags=language",
            "-of",
            "json",
            video_path,
        ]
        try:
            proc = subprocess.run(
                probe_cmd, capture_output=True, text=True, timeout=15, check=True
            )
            streams = json.loads(proc.stdout or "{}").get("streams") or []
            for sub_pos, stream in enumerate(streams):
                codec = str(stream.get("codec_name") or "").lower()
                if codec in TEXT_SUBTITLE_CODECS:
                    chosen_spec = f"s:{sub_pos}"
                    break
        except Exception:
            logger.debug("elasticsync ffprobe inspection failed on %s", video_path, exc_info=True)
            return None

    if not chosen_spec:
        return None

    fd, temp_srt = tempfile.mkstemp(prefix=".bazarr-elastic-ref-", suffix=".srt")
    os.close(fd)
    extract_cmd = [
        ffmpeg_exe,
        "-y",
        "-v",
        "error",
        "-i",
        video_path,
        "-map",
        f"0:{chosen_spec}",
        temp_srt,
    ]
    try:
        subprocess.run(extract_cmd, capture_output=True, text=True, timeout=120, check=True)
        if os.path.isfile(temp_srt) and os.path.getsize(temp_srt) > 0:
            return temp_srt
    except Exception:
        logger.debug("elasticsync ffmpeg subtitle extraction failed on %s", video_path, exc_info=True)
    if os.path.exists(temp_srt):
        try:
            os.unlink(temp_srt)
        except OSError:
            pass
    return None


def run_elasticsync(
    reference: str,
    subtitle_file: str,
    output_file: str,
    video_path: Optional[str] = None,
    stream_spec: Optional[str] = None,
) -> Dict[str, Any]:
    """Run elastic subtitle alignment and write the synchronized file to output_file."""
    if not subtitle_file or not os.path.isfile(subtitle_file):
        raise MissingSyncEngineError("elasticsync", f"Source subtitle not found: {subtitle_file}")

    extracted_temp: Optional[str] = None
    ref_sub_path: Optional[str] = None

    try:
        if reference and os.path.isfile(reference) and reference.lower().endswith(SUBTITLE_EXTENSIONS):
            ref_sub_path = reference
        else:
            media_file = (
                video_path
                if (video_path and os.path.isfile(video_path))
                else (reference if (reference and os.path.isfile(reference)) else None)
            )
            spec = stream_spec
            if not spec and isinstance(reference, str) and reference.startswith("s:"):
                spec = reference
            if media_file:
                extracted_temp = _extract_embedded_subtitle_track(media_file, stream_spec=spec)
                ref_sub_path = extracted_temp

        if not ref_sub_path or not os.path.isfile(ref_sub_path):
            raise SyncEngineDeclinedError(
                "elasticsync",
                "elasticsync declined: no reference or embedded text subtitle track available.",
            )

        target_subs = pysubs2.load(subtitle_file, encoding="utf-8")
        ref_subs = pysubs2.load(ref_sub_path, encoding="utf-8")
        aligned_subs, diagnostics = align_subtitles_elastic(target_subs, ref_subs)
        aligned_subs.save(str(output_file), encoding="utf-8")
        return diagnostics
    finally:
        if extracted_temp and os.path.exists(extracted_temp):
            try:
                os.unlink(extracted_temp)
            except OSError:
                pass
