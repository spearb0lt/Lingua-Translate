"""Estimating how good a translation is, without a reference translation.

The brief this project answers says accuracy is what decides whether a
translation service is worth using — DeepL overtook Google on quality, not on
features. So the system should be able to say how much to trust its own
output, not just produce it.

With no human reference to compare against, the standard trick is a **round
trip**: translate the result back to the source language and see how much of
the original survived. Done naively that measures the wrong thing — a single
engine's mistakes can cancel out on the way back, flattering itself. So the
return leg is forced through a *different* provider wherever one exists.

Similarity is chrF: an F-score over character n-grams. It is the usual choice
for exactly this situation because it needs no tokeniser, which matters when
the languages involved include Hindi, Japanese and Tamil — word boundaries
mean different things in each, and Japanese has none. Character n-grams
sidestep that entirely, and partial credit for shared word-stems is a feature
here, since a good round trip rarely returns the identical wording.

What the number is and is not, measured rather than asserted:

    1.00  identical round trip
    0.66  "My name is Rahul, I come from Delhi"      good paraphrase
    0.59  "My boat is Rahul and I am from Mumbai"    WRONG, but scores fine
    0.31  "I am Rahul from Delhi"                    terse but correct
    0.08  "The weather is nice today"                unrelated
    0.08  "ami tomake bhalobashi"                    never translated at all

Read rows three and four together: this is surface similarity, so a single
wrong content word costs little, while a correct-but-terse rendering is
punished. The metric is a smoke alarm for gross failure — untranslated
passthrough, gibberish, truncation — and a reasonable tie-breaker between two
candidates for the *same* sentence. It is not a judge of meaning, and it
cannot be; that needs a semantic model far larger than anything else here.

So it is reported as ``quality`` next to the back translation itself, and the
interface shows both. A person reading "My boat is Rahul" needs no metric.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

log = logging.getLogger(__name__)

#: chrF's standard settings: character 6-grams, recall weighted 2x precision.
_MAX_N = 6
_BETA = 2.0

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFC-normalise.

    NFC matters for Indic scripts: the same visible syllable can be encoded
    as one code point or as a base plus combining mark, and comparing those
    naively reports a difference no reader would see.
    """
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _ngrams(text: str, n: int) -> Counter:
    stripped = text.replace(" ", "")
    if len(stripped) < n:
        return Counter()
    return Counter(stripped[i:i + n] for i in range(len(stripped) - n + 1))


def chrf(hypothesis: str, reference: str, max_n: int = _MAX_N,
         beta: float = _BETA) -> float:
    """Character n-gram F-score between two strings, 0..1."""
    hyp, ref = normalise(hypothesis), normalise(reference)
    if not hyp or not ref:
        return 0.0
    if hyp == ref:
        return 1.0

    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        hyp_grams, ref_grams = _ngrams(hyp, n), _ngrams(ref, n)
        if not hyp_grams or not ref_grams:
            continue
        overlap = sum((hyp_grams & ref_grams).values())
        precisions.append(overlap / sum(hyp_grams.values()))
        recalls.append(overlap / sum(ref_grams.values()))

    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def length_penalty(original: str, round_trip: str) -> float:
    """Penalise a round trip that lost or invented a lot of text.

    chrF alone is forgiving of truncation: a translation that drops half the
    sentence can still score respectably on the half it kept. Comparing
    lengths catches that cheaply.
    """
    a, b = len(normalise(original)), len(normalise(round_trip))
    if a == 0 or b == 0:
        return 0.0
    ratio = min(a, b) / max(a, b)
    return ratio ** 0.5        # soften it; some length change is normal


def round_trip_score(original: str, round_trip: str) -> float:
    """Combined 0..1 estimate of how much meaning survived the round trip."""
    if not original.strip() or not round_trip.strip():
        return 0.0
    return round(chrf(round_trip, original) * length_penalty(original, round_trip), 4)


def describe(score: float | None) -> str:
    """A word for the number, for the interface."""
    if score is None:
        return "unscored"
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "moderate"
    if score >= 0.3:
        return "low"
    return "very low"


#: Above this similarity to the input, a "translation" probably never happened.
_PASSTHROUGH_SIMILARITY = 0.9


def looks_untranslated(source_text: str, translated: str) -> bool:
    """True when the output is suspiciously close to the input.

    Round-trip scoring has a blind spot here: if a provider hands the text
    back unchanged, translating it back also returns it unchanged, so the
    round trip looks perfect. Measured, an untranslated
    "ami tomake bhalobashi" scored 0.61 — high enough to beat real
    translations. Comparing the output against the *input* catches it
    directly, and costs nothing.

    Deliberately not applied when the two are meant to match: identical
    source and target languages are short-circuited long before this.
    """
    src, out = normalise(source_text), normalise(translated)
    if not src or not out:
        return False
    if src == out:
        return True
    # Very short strings (names, "OK", numbers) legitimately survive intact.
    if len(src) < 12:
        return False
    return chrf(out, src, max_n=4) >= _PASSTHROUGH_SIMILARITY
