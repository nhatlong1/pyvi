from __future__ import annotations

import pickle
import re
import string
import threading
import unicodedata as ud
from pathlib import Path
from typing import ClassVar

from underthesea_core import CRFFeaturizer, CRFTagger


class ViTokenizer:
    """Modern Vietnamese Word Tokenizer using VLSP 2013 CRF model with rule-based safety guards."""

    bi_grams: ClassVar[set[str]] = set()
    tri_grams: ClassVar[set[str]] = set()

    _model_dir: ClassVar[Path] = Path(__file__).resolve().parent / "models" / "ws_crf_vlsp2013_20230727"
    _estimator: ClassVar[CRFTagger | None] = None
    _featurizer: ClassVar[CRFFeaturizer | None] = None
    _model_lock: ClassVar[threading.Lock] = threading.Lock()

    _patterns: ClassVar[re.Pattern | None] = None
    _patterns_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _ensure_model(cls) -> tuple[CRFTagger, CRFFeaturizer]:
        estimator = cls._estimator
        featurizer = cls._featurizer
        if estimator is None or featurizer is None:
            with cls._model_lock:
                estimator = cls._estimator
                featurizer = cls._featurizer
                if estimator is None or featurizer is None:
                    estimator = CRFTagger()
                    estimator.load(str(cls._model_dir / "models.bin"))
                    with open(cls._model_dir / "features.bin", "rb") as f:
                        features = pickle.load(f)
                    with open(cls._model_dir / "dictionary.bin", "rb") as f:
                        dictionary = pickle.load(f)
                    featurizer = CRFFeaturizer(features, dictionary)
                    cls._featurizer = featurizer
                    cls._estimator = estimator
        return estimator, featurizer

    @classmethod
    def _get_patterns(cls) -> re.Pattern:
        patterns = cls._patterns
        if patterns is None:
            with cls._patterns_lock:
                patterns = cls._patterns
                if patterns is None:
                    specials = [r"==>", r"->", r"\.\.\.", r">>", r"\n"]
                    digit = r"\d+(?:[\.,_]\d+)+"
                    email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
                    web = r"\w+://[^\s]+"
                    abbreviations = [
                        r"[A-ZĐ]+\.",
                        r"Tp\.",
                        r"Mr\.",
                        r"Mrs\.",
                        r"Ms\.",
                        r"Dr\.",
                        r"ThS\.",
                    ]
                    word_hyphen = r"(?<=\b)\w+\-[\w+-]*\w+"
                    word = r"\w+"
                    non_word = r"[^\w\s]"

                    combined = (
                        specials
                        + abbreviations
                        + [web, email, word_hyphen, digit, word, non_word]
                    )
                    pattern_str = "(" + "|".join(combined) + ")"
                    patterns = re.compile(pattern_str, re.UNICODE)
                    cls._patterns = patterns
        return patterns

    @staticmethod
    def sylabelize(text: str) -> tuple[str, list[str]]:
        """Split text into raw atomic syllables and punctuation tokens."""
        text = ud.normalize("NFC", text)
        pat = ViTokenizer._get_patterns()
        matches = list(re.finditer(pat, text))
        tokens = [m.group(0) for m in matches if m.group(0).strip() or m.group(0) == "\n"]
        return text, tokens

    @staticmethod
    def is_code_or_sku(token: str) -> bool:
        """Detect mixed alphanumeric identifiers, package codes, or SKUs."""
        if not token:
            return False
        return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)

    @staticmethod
    def tokenize(text: str) -> str:
        """Segment Vietnamese text, grouping compound words with '_'."""
        if not text:
            return text
        orig_text, tmp = ViTokenizer.sylabelize(text)
        if len(tmp) == 0:
            return text

        estimator, featurizer = ViTokenizer._ensure_model()
        samples = [[[token] for token in tmp]]
        x = featurizer.process(samples)[0]
        labels = estimator.tag(x)

        output = tmp[0]
        for i in range(1, len(labels)):
            tag = labels[i]
            curr = tmp[i]
            prev = tmp[i - 1]

            should_join = (tag in ("I-W", "I_W"))
            if should_join:
                # 1. Punctuation guard
                if curr in string.punctuation or prev in string.punctuation:
                    should_join = False
                # 2. Alphanumeric / SKU guard (prevents BH-06_Bó, YR01_duoc)
                elif ViTokenizer.is_code_or_sku(curr) or ViTokenizer.is_code_or_sku(prev):
                    should_join = False
                # 3. Capitalization anomaly: Titlecase after lowercase
                elif curr[0].isupper() and not prev[0].isupper():
                    should_join = False

            if should_join:
                output = output + "_" + curr
            else:
                output = output + " " + curr

        return output

    @staticmethod
    def spacy_tokenize(text: str) -> tuple[list[str], list[bool]]:
        """Tokenize text returning a list of tokens and whitespace flags."""
        if not text:
            return [], []
        orig_text, tmp = ViTokenizer.sylabelize(text)
        if len(tmp) == 0:
            return [], []

        estimator, featurizer = ViTokenizer._ensure_model()
        samples = [[[token] for token in tmp]]
        x = featurizer.process(samples)[0]
        labels = estimator.tag(x)

        tokens: list[str] = []
        token = tmp[0]

        for i in range(1, len(labels)):
            tag = labels[i]
            curr = tmp[i]
            prev = tmp[i - 1]

            should_join = (tag in ("I-W", "I_W"))
            if should_join:
                if curr in string.punctuation or prev in string.punctuation:
                    should_join = False
                elif ViTokenizer.is_code_or_sku(curr) or ViTokenizer.is_code_or_sku(prev):
                    should_join = False
                elif curr[0].isupper() and not prev[0].isupper():
                    should_join = False

            if should_join:
                token = token + "_" + curr
            else:
                tokens.append(token)
                token = curr
        tokens.append(token)

        spaces: list[bool] = []
        idx = 0
        for tok in tokens:
            idx += len(tok)
            if idx < len(orig_text) and orig_text[idx] == " ":
                spaces.append(True)
                idx += 1
            else:
                spaces.append(False)

        return tokens, spaces


def spacy_tokenize(text: str) -> tuple[list[str], list[bool]]:
    return ViTokenizer.spacy_tokenize(text)


def tokenize(text: str) -> str:
    return ViTokenizer.tokenize(text)
