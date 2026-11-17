import pickle
import re
import threading
import time
from unittest import TestCase
from unittest.mock import patch

from pyvi.ViTokenizer import ViTokenizer


class TestViTokenizerGuards(TestCase):
    def test_is_code_or_sku_strictly_requires_letters_and_digits(self) -> None:
        mixed_skus = [
            "BH-06",
            "SP-001",
            "SP_001",
            "YR01",
            "DAY01",
            "Z3",
            "KM-04",
            "QĐ-01",
            "24h",
        ]
        for token in mixed_skus:
            with self.subTest(token=token):
                self.assertTrue(ViTokenizer.is_code_or_sku(token))

        non_skus = [
            "",
            "30",
            "7",
            "2026",
            "14-15",
            "no-show",
            "pro-rata",
            "miễn",
            "/",
            "-",
            "_",
        ]
        for token in non_skus:
            with self.subTest(token=token):
                self.assertFalse(ViTokenizer.is_code_or_sku(token))

    def test_tokenize_preserves_skus_and_segments_vietnamese_compounds(self) -> None:
        self.assertEqual(
            ViTokenizer.tokenize("Sản phẩm SP-001 miễn phí"),
            "Sản_phẩm SP-001 miễn_phí",
        )
        self.assertIn("BH-06", ViTokenizer.tokenize("Mẫu hoa BH-06 Bó Tulip"))
        self.assertEqual(
            ViTokenizer.tokenize("Điều 15 Khoản 2 trong 30 ngày"),
            "Điều 15 Khoản 2 trong 30 ngày",
        )
        self.assertNotIn("no_show", ViTokenizer.tokenize("phí no-show"))

    def test_ensure_model_is_thread_safe_single_load(self) -> None:
        orig_estimator = ViTokenizer._estimator
        orig_featurizer = ViTokenizer._featurizer
        real_load = pickle.load
        load_calls = 0
        count_lock = threading.Lock()

        def slow_load(file_obj):
            nonlocal load_calls
            with count_lock:
                load_calls += 1
            time.sleep(0.05)
            return real_load(file_obj)

        ViTokenizer._estimator = None
        ViTokenizer._featurizer = None
        barrier = threading.Barrier(6)
        results = []
        errors = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                results.append(ViTokenizer._ensure_model())
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        try:
            with patch("pickle.load", side_effect=slow_load):
                threads = [threading.Thread(target=worker) for _ in range(6)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=15)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 6)
            self.assertEqual(load_calls, 2)
            first_pair = results[0]
            for pair in results[1:]:
                self.assertIs(pair[0], first_pair[0])
                self.assertIs(pair[1], first_pair[1])
        finally:
            if ViTokenizer._estimator is None:
                ViTokenizer._estimator = orig_estimator
            if ViTokenizer._featurizer is None:
                ViTokenizer._featurizer = orig_featurizer

    def test_get_patterns_is_thread_safe_single_compile(self) -> None:
        orig_patterns = ViTokenizer._patterns
        real_compile = re.compile
        compile_calls = 0
        count_lock = threading.Lock()

        def slow_compile(pattern, flags=0):
            nonlocal compile_calls
            with count_lock:
                compile_calls += 1
            time.sleep(0.05)
            return real_compile(pattern, flags)

        ViTokenizer._patterns = None
        barrier = threading.Barrier(6)
        results = []
        errors = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                results.append(ViTokenizer._get_patterns())
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        try:
            with patch("re.compile", side_effect=slow_compile):
                threads = [threading.Thread(target=worker) for _ in range(6)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=15)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 6)
            self.assertEqual(compile_calls, 1)
            for compiled in results[1:]:
                self.assertIs(compiled, results[0])
        finally:
            if ViTokenizer._patterns is None:
                ViTokenizer._patterns = orig_patterns
