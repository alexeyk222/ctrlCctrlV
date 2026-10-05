import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ocr_paddle import main, reconstruct


class ReconstructTests(unittest.TestCase):
    def test_fragments_keep_measured_gap(self):
        self.assertEqual(
            reconstruct(["foo", "bar"], [[0, 0, 30, 10], [40, 0, 70, 10]]),
            "foo bar",
        )

    def test_rows_and_indent(self):
        self.assertEqual(
            reconstruct(["b", "a"], [[40, 20, 50, 30], [0, 0, 10, 10]]),
            "a\n    b",
        )

    def test_ocr_symbols_are_not_rewritten(self):
        self.assertEqual(
            reconstruct(["“x”|"], [[0, 0, 40, 10]]),
            "“x”|",
        )

    def test_missing_boxes_fail_instead_of_dropping_text(self):
        with self.assertRaises(ValueError):
            reconstruct(["one", "two"], [[0, 0, 10, 10]])

    def test_cli_prints_only_recognized_text(self):
        class FakePaddleOCR:
            def __init__(self, **kwargs):
                self.language = kwargs["lang"]

            def predict(self, input):
                assert input == "test.png"
                return [SimpleNamespace(json={"res": {
                    "rec_texts": ["hello"],
                    "rec_boxes": [[0, 0, 50, 10]],
                }})]

        fake_module = SimpleNamespace(PaddleOCR=FakePaddleOCR)
        output = StringIO()
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            with patch.object(
                sys,
                "argv",
                [
                    "ocr_paddle.py",
                    "test.png",
                    "--lang",
                    "en",
                    "--accelerator",
                    "paddle",
                ],
            ):
                with redirect_stdout(output):
                    self.assertEqual(main(), 0)
        self.assertEqual(output.getvalue(), "hello\n")


if __name__ == "__main__":
    unittest.main()
