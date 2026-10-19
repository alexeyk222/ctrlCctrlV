import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ocr_coreml import CoreMLDetector


class CoreMLFallbackTests(unittest.TestCase):
    def test_failed_shape_uses_cpu_without_repeated_compilation(self):
        expected = [np.ones((1, 1, 32, 32), dtype=np.float32)]
        cpu = Mock(return_value=expected)
        detector = CoreMLDetector(cpu, "/unused/model", "/unused/cache")
        tensor = [np.zeros((1, 3, 32, 32), dtype=np.float32)]
        with patch.object(detector, "_session", side_effect=RuntimeError("compile failed")) as compile:
            self.assertIs(detector(tensor), expected)
            self.assertIs(detector(tensor), expected)
        self.assertEqual(compile.call_count, 1)
        self.assertEqual(cpu.call_count, 2)
        self.assertEqual(detector.last_backend, "ONNX CPU fallback")

    def test_invalid_output_falls_back(self):
        cpu = Mock(return_value=["cpu result"])
        detector = CoreMLDetector(cpu, "/unused/model", "/unused/cache")
        session = Mock()
        session.run.return_value = [np.full((1, 1, 32, 32), np.nan)]
        with patch.object(detector, "_session", return_value=session):
            self.assertEqual(detector([np.zeros((1, 3, 32, 32))]), ["cpu result"])


if __name__ == "__main__":
    unittest.main()
