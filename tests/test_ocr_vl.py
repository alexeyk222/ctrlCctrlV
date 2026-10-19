import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ocr_vl


class VLResultTests(unittest.TestCase):
    def infer(self, result, profile="paddle"):
        generate = Mock(return_value=result)
        modules = {
            "mlx_vlm": SimpleNamespace(generate=generate),
            "mlx_vlm.prompt_utils": SimpleNamespace(apply_chat_template=Mock(return_value="OCR:")),
        }
        with patch.dict(sys.modules, modules):
            return ocr_vl.recognize(SimpleNamespace(config={}), object(), object(),
                                    ocr_vl.PROFILES[profile], max_tokens=100)

    def test_preserves_literal_output(self):
        text = "SELECT col_a\n  -- x != '~y~' | l1O0\n"
        self.assertEqual(self.infer(SimpleNamespace(text=text, finish_reason="stop", generation_tokens=30)), text)

    def test_rejects_truncated_text(self):
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            self.infer(SimpleNamespace(text="incomplete", finish_reason="length", generation_tokens=100))

    def test_fast_profile_strips_markdown_wrappers(self):
        text = "Use `CAST(x AS STRING)` here:\n```sql\nSELECT `col`\n  FROM t\n```"
        result = SimpleNamespace(text=text, finish_reason="stop", generation_tokens=30)
        self.assertEqual(self.infer(result, "fast"),
                         "Use CAST(x AS STRING) here:\nSELECT `col`\n  FROM t")

    def test_accurate_profile_converts_layout_html(self):
        text = ('<div data-bbox="1 2 3 4"><h2>Title</h2></div>'
                '<div data-bbox="1 2 3 4"><pre>LEFT JOIN l\n  ON a &lt;&gt; b</pre></div>')
        result = SimpleNamespace(text=text, finish_reason="stop", generation_tokens=30)
        self.assertEqual(self.infer(result, "accurate"), "Title\n\nLEFT JOIN l\n  ON a <> b")


if __name__ == "__main__":
    unittest.main()
