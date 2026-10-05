import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ocr_worker


class WorkerProtocolTests(unittest.TestCase):
    def test_reuses_model_and_preserves_request_ids(self):
        requests = StringIO(
            json.dumps({"id": 1, "image": "first.png"}) + "\n"
            + json.dumps({"id": 2, "image": "second.png"}) + "\n"
        )
        output = StringIO()
        model = object()

        with patch.object(ocr_worker, "create_ocr", return_value=model) as create:
            with patch.object(
                ocr_worker, "recognize", side_effect=["a\nb", "Привет"]
            ) as recognize:
                with patch.object(sys, "stdin", requests), redirect_stdout(output):
                    ocr_worker.serve("ru")

        self.assertEqual(create.call_count, 1)
        self.assertEqual(recognize.call_count, 2)
        self.assertEqual(recognize.call_args_list[0].args, (model, "first.png"))
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        for response in responses[1:]:
            self.assertGreaterEqual(response.pop("seconds"), 0)
            self.assertEqual(response.pop("backend"), "unknown")
        self.assertEqual(
            responses,
            [
                {"ready": True, "backend": "unknown"},
                {"id": 1, "text": "a\nb"},
                {"id": 2, "text": "Привет"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
