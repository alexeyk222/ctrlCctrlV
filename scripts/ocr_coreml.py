"""Shape-specialized Core ML detector with an ONNX CPU fallback."""

from collections import OrderedDict
from pathlib import Path
import re
import shutil
import sys
import time


class CoreMLDetector:
    def __init__(self, cpu_runner, model, cache, *, compute_units="CPUAndNeuralEngine", profile=False):
        self.cpu_runner = cpu_runner
        self.model = Path(model)
        self.cache = Path(cache)
        self.compute_units = compute_units
        self.profile = profile
        self.sessions = OrderedDict()
        self.failed_shapes = set()
        self.last_backend = "not run"

    def _session(self, shape):
        import onnxruntime as ort

        if shape in self.sessions:
            self.sessions.move_to_end(shape)
            return self.sessions[shape]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        for name, size in zip(("DynamicDimension.0", "DynamicDimension.1", "DynamicDimension.2"),
                              (shape[0], shape[2], shape[3])):
            options.add_free_dimension_override_by_name(name, size)
        provider = {"ModelFormat": "MLProgram", "MLComputeUnits": self.compute_units,
                    "RequireStaticInputShapes": "1",
                    "ProfileComputePlan": "1" if self.profile else "0",
                    "ModelCacheDirectory": str(self.cache / "x".join(map(str, shape)))}
        start = time.perf_counter()
        session = ort.InferenceSession(str(self.model), options,
            providers=[("CoreMLExecutionProvider", provider), "CPUExecutionProvider"])
        if "CoreMLExecutionProvider" not in session.get_providers():
            raise RuntimeError("Core ML provider was not activated")
        if session.get_inputs()[0].shape != list(shape):
            raise RuntimeError("Core ML input dimensions were not specialized")
        session.disable_fallback()
        self.sessions[shape] = session
        if len(self.sessions) > 4:
            self.sessions.popitem(last=False)
        self._prune_cache()
        print(f"Core ML detector prepared {shape} in {time.perf_counter() - start:.2f}s", file=sys.stderr)
        return session

    def _prune_cache(self):
        # Only evict our generated shape folders, never unrelated cache entries.
        if not self.cache.exists():
            return
        folders = [p for p in self.cache.iterdir()
                   if p.is_dir() and not p.is_symlink() and re.fullmatch(r"1x3x\d+x\d+", p.name)]
        retained = {"x".join(map(str, shape)) for shape in self.sessions}
        for folder in sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)[8:]:
            if folder.name not in retained:
                shutil.rmtree(folder)

    def __call__(self, x):
        import numpy as np

        tensor = x[0]
        shape = tuple(tensor.shape)
        if shape not in self.failed_shapes:
            try:
                session = self._session(shape)
                result = session.run(None, {"x": tensor})
                if result[0].shape != (shape[0], 1, shape[2], shape[3]) or not np.isfinite(result[0]).all():
                    raise RuntimeError("Core ML returned an invalid probability map")
                self.last_backend = "Core ML FP16 (CPU + ANE)"
                return result
            except Exception as exc:
                self.failed_shapes.add(shape)
                self.sessions.pop(shape, None)
                print(f"Core ML detector failed for {shape}; using ONNX CPU: {exc}", file=sys.stderr)
        self.last_backend = "ONNX CPU fallback"
        return self.cpu_runner(x=x)

    def close(self):
        self.sessions.clear()
        self.cpu_runner.close()
