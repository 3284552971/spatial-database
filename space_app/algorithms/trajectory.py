from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Optional


def _try_load_native() -> Optional[Any]:
    try:
        import trajectory_cpp  # type: ignore

        return trajectory_cpp
    except Exception:
        pass

    build_dir = Path(__file__).resolve().parents[1] / "cpp_model" / "build"
    if not build_dir.exists():
        return None

    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidates = [build_dir / f"trajectory_cpp{ext}"]
    if not candidates[0].exists():
        candidates = sorted(build_dir.glob("trajectory_cpp*.so"))
    if not candidates:
        return None

    path = candidates[0]
    modname = "space_app.algorithms.trajectory_cpp"
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        return None
    return module


_trajectory_cpp = _try_load_native()


def native_available() -> bool:
    global _trajectory_cpp
    if _trajectory_cpp is None:
        _trajectory_cpp = _try_load_native()
    return _trajectory_cpp is not None and hasattr(_trajectory_cpp, "hmm_match_and_kalman")


def hmm_match_and_kalman(
    obs_lon: List[float],
    obs_lat: List[float],
    obs_t: List[float],
    nodes_lon: List[float],
    nodes_lat: List[float],
    k_candidates: int = 2,
    grid_cell_m: float = 50.0,
    sigma_emission_m: float = 15.0,
    sigma_transition_m: float = 80.0,
) -> Dict[str, Any]:
    global _trajectory_cpp
    if _trajectory_cpp is None:
        _trajectory_cpp = _try_load_native()
    if _trajectory_cpp is None or not hasattr(_trajectory_cpp, "hmm_match_and_kalman"):
        raise RuntimeError("trajectory native 扩展不可用，请先构建 cpp_model（trajectory_cpp）")
    return _trajectory_cpp.hmm_match_and_kalman(  # type: ignore[no-any-return]
        obs_lon,
        obs_lat,
        obs_t,
        nodes_lon,
        nodes_lat,
        int(k_candidates),
        float(grid_cell_m),
        float(sigma_emission_m),
        float(sigma_transition_m),
    )
