#!/usr/bin/env python3
"""
根据 gemini_video_prompt_critique 输出的 JSON，对每条 difference 的「问题时间段」
从原视频中抽取关键帧（默认：区间起点 / 中点 / 终点），供后续 GPT-Image 等局部编辑使用。

时间优先使用 approx_time_span_sec [t0, t1]；若缺失则按 basis_snapshot_range
与 critique 均匀抽帧数（默认 24，与第 1 步 critique 一致）换算为秒。

  python3 extract_critique_issue_frames.py \\
    --video IPVG2026-Test-Track1/.../id014/id014.mp4 \\
    --critique-json .../id014_gemini_critique.json \\
    --out .../id014/critique_issue_frames

输出目录结构示例：
  critique_issue_frames/
    manifest.json
    issue_000/
      issue.json
      t0000.00s.jpg
      t0002.50s.jpg
      t0005.00s.jpg
    issue_001/
      ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gemini_video_prompt_critique as gvc  # noqa: E402
from public_safety import public_path  # noqa: E402


def normalize_differences(data: dict) -> list[dict]:
    raw = data.get("differences") or []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            out.append(
                {
                    "point": item.strip(),
                    "approx_time_span_sec": None,
                    "basis_snapshot_range": None,
                }
            )
        elif isinstance(item, dict):
            out.append(
                {
                    "point": str(item.get("point") or item.get("text") or "").strip(),
                    "approx_time_span_sec": item.get("approx_time_span_sec"),
                    "basis_snapshot_range": item.get("basis_snapshot_range"),
                }
            )
        else:
            out.append(
                {
                    "point": str(item).strip(),
                    "approx_time_span_sec": None,
                    "basis_snapshot_range": None,
                }
            )
    return [x for x in out if x.get("point")]


def _parse_time_span(span) -> tuple[float, float] | None:
    if not isinstance(span, (list, tuple)) or len(span) < 2:
        return None
    try:
        t0, t1 = float(span[0]), float(span[1])
    except (TypeError, ValueError):
        return None
    if t1 < t0:
        t0, t1 = t1, t0
    return t0, t1


def _parse_snapshot_range(rng) -> tuple[int, int] | None:
    if not isinstance(rng, (list, tuple)) or len(rng) < 2:
        return None
    try:
        i0, i1 = int(rng[0]), int(rng[1])
    except (TypeError, ValueError):
        return None
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1


def snapshot_index_to_sec(index_1based: int, n_uniform: int, duration_sec: float) -> float:
    """与 critique 均匀 N 帧一致：第 i 帧中心时刻 ≈ (i - 0.5) / N * T。"""
    i = max(1, min(n_uniform, index_1based))
    return max(0.0, min(duration_sec, ((i - 0.5) / n_uniform) * duration_sec))


def resolve_issue_time_span(
    issue: dict,
    *,
    duration_sec: float,
    uniform_frame_count: int,
) -> tuple[float, float, str]:
    """
    返回 (t0, t1, source) ；source 为 approx_time_span | basis_snapshot_range | full_video_fallback。
    """
    span = _parse_time_span(issue.get("approx_time_span_sec"))
    if span is not None:
        t0, t1 = span
        t0 = max(0.0, min(duration_sec, t0))
        t1 = max(0.0, min(duration_sec, t1))
        if t1 <= t0:
            t1 = min(duration_sec, t0 + 0.05)
        return t0, t1, "approx_time_span_sec"

    snap = _parse_snapshot_range(issue.get("basis_snapshot_range"))
    if snap is not None and uniform_frame_count > 0:
        i0, i1 = snap
        t0 = snapshot_index_to_sec(i0, uniform_frame_count, duration_sec)
        t1 = snapshot_index_to_sec(i1, uniform_frame_count, duration_sec)
        if t1 <= t0:
            t1 = min(duration_sec, t0 + 0.05)
        return t0, t1, "basis_snapshot_range"

    return 0.0, duration_sec, "full_video_fallback"


def sample_timestamps(t0: float, t1: float, n: int, duration_sec: float) -> list[float]:
    """在 [t0, t1] 内取 n 个时间点（含端点）；去重并限制在片长内。"""
    n = max(1, n)
    t0 = max(0.0, min(duration_sec, t0))
    t1 = max(0.0, min(duration_sec, t1))
    if t1 < t0:
        t0, t1 = t1, t0
    if n == 1:
        ts = [(t0 + t1) / 2.0]
    else:
        step = (t1 - t0) / (n - 1) if t1 > t0 else 0.0
        ts = [t0 + i * step for i in range(n)]
    out: list[float] = []
    seen: set[int] = set()
    for t in ts:
        t = max(0.0, min(duration_sec, t))
        key = int(round(t * 1000))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return sorted(out)


def format_ts_filename(t_sec: float) -> str:
    return f"t{t_sec:07.2f}s.jpg"


def extract_frame_at(
    video_path: Path,
    t_sec: float,
    out_path: Path,
    *,
    max_width: int = 0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf_parts: list[str] = []
    if max_width > 0:
        vf_parts.append(f"scale='min({max_width},iw)':-2")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
    ]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    cmd.extend(["-q:v", "2", str(out_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg 抽帧失败 @ {t_sec:.3f}s: {err}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"未生成有效帧文件: {out_path}")


def slug_issue_index(i: int) -> str:
    return f"issue_{i:03d}"


def run_extract(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    critique_path = Path(args.critique_json).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not video_path.is_file():
        raise SystemExit(f"找不到视频: {video_path}")
    if not critique_path.is_file():
        raise SystemExit(f"找不到 critique JSON: {critique_path}")

    data = json.loads(critique_path.read_text(encoding="utf-8"))
    issues = normalize_differences(data)
    if not issues:
        print("❌ critique 中无有效 differences", file=sys.stderr)
        return 1

    duration = gvc.probe_video_duration_sec(video_path)
    if duration is None or duration <= 0:
        duration = float(args.fallback_duration_sec)
        print(f"[warn] 无法探测片长，使用 fallback {duration}s", file=sys.stderr)
    else:
        print(f"[info] video duration ≈ {duration:.2f}s", file=sys.stderr)

    uniform_n = max(1, int(args.critique_uniform_frames))
    samples = max(1, int(args.samples_per_issue))
    max_width = max(0, int(args.max_width))

    manifest_issues: list[dict] = []
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, issue in enumerate(issues):
        t0, t1, span_source = resolve_issue_time_span(
            issue,
            duration_sec=duration,
            uniform_frame_count=uniform_n,
        )
        timestamps = sample_timestamps(t0, t1, samples, duration)
        issue_dir = out_root / slug_issue_index(idx)
        issue_dir.mkdir(parents=True, exist_ok=True)

        snap = issue.get("basis_snapshot_range")
        frames_meta: list[dict] = []
        for t in timestamps:
            fname = format_ts_filename(t)
            fpath = issue_dir / fname
            extract_frame_at(video_path, t, fpath, max_width=max_width)
            frames_meta.append({"time_sec": round(t, 3), "path": str(fpath.relative_to(out_root))})
            print(f"  [{slug_issue_index(idx)}] {fname}  ({span_source})", file=sys.stderr)

        issue_meta = {
            "index": idx,
            "dir": issue_dir.name,
            "point": issue.get("point"),
            "approx_time_span_sec": issue.get("approx_time_span_sec"),
            "basis_snapshot_range": snap,
            "resolved_time_span_sec": [round(t0, 3), round(t1, 3)],
            "time_span_source": span_source,
            "frames": frames_meta,
        }
        (issue_dir / "issue.json").write_text(
            json.dumps(issue_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_issues.append(issue_meta)

    manifest = {
        "video": public_path(video_path),
        "critique_json": str(critique_path),
        "duration_sec": round(duration, 3),
        "critique_uniform_frames": uniform_n,
        "samples_per_issue": samples,
        "max_width": max_width if max_width > 0 else None,
        "summary": data.get("summary"),
        "issues": manifest_issues,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ 已写入 {manifest_path}（{len(manifest_issues)} 条 issue）", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="按 critique 问题时间段从视频抽取关键帧。")
    p.add_argument("--video", type=Path, required=True, help="原成片 mp4（与 critique 对应）")
    p.add_argument("--critique-json", type=Path, required=True, help="gemini_video_prompt_critique 输出 JSON")
    p.add_argument("--out", type=Path, required=True, help="输出根目录，如 .../id014/critique_issue_frames")
    p.add_argument(
        "--samples-per-issue",
        type=int,
        default=int(__import__("os").environ.get("ISSUE_FRAMES_SAMPLES", "3")),
        help="每条 difference 在时间段内抽取的张数（默认 3：起/中/止）",
    )
    p.add_argument(
        "--critique-uniform-frames",
        type=int,
        default=int(__import__("os").environ.get("CRITIQUE_FRAME_SCREENSHOTS", "24")),
        help="第 1 步 critique 均匀抽帧数，用于 basis_snapshot_range 换算（默认 24）",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=int(__import__("os").environ.get("ISSUE_FRAMES_MAX_WIDTH", "0")),
        help="输出 JPEG 最大宽度，0 表示保持原分辨率",
    )
    p.add_argument(
        "--fallback-duration-sec",
        type=float,
        default=14.0,
        help="ffprobe 失败时假定片长（秒）",
    )
    args = p.parse_args()
    if args.samples_per_issue < 1:
        raise SystemExit("--samples-per-issue 至少为 1")
    raise SystemExit(run_extract(args))


if __name__ == "__main__":
    main()
