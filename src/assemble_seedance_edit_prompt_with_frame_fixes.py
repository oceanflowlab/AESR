#!/usr/bin/env python3
"""
将前序步骤已产出的文本/JSON 直接拼成 Seedance 2.0 编辑 prompt（不调 LLM、不传图/视频）。

输入（均已整理好）：
  - *_gemini_critique_typed.json（summary + motion 等差异）
  - missing_visual_fix/manifest.json + issue_*/edit_prompt_en.txt

输出与 gemini_seedance_edit_prompt_with_frame_fixes.py 同结构的 JSON + *_en.txt。

  python3 assemble_seedance_edit_prompt_with_frame_fixes.py \\
    --critique-json .../id014_gemini_critique_typed.json \\
    --missing-visual-fix-dir .../id014/missing_visual_fix \\
    --out .../id014_seedance_edit_prompt_with_frame_fixes.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from gemini_seedance_edit_prompt_with_frame_fixes import (  # noqa: E402
    collect_other_differences,
    load_frame_fix_plan,
)

_PAREN_EN_RE = re.compile(r"[（(]([^）)]+)[）)]")


def format_span_sec(span: list | tuple | None) -> str | None:
    if not span or len(span) < 2:
        return None
    t0, t1 = float(span[0]), float(span[1])
    return f"[{t0:.1f}, {t1:.1f}]"


def english_phrases_from_point(point: str) -> str:
    """从 critique point 中提取括号内英文短语。"""
    bits: list[str] = []
    for m in _PAREN_EN_RE.finditer(point or ""):
        s = m.group(1).strip()
        if not s:
            continue
        if re.search(r"[a-zA-Z]", s):
            bits.append(s)
    return "; ".join(bits)


def motion_instruction(point: str, issue_type: str | None) -> str:
    from critique_issue_types import is_motion_prompt_only, normalize_issue_type

    en = english_phrases_from_point(point)
    it = normalize_issue_type(issue_type)
    if is_motion_prompt_only(it) and en:
        if "speak" in en.lower():
            return "modify the subject in Video 1 so her lips move naturally as she speaks"
        return f"modify motion in Video 1 to match: {en}"
    if en:
        return f"adjust Video 1 to match: {en}"
    return "apply the described correction in Video 1"


def build_fix_clause(entry: dict) -> tuple[str, str]:
    """返回 (完整分句, frame_fix_intervals 用的 excerpt)。"""
    issue = entry["issue_dir"]
    span = format_span_sec(entry.get("video_edit_span_sec"))
    if not span:
        span = "[0.0, end]"

    detail = english_phrases_from_point(str(entry.get("point") or ""))
    if not detail:
        detail = "apply the visual correction from the GPT-Image edited keyframe"
    low = detail.lower()
    if "shipping manifest" in low:
        detail += "; show a readable shipping manifest on the smartphone screen"
    if "lock eyes" in low or "clerk" in low:
        detail += "; add a post office clerk in the foreground for eye contact"

    excerpt = (
        f"During {span}s, edit Video 1 to match the GPT-Image fix target ({issue}_edited): {detail}"
    )
    return excerpt, excerpt


def build_motion_clause(diff: dict) -> str:
    span = format_span_sec(diff.get("approx_time_span_sec"))
    span_s = span or "[0.0, end]"
    instr = motion_instruction(str(diff.get("point") or ""), diff.get("issue_type"))
    return f"During {span_s}s, {instr}."


def assemble_prompt_en(fix_plan: list[dict], other_diffs: list[dict]) -> tuple[str, list[dict]]:
    parts: list[str] = [
        "Referencing Image 1 for the subject identity, strictly edit Video 1.",
    ]
    intervals: list[dict] = []

    for entry in fix_plan:
        clause, excerpt = build_fix_clause(entry)
        parts.append(clause + ".")
        intervals.append(
            {
                "issue_dir": entry["issue_dir"],
                "video_edit_span_sec": entry.get("video_edit_span_sec"),
                "keyframe_time_sec": entry.get("keyframe_time_sec"),
                "prompt_excerpt_en": excerpt,
            }
        )

    for diff in other_diffs:
        parts.append(build_motion_clause(diff))

    parts.append(
        "Keep all unmentioned elements of Video 1 unchanged, including original camera movement, "
        "lighting, and background. Photorealistic, no subtitles, no watermarks."
    )
    return " ".join(parts), intervals


def assemble_prompt_zh(fix_plan: list[dict], other_diffs: list[dict], summary: str) -> str:
    lines = ["参考图片1的人物身份，严格编辑视频1。"]
    for entry in fix_plan:
        span = format_span_sec(entry.get("video_edit_span_sec")) or "[全片]"
        lines.append(
            f"在 {span}s 内，按 GPT-Image 修图目标（{entry['issue_dir']}_edited）修改视频1："
            f"{(entry.get('point') or '').strip()}"
        )
    for diff in other_diffs:
        span = format_span_sec(diff.get("approx_time_span_sec")) or "[全片]"
        lines.append(f"在 {span}s 内，{diff.get('point', '').strip()}")
    lines.append("未提及部分保持视频1不变。逼真画质，无字幕，无水印。")
    if summary.strip():
        lines.append(f"背景：{summary.strip()}")
    return " ".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="纯文本拼装 Seedance 编辑 prompt（无 LLM）。")
    p.add_argument("--critique-json", type=Path, required=True)
    p.add_argument("--missing-visual-fix-dir", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, default=None, help="可选，写入 metadata")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--prompt-txt-out",
        type=Path,
        default=None,
        help="另存 seedance_edit_prompt_en（默认同目录 <out 主名>_en.txt）",
    )
    args = p.parse_args()

    critique_path = args.critique_json.expanduser().resolve()
    data = json.loads(critique_path.read_text(encoding="utf-8"))
    fix_plan = load_frame_fix_plan(Path(args.missing_visual_fix_dir))
    if not fix_plan:
        raise SystemExit("missing_visual_fix 中无有效 edited_frame")

    fix_src = {
        int(x["source_difference_index"])
        for x in fix_plan
        if x.get("source_difference_index") is not None
    }
    other_diffs = collect_other_differences(data, fix_source_indices=fix_src)

    en, intervals = assemble_prompt_en(fix_plan, other_diffs)
    summary = str(data.get("summary") or "")
    zh = assemble_prompt_zh(fix_plan, other_diffs, summary)

    rationale = (
        "Assembled locally from typed critique + missing_visual_fix manifest/edit_prompt_en "
        f"({len(fix_plan)} frame-fix interval(s), {len(other_diffs)} other diff(s)); no LLM call."
    )

    result = {
        "seedance_edit_prompt_en": en,
        "seedance_edit_prompt_zh": zh,
        "editing_rationale": rationale,
        "frame_fix_intervals": intervals,
        "assembly_mode": "text_only",
        "input_frame_fix_plan": fix_plan,
        "input_other_differences": other_diffs,
    }
    if args.prompt_file and Path(args.prompt_file).is_file():
        result["input_original_prompt"] = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    txt_path = (
        Path(args.prompt_txt_out).expanduser().resolve()
        if args.prompt_txt_out
        else out_path.with_name(out_path.stem + "_en.txt")
    )
    txt_path.write_text(en.strip() + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n✅ {out_path}", file=sys.stderr)
    print(f"✅ {txt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
