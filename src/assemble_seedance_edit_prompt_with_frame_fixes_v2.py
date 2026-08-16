#!/usr/bin/env python3
"""
v2：按火山 Seedance 2.0《提示词指南》「编辑视频」句式，结合 GPT-Image 修帧结果
拼装最终给视频模型的 prompt（不调 LLM）。

与 assemble_seedance_edit_prompt_with_frame_fixes.py 独立，不修改 v1。

相对 v1：
  - missing_visual：区分「增加元素 / 修改元素」模板（指南推荐句式）
  - 修图目标映射为逻辑 @图片2..N（Image 2..N），正文写清 keyframe 时刻与 visual_target
  - motion：单独 motion_intervals + 动作细化句式
  - 输出 edited_frame_references，供后续多图 Ark 调用（见 seedance2_edit_from_prompt_json_v2.py）

  python3 assemble_seedance_edit_prompt_with_frame_fixes_v2.py \\
    --critique-json .../id014_gemini_critique_typed.json \\
    --missing-visual-fix-dir .../id014/missing_visual_fix \\
    --out .../id014_seedance_edit_prompt_with_frame_fixes_v2.json
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

from critique_issue_types import is_motion_prompt_only, normalize_issue_type  # noqa: E402
from gemini_seedance_edit_prompt_with_frame_fixes import (  # noqa: E402
    collect_other_differences,
    load_frame_fix_plan,
)
from seedance20_edit_prompt_templates import (  # noqa: E402
    ADD_SEGMENT_EN,
    ADD_SEGMENT_ZH,
    ARK_MEDIA_NOTE,
    CLOSING_EN,
    CLOSING_ZH,
    MODIFY_SEGMENT_EN,
    MODIFY_SEGMENT_ZH,
    MOTION_SEGMENT_EN,
    MOTION_SEGMENT_ZH,
    OPENING_EN,
    OPENING_ZH,
)

_PAREN_EN_RE = re.compile(r"[（(]([^）)]+)[）)]")


def format_span_sec(span: list | tuple | None) -> str:
    if not span or len(span) < 2:
        return "full clip"
    t0, t1 = float(span[0]), float(span[1])
    return f"[{t0:.1f}, {t1:.1f}]"


def english_phrases_from_point(point: str) -> str:
    bits: list[str] = []
    for m in _PAREN_EN_RE.finditer(point or ""):
        s = m.group(1).strip()
        if s and re.search(r"[a-zA-Z]", s):
            bits.append(s)
    return "; ".join(bits)


def infer_operation(point: str, frame_edit_prompt_en: str | None) -> str:
    """add | modify — 对齐指南「增加元素 / 修改元素」。"""
    blob = f"{point or ''} {frame_edit_prompt_en or ''}".lower()
    modify_keys = (
        "看不清",
        "模糊",
        "无法辨认",
        "无从证实",
        "incorrect",
        "blur",
        "unclear",
        "screen",
        "手机",
        "manifest",
        "清单",
        "shipping manifest",
    )
    add_keys = (
        "缺失",
        "未出现",
        "根本没有",
        "missing",
        "not appear",
        "clerk",
        "职员",
    )
    # 屏幕/内容不清 → 修改；人物/物体缺失 → 增加（指南分句式）
    if any(k in blob for k in modify_keys):
        return "modify"
    if any(k in blob for k in add_keys):
        return "add"
    return "add"


def visual_target_en(entry: dict) -> str:
    detail = english_phrases_from_point(str(entry.get("point") or ""))
    if not detail:
        detail = "match the composition and visible elements in the reference still"
    low = detail.lower()
    if "shipping manifest" in low or "manifest" in low:
        detail += "; show a readable shipping manifest on the smartphone screen"
    if "lock eyes" in low or "clerk" in low:
        detail += "; add a post office clerk in the foreground for natural eye contact"
    return detail


def build_frame_fix_clause(
    entry: dict,
    *,
    logical_image_index: int,
) -> tuple[str, str, dict]:
    issue = entry["issue_dir"]
    span = format_span_sec(entry.get("video_edit_span_sec"))
    target = visual_target_en(entry)
    op = infer_operation(str(entry.get("point") or ""), entry.get("frame_edit_prompt_en"))
    image_label = f"Image {logical_image_index}"
    image_label_zh = f"@图片{logical_image_index}"

    if op == "modify":
        en = MODIFY_SEGMENT_EN.format(span=span, image_label=image_label, target=target)
        zh = MODIFY_SEGMENT_ZH.format(span=span, image_label=image_label_zh, target=target)
    else:
        en = ADD_SEGMENT_EN.format(span=span, image_label=image_label, target=target)
        zh = ADD_SEGMENT_ZH.format(span=span, image_label=image_label_zh, target=target)

    meta = {
        "issue_dir": issue,
        "operation": op,
        "logical_image_index": logical_image_index,
        "image_label_en": image_label,
        "image_label_zh": f"@图片{logical_image_index}",
        "video_edit_span_sec": entry.get("video_edit_span_sec"),
        "keyframe_time_sec": entry.get("keyframe_time_sec"),
        "visual_target_en": target,
        "edited_frame": entry.get("edited_frame"),
        "aligned_frame": entry.get("aligned_frame"),
        "prompt_excerpt_en": en,
        "ark_upload_recommended": True,
    }
    return en, zh, meta


def build_motion_segments(other_diffs: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    en_parts: list[str] = []
    zh_parts: list[str] = []
    intervals: list[dict] = []
    for diff in other_diffs:
        it = normalize_issue_type(diff.get("issue_type"))
        if not is_motion_prompt_only(it) and it != "other":
            continue
        span = format_span_sec(diff.get("approx_time_span_sec"))
        en = MOTION_SEGMENT_EN.format(span=span)
        zh = MOTION_SEGMENT_ZH.format(span=span)
        en_parts.append(en)
        zh_parts.append(zh)
        intervals.append(
            {
                "source_difference_index": diff.get("source_difference_index"),
                "issue_type": diff.get("issue_type"),
                "approx_time_span_sec": diff.get("approx_time_span_sec"),
                "prompt_excerpt_en": en,
                "prompt_excerpt_zh": zh,
            }
        )
    return en_parts, zh_parts, intervals


def assemble(
    fix_plan: list[dict],
    other_diffs: list[dict],
    summary: str,
) -> dict:
    en_parts = [OPENING_EN]
    zh_parts = [OPENING_ZH]
    frame_intervals: list[dict] = []
    edited_refs: list[dict] = []

    # Image 1 = pencil（下游 Ark）；修帧从 Image 2 起编号（指南多图参考）
    for i, entry in enumerate(fix_plan, start=2):
        clause_en, clause_zh, meta = build_frame_fix_clause(entry, logical_image_index=i)
        en_parts.append(clause_en)
        zh_parts.append(clause_zh)
        frame_intervals.append(meta)
        edited_refs.append(
            {
                "logical_image_index": i,
                "issue_dir": entry["issue_dir"],
                "edited_frame": entry.get("edited_frame"),
                "aligned_frame": entry.get("aligned_frame"),
                "keyframe_time_sec": entry.get("keyframe_time_sec"),
                "video_edit_span_sec": entry.get("video_edit_span_sec"),
                "operation": meta["operation"],
                "visual_target_en": meta["visual_target_en"],
                "prompt_excerpt_en": meta["prompt_excerpt_en"],
            }
        )

    motion_en, motion_zh, motion_intervals = build_motion_segments(other_diffs)
    en_parts.extend(motion_en)
    zh_parts.extend(motion_zh)

    en_parts.append(CLOSING_EN)
    zh_parts.append(CLOSING_ZH)
    if summary.strip():
        zh_parts.append(f"背景：{summary.strip()}")

    en = " ".join(en_parts)
    zh = " ".join(zh_parts)

    return {
        "seedance_edit_prompt_en": en,
        "seedance_edit_prompt_zh": zh,
        "editing_rationale": (
            f"v2 assembly per Volcengine Seedance 2.0 edit-video templates "
            f"({len(frame_intervals)} frame-fix ops, {len(motion_intervals)} motion ops); "
            "edited PNGs mapped to logical Image 2..N in prompt text."
        ),
        "frame_fix_intervals": frame_intervals,
        "motion_intervals": motion_intervals,
        "edited_frame_references": edited_refs,
        "media_plan": {
            "image_1": "pencil identity (uploaded as reference_image)",
            "video_1": "source clip (uploaded as reference_video)",
            "image_2_onwards": "GPT-Image edited_frame per issue; verbalized in prompt; "
            "optional upload via seedance2_edit_from_prompt_json_v2.py",
            "ark_note": ARK_MEDIA_NOTE,
        },
        "assembly_mode": "text_only_v2",
        "prompt_guide_version": "seedance20_edit_prompt_templates",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v2：按官方编辑视频模板拼装 Seedance prompt。")
    p.add_argument("--critique-json", type=Path, required=True)
    p.add_argument("--missing-visual-fix-dir", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt-txt-out", type=Path, default=None)
    args = p.parse_args()

    data = json.loads(Path(args.critique_json).read_text(encoding="utf-8"))
    fix_plan = load_frame_fix_plan(Path(args.missing_visual_fix_dir))
    fix_src = {
        int(x["source_difference_index"])
        for x in fix_plan
        if x.get("source_difference_index") is not None
    }
    other_diffs = collect_other_differences(data, fix_source_indices=fix_src)
    n_diffs = len(data.get("differences") or [])
    if not fix_plan and not other_diffs and n_diffs == 0:
        raise SystemExit("typed critique 中无任何 differences 条目")
    if not fix_plan:
        print(
            "[info] 无 edited_frame；仅拼装 motion/other 段落",
            file=sys.stderr,
        )

    result = assemble(fix_plan, other_diffs, str(data.get("summary") or ""))
    result["input_frame_fix_plan"] = fix_plan
    result["input_other_differences"] = other_diffs
    if args.prompt_file and Path(args.prompt_file).is_file():
        result["input_original_prompt"] = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    txt_path = (
        Path(args.prompt_txt_out).expanduser().resolve()
        if args.prompt_txt_out
        else out_path.with_name(out_path.stem + "_en.txt")
    )
    txt_path.write_text(result["seedance_edit_prompt_en"].strip() + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n✅ {out_path}", file=sys.stderr)
    print(f"✅ {txt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
