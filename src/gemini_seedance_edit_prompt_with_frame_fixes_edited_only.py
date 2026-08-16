#!/usr/bin/env python3
"""
用 Gemini（云雾）根据修帧结果写 Seedance 编辑 prompt。

与 gemini_seedance_edit_prompt_with_frame_fixes.py 独立，不修改原脚本。

默认媒体（edited-only，不再传视频均匀抽帧）：
  - 图片1：铅笔（身份锚点）
  - 图片2..N：各 issue 的 edited_frame.png（GPT-Image 修图目标）
  不传原视频、不传 critique 阶段的 24 张均匀抽帧。

文本必传（前序 VLM 已产出，本步不再看图做 critique）：
  - 完整 *_gemini_critique_typed.json（summary + differences 含 issue_type / 时间段）
  - missing_visual_fix 计划 + motion 等条目

可选：
  --attach-aligned  每条 issue 再多一张 aligned_frame.jpg（底帧对比）
  --include-video   额外附上整段 mp4（一般不必）
  --frame-screenshots N  若仍要抽帧可显式开启（默认 0）

提示词内嵌：《Seedance 2.0 提示词指南》编辑视频句式模板（seedance20_edit_prompt_templates.py）。

  export YUNWU_API_KEY=...

  python3 gemini_seedance_edit_prompt_with_frame_fixes_edited_only.py \\
    --critique-json .../id014_gemini_critique_typed.json \\
    --missing-visual-fix-dir .../id014/missing_visual_fix \\
    --reference-image .../pencil.png \\
    --prompt-file .../prompt.txt \\
    --out .../id014_seedance_edit_prompt_vlm_edited_only.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gemini_video_prompt_critique as gvc  # noqa: E402
from gemini_seedance_edit_prompt_from_critique import (  # noqa: E402
    DEFAULT_PENCIL_ROOT,
    SEEDANCE20_EDIT_GUIDE,
    resolve_reference_image,
)
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
    PHOTOREAL_VIDEO_RULE_EN,
    PHOTOREAL_VIDEO_RULE_ZH,
    SEEDANCE_PROMPT_FORBIDDEN_PHRASES,
    STEP3_LLM_PHOTOREAL_INSTRUCTION,
)


def _guide_templates_block() -> str:
    return "\n".join(
        [
            "【官方指南 · 编辑视频推荐句式模板（须遵循）】",
            f"开场 EN 示例: {OPENING_EN}",
            f"开场 ZH 示例: {OPENING_ZH}",
            f"增加元素 EN: {ADD_SEGMENT_EN}",
            f"增加元素 ZH: {ADD_SEGMENT_ZH}",
            f"修改元素 EN: {MODIFY_SEGMENT_EN}",
            f"修改元素 ZH: {MODIFY_SEGMENT_ZH}",
            f"运动修改 EN: {MOTION_SEGMENT_EN}",
            f"运动修改 ZH: {MOTION_SEGMENT_ZH}",
            f"结尾 EN: {CLOSING_EN}",
            f"结尾 ZH: {CLOSING_ZH}",
            f"Ark 说明: {ARK_MEDIA_NOTE}",
        ]
    )


def build_media_paths_edited_only(
    ref_path: Path,
    video_path: Path | None,
    frame_n: int,
    fix_plan: list[dict],
    *,
    attach_aligned: bool,
    include_video: bool,
) -> tuple[list[str], tempfile.TemporaryDirectory | None, dict]:
    """
    返回 (paths, tmp, stats)。
    stats: n_pencil, n_edited, n_aligned, n_video_frames, n_video_file
    """
    tmp = tempfile.TemporaryDirectory(prefix="seedance_prompt_edited_") if frame_n > 0 else None
    tdir = Path(tmp.__enter__()) if tmp else None

    media: list[str] = [str(ref_path)]
    stats = {
        "n_pencil": 1,
        "n_edited": 0,
        "n_aligned": 0,
        "n_video_frames": 0,
        "n_video_file": 0,
    }

    if frame_n > 0 and video_path and video_path.is_file():
        assert tdir is not None
        frames = gvc.extract_video_frames_jpg(video_path, frame_n, tdir)
        media.extend(str(p) for p in frames)
        stats["n_video_frames"] = len(frames)

    if include_video and video_path and video_path.is_file():
        media.append(str(video_path))
        stats["n_video_file"] = 1

    for entry in fix_plan:
        if attach_aligned and entry.get("aligned_frame"):
            media.append(entry["aligned_frame"])
            stats["n_aligned"] += 1
        if entry.get("edited_frame"):
            media.append(entry["edited_frame"])
            stats["n_edited"] += 1

    return media, tmp, stats


def build_llm_user_text(
    *,
    original_prompt: str,
    critique_data: dict,
    fix_plan: list[dict],
    other_diffs: list[dict],
    video_duration_sec: float | None,
    stats: dict,
    attach_aligned: bool,
) -> str:
    summary = str(critique_data.get("summary") or "")
    differences = critique_data.get("differences") or []
    critique_full_json = json.dumps(
        {
            "summary": summary,
            "differences": differences,
            "classification_mode": critique_data.get("classification_mode"),
            "source_critique_json": critique_data.get("source_critique_json"),
        },
        ensure_ascii=False,
        indent=2,
    )
    fix_json = json.dumps(fix_plan, ensure_ascii=False, indent=2)
    other_json = json.dumps(other_diffs, ensure_ascii=False, indent=2)
    dur_line = ""
    if video_duration_sec is not None:
        dur_line = f"原视频时长约 {video_duration_sec:.1f} 秒（仅文本说明，未附抽帧）。\n"

    attach_lines = [
        "【多模态附件顺序 · edited-only 模式】",
        "1) 图片1：人物铅笔参考图（Image 1 / @图片1，身份锚点；下游 Ark 的 reference_image）。",
    ]
    if not fix_plan:
        attach_lines.append(
            "（无 Image 2..N 修帧附图：typed critique 中无 missing_visual_element/motion_state，或修帧未产出；"
            "仅依据下方完整 critique JSON 写 motion_process/other 编辑指令。）"
        )
    k = 2
    for entry in fix_plan:
        issue = entry["issue_dir"]
        if attach_aligned and entry.get("aligned_frame"):
            attach_lines.append(
                f"{k}) {issue}_aligned：原片 keyframe≈{entry.get('keyframe_time_sec')}s 对齐底帧（对比用）。"
            )
            k += 1
        attach_lines.append(
            f"{k}) 附图 {issue}/edited_frame（**铅笔素描**，仅作构图/姿态/缺失元素布局参考，**不是**成片目标画风）"
            f"→ 写入正文时称 Image {k} / @图片{k}，描述应对应**逼真实拍**画面（勿写 issue 编号进 Seedance 正文）；"
            f"编辑时段 video_edit_span_sec={entry.get('video_edit_span_sec')}。"
        )
        k += 1

    if stats.get("n_video_frames"):
        attach_lines.append(
            f"（额外）附 {stats['n_video_frames']} 张原片抽帧，仅因用户开启 --frame-screenshots。"
        )
    if stats.get("n_video_file"):
        attach_lines.append("（额外）附完整原视频文件，仅因用户开启 --include-video。")

    return "\n".join(
        [
            SEEDANCE20_EDIT_GUIDE.strip(),
            "",
            _guide_templates_block(),
            "",
            "\n".join(attach_lines),
            "",
            dur_line + "【用户原始 HOI 文案】",
            original_prompt.strip() or "（无）",
            "",
            "【完整 typed critique（前序 VLM 文本 vs 视频分析结果，必须据此写 Seedance 指令）】",
            "含每条 difference 的 issue_type、point、approx_time_span_sec、issue_type_rationale 等。",
            "本请求不再附 critique 阶段的均匀抽帧图；差异判断以该 JSON 为准。"
            + (
                "修图目标以附图 Image 2..N 为准。"
                if fix_plan
                else "无修帧附图时仅据 JSON 中的 motion_process/other 等条目写指令。"
            ),
            critique_full_json,
            "",
            "【missing_visual 修帧计划（manifest + edit_prompt_en，与附图 issue_XXX_edited 对应）】",
            fix_json,
            "",
            "【其它差异条目（motion_process/other；通常已包含在上方 differences 中，此处便于单独强调）】",
            other_json,
            "",
            "【下游 Ark】",
            "默认仅上传：英文 prompt + Image1(铅笔，身份锚点) + Video1(原片 HTTPS)。",
            "edited 铅笔图在本请求中供你**看图写词**（提炼实拍应出现的物体/姿态/构图）；正文里用 Image 2..N / @图片2..N 指代即可。",
            "",
            STEP3_LLM_PHOTOREAL_INSTRUCTION.strip(),
            "",
            f"【Seedance 正文须体现的成片风格】\nEN: {PHOTOREAL_VIDEO_RULE_EN}\nZH: {PHOTOREAL_VIDEO_RULE_ZH}",
            "",
            f"【Seedance 正文禁止项】{SEEDANCE_PROMPT_FORBIDDEN_PHRASES}",
            "另禁止在 seedance 正文中要求 pencil sketch / line art / 铅笔素描 / 保持素描画风 作为成片风格。",
            "",
            "【任务】",
            "输出 JSON：",
            "- seedance_edit_prompt_en：一条连贯英文，仅 Video 1 / Image 1..N、时间段、纯视觉描述；",
            "  按指南模板写增加/修改/运动；**全文须要求 Video 1 输出为 photorealistic live-action**；",
            "  禁止 GPT-Image、issue_XXX_edited、aligned near、keyframe fix 等；",
            "- seedance_edit_prompt_zh：对应中文，仅 @视频1 @图片1..N、时间段、纯视觉描述；**须含逼真实拍/非铅笔成片**；",
            "- editing_rationale；",
            "- frame_fix_intervals（含 issue_dir, operation add|modify, video_edit_span_sec, keyframe_time_sec, "
            "visual_target_en, prompt_excerpt_en）；",
            "- motion_intervals（若有 motion_process 等过程类差异）。",
            "不要其它文字。",
        ]
    )


async def _run(args: argparse.Namespace) -> int:
    import ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi as ace

    critique_path = Path(args.critique_json).expanduser().resolve()
    data = json.loads(critique_path.read_text(encoding="utf-8"))
    fix_plan = load_frame_fix_plan(Path(args.missing_visual_fix_dir))
    fix_src = {
        int(x["source_difference_index"])
        for x in fix_plan
        if x.get("source_difference_index") is not None
    }
    other_diffs = collect_other_differences(data, fix_source_indices=fix_src)
    n_diffs = len(data.get("differences") or [])
    if not fix_plan and not other_diffs and n_diffs == 0:
        print("❌ typed critique 中无任何 differences 条目", file=sys.stderr)
        return 1
    if not fix_plan:
        print(
            "[info] 无 frame-fix 修帧图（仅 motion_process/other 或无可修帧条目）；"
            "仅上传铅笔图 + 依据 critique JSON 写 prompt",
            file=sys.stderr,
        )

    ref_path = resolve_reference_image(
        args.pencil_root.expanduser().resolve(),
        args.id.strip() if args.id else None,
        Path(args.reference_image).expanduser().resolve() if args.reference_image else None,
    )

    video_path: Path | None = None
    dur = None
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if video_path.is_file():
            dur = gvc.probe_video_duration_sec(video_path)

    prompt_body = ""
    if args.prompt_file:
        pf = Path(args.prompt_file).expanduser().resolve()
        if pf.is_file():
            prompt_body = pf.read_text(encoding="utf-8").strip()

    frame_n = max(0, int(args.frame_screenshots))
    print(
        f"[info] fix={len(fix_plan)} motion_process/other={len(other_diffs)} | "
        f"frame_screenshots={frame_n} include_video={bool(args.include_video)} "
        f"attach_aligned={bool(args.attach_aligned)}",
        file=sys.stderr,
    )

    tmp = None
    result: dict | None = None
    try:
        media_paths, tmp, stats = build_media_paths_edited_only(
            ref_path,
            video_path,
            frame_n,
            fix_plan,
            attach_aligned=bool(args.attach_aligned),
            include_video=bool(args.include_video),
        )
        total = len(media_paths)
        print(
            f"[info] 上传图片共 {total} 项: "
            f"铅笔={stats['n_pencil']} edited={stats['n_edited']} "
            f"aligned={stats['n_aligned']} "
            f"video_frames={stats['n_video_frames']} video_file={stats['n_video_file']}",
            file=sys.stderr,
        )

        user_text = build_llm_user_text(
            original_prompt=prompt_body,
            critique_data=data,
            fix_plan=fix_plan,
            other_diffs=other_diffs,
            video_duration_sec=dur,
            stats=stats,
            attach_aligned=bool(args.attach_aligned),
        )
        result = await ace.call_qwen_json(
            user_text,
            media_paths=media_paths,
            temperature=float(args.temperature),
            max_retries=args.max_retries,
            timeout=float(args.timeout),
            max_tokens=int(args.max_output_tokens),
            yunwu_text_first=True,
        )
    finally:
        if tmp is not None:
            tmp.__exit__(None, None, None)

    if not result:
        print("❌ 未得到 JSON", file=sys.stderr)
        return 1

    result["vlm_media_stats"] = stats
    result["assembly_mode"] = "gemini_edited_only"
    result["input_critique_typed"] = {
        "summary": data.get("summary"),
        "differences": data.get("differences"),
        "source_critique_json": data.get("source_critique_json"),
    }
    result["input_frame_fix_plan"] = fix_plan
    result["input_other_differences"] = other_diffs

    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(out_text)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        en = result.get("seedance_edit_prompt_en")
        if isinstance(en, str) and en.strip():
            txt_path = out_path.with_name(out_path.stem + "_en.txt")
            if args.prompt_txt_out:
                txt_path = Path(args.prompt_txt_out).expanduser().resolve()
            txt_path.write_text(en.strip() + "\n", encoding="utf-8")
            print(f"\n✅ {out_path}\n✅ {txt_path}", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gemini 写 Seedance prompt：仅铅笔+edited 图（默认无抽帧/无视频）。"
    )
    p.add_argument("--critique-json", type=Path, required=True)
    p.add_argument("--missing-visual-fix-dir", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None, help="仅用于时长/可选抽帧或 --include-video")
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--reference-image", type=Path, default=None)
    p.add_argument("--pencil-root", type=Path, default=DEFAULT_PENCIL_ROOT)
    p.add_argument("--id", type=str, default=None)
    p.add_argument(
        "--frame-screenshots",
        type=int,
        default=int(os.environ.get("EDIT_PROMPT_FRAME_SCREENSHOTS", "0")),
        help="默认 0：不传原片均匀抽帧",
    )
    p.add_argument(
        "--include-video",
        action="store_true",
        help="额外附上完整 mp4（默认不传）",
    )
    p.add_argument(
        "--attach-aligned",
        action="store_true",
        help="每条 issue 额外附 aligned_frame.jpg",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--prompt-txt-out", type=Path, default=None)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.environ.get("EDIT_PROMPT_MAX_OUTPUT_TOKENS", "8192")),
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()
    if args.id:
        args.id = args.id.strip()

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
