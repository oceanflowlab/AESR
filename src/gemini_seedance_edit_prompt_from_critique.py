#!/usr/bin/env python3
"""
第二轮：根据 gemini_video_prompt_critique 产出的 differences，结合火山 Seedance 2.0
「编辑视频」提示词规范，生成可直接交给 Seedance 2.0（方舟 content_generation）的**编辑指令**。

下游真实调用时（与 seedance2_local_video_edit / batch_seedance2_r2v 一致）输入为：
  - 参考视频：待编辑的原成片（本脚本里记为 视频1）
  - 参考图：Track1 与成片 crop 对齐的铅笔图根目录 ``IPVG2026-Test-Track1/out_gpt_image2_pencil_same_crop/<id>/``
    （默认 ``--pencil-root``；脚本内记为 图片1，用于身份与外观锚点）。若仍用 ``batch_in_gpt_pencil``，请显式 ``--pencil-root``。

  export YUNWU_API_KEY=...
  python3 gemini_seedance_edit_prompt_from_critique.py \\
    --video /path/to.mp4 \\
    --critique-json /path/to_gemini_critique.json \\
    --prompt-file /path/to/prompt.txt \\
    --id id010 \\
    --out /path/to_seedance_edit_prompt.json

也可显式传入 --reference-image /path/to.png。多模态默认均匀抽 **24** 帧辅助云雾识别画面（可用 --frame-screenshots 0 尝试整段视频）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.resolve()
# 与 ace / run_track1_* 一致：与 Seedance 成片同 crop 的铅笔参考图
DEFAULT_PENCIL_ROOT = (_REPO_ROOT / "examples" / "data" / "references").resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# 自 gemini_video_prompt_critique 复用抽帧与时长
import gemini_video_prompt_critique as gvc  # noqa: E402

SEEDANCE20_EDIT_GUIDE = """
【Seedance 2.0 编辑类提示词要点（摘自官方指南，须遵守）】
1) 任务类型：这是「编辑视频」——在**原视频**上改；未写到的画面默认保持不变。
2) 指代：编辑/延长任务里用「视频1」「图片1」指代输入；**不要**写「参考视频1」，以免被判成「参考生视频」而非编辑。
3) 句式模板：
   - 修改：严格编辑视频1，将其中的<原特征>修改为<新特征>（可补充时间或镜头描述）。
   - 增加：在视频1的<位置>添加<元素>，<出现时机>写清楚。
   - 删除：清除视频1中的<元素>；对要保持不变的部分在提示词里明确强调。
   - 若需同时用参考图约束人物外观：参考图片1的<身份/外观维度>，严格编辑视频1，<具体编辑内容>。
4) 进阶要素（在一条编辑指令里尽量写清）：主体锚点 + 动作/时序 + 场景 + 运镜（一次只一种主运镜）+ 风格/画质/约束（如 photoreal、无字幕、无水印）。
5) 方舟多模态顺序与 seedance2_local_video_edit 一致：文本 + reference_image（图片1）+ reference_video（视频1）。你生成的英文 prompt 中须显式出现「视频1」「图片1」与上述逻辑一致。
6) **画风**：图片1（及若有修帧附图）可为铅笔，仅作身份/构图参考；**成片视频1 必须是逼真实拍**（photorealistic live-action），不得把编辑结果做成铅笔素描、线稿或插画，除非用户原文明确要求该风格。
"""


def normalize_critique_differences(data: dict) -> list[dict]:
    """兼容 critique 旧版（字符串数组）与新版（带 point 的对象）。"""
    raw = data.get("differences") or []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"point": item.strip(), "approx_time_span_sec": None, "basis_snapshot_range": None})
        elif isinstance(item, dict):
            out.append(
                {
                    "point": str(item.get("point") or item.get("text") or "").strip(),
                    "approx_time_span_sec": item.get("approx_time_span_sec"),
                    "basis_snapshot_range": item.get("basis_snapshot_range"),
                }
            )
        else:
            out.append({"point": str(item).strip(), "approx_time_span_sec": None, "basis_snapshot_range": None})
    return [x for x in out if x.get("point")]


def resolve_reference_image(pencil_root: Path | None, item_id: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"找不到参考图: {p}")
        return p
    if not pencil_root or not item_id:
        raise SystemExit("请提供 --reference-image，或同时提供 --pencil-root 与 --id")
    base = pencil_root.expanduser().resolve() / item_id
    if not base.is_dir():
        raise SystemExit(f"找不到 id 目录: {base}")
    candidates = [
        base / "pencil_full_body.png",
        base / "pencil.png",
        base / "image.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        found = sorted(base.glob(pat))
        if found:
            return found[0]
    raise SystemExit(f"在 {base} 下未找到 png/jpg 参考图（可改用 --reference-image）")


def build_llm_user_text(
    *,
    original_prompt: str,
    summary: str,
    diff_items: list[dict],
    video_duration_sec: float | None,
    auxiliary_video_frame_count: int,
) -> str:
    diff_json = json.dumps(diff_items, ensure_ascii=False, indent=2)
    dur_line = ""
    if video_duration_sec is not None:
        dur_line = f"原视频时长约 {video_duration_sec:.1f} 秒。\n"

    if auxiliary_video_frame_count > 0:
        attach_note = (
            f"【多模态附件】第一张图为图片1（人物参考）。其后 {auxiliary_video_frame_count} 张为视频1 的均匀抽帧，"
            "仅便于对齐时间与动作；写编辑指令时仍以整段视频1 为被编辑对象，勿把抽帧当作独立素材编号。\n"
        )
    else:
        attach_note = "【多模态附件】第一张图为图片1（人物参考）；其后为完整视频1（待编辑原片）。\n"

    return "\n".join(
        [
            SEEDANCE20_EDIT_GUIDE.strip(),
            "",
            attach_note.strip(),
            "",
            dur_line + "【用户原始文案 / 目标描述（可能与当前成片不一致，编辑是为了向其对齐）】",
            original_prompt.strip() or "（无单独文案文件，仅以 differences 为准）",
            "",
            "【上一轮 Gemini 对「文案 vs 成片」的归纳】",
            (summary or "").strip() or "（无 summary）",
            "",
            "【上一轮列出的具体出入（每条可带时间段与抽帧区间；编辑 prompt 应逐条回应或合并为连贯指令）】",
            diff_json,
            "",
            "【你的任务】",
            "结合图片1（人物参考图）与视频1（待编辑原片），写一条给 Seedance 2.0 的**英文**编辑指令，使成片在保留未提及内容的前提下，尽量消除上述出入、并与用户文案对齐。",
            "指令须：显式使用「视频1」「图片1」；符合上面指南（编辑句式、勿写「参考视频1」）；一条主 prompt 内写清优先级（身份一致 > 关键动作/物体 > 运镜/场景）。",
            "若 differences 中给了 approx_time_span_sec 或 basis_snapshot_range，请在英文里用简短从句标出大致时间范围或镜头阶段（秒），便于模型分时编辑。",
            "",
            "只输出一个 JSON 对象，键：",
            '- "seedance_edit_prompt_en"：字符串，给方舟用的英文编辑 prompt（单段，可直接粘贴到 seedance2_local_video_edit --text 或 --prompt-file）。',
            '- "seedance_edit_prompt_zh"：字符串，同义中文摘要（便于人读）。',
            '- "editing_rationale"：字符串，一两句说明为何这样组织指令。',
            "不要其它说明文字。",
            "",
            "seedance_edit_prompt_schema",
        ]
    )


async def _run(args: argparse.Namespace) -> int:
    import ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi as ace

    critique_path = Path(args.critique_json).expanduser().resolve()
    data = json.loads(critique_path.read_text(encoding="utf-8"))
    diff_items = normalize_critique_differences(data)
    if not diff_items:
        print("❌ critique JSON 中 differences 为空", file=sys.stderr)
        return 1

    ref_path = resolve_reference_image(
        args.pencil_root.expanduser().resolve(),
        args.id.strip() if args.id else None,
        Path(args.reference_image).expanduser().resolve() if args.reference_image else None,
    )
    print(f"[info] reference image: {ref_path}", file=sys.stderr)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"找不到视频: {video_path}")

    prompt_body = ""
    if args.prompt_file:
        pf = Path(args.prompt_file).expanduser().resolve()
        if pf.is_file():
            prompt_body = pf.read_text(encoding="utf-8").strip()

    dur = gvc.probe_video_duration_sec(video_path)
    if dur is not None:
        print(f"[info] video duration ≈ {dur:.1f}s", file=sys.stderr)

    frame_n = max(0, int(args.frame_screenshots))
    tmp = None
    tdir: Path | None = None
    if frame_n > 0:
        import tempfile

        tmp = tempfile.TemporaryDirectory(prefix="edit_prompt_frames_")
        tdir = Path(tmp.__enter__())

    result: dict | None = None
    try:
        media_paths: list[str] = [str(ref_path)]
        if frame_n > 0:
            assert tdir is not None
            try:
                frames = gvc.extract_video_frames_jpg(video_path, frame_n, tdir)
            except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
                print(f"❌ 抽帧失败: {e}", file=sys.stderr)
                return 1
            media_paths.extend(str(p) for p in frames)
            print(f"[info] 附带 {len(frames)} 张视频抽帧供模型对齐画面", file=sys.stderr)
        else:
            media_paths.append(str(video_path))

        user_text = build_llm_user_text(
            original_prompt=prompt_body,
            summary=str(data.get("summary") or ""),
            diff_items=diff_items,
            video_duration_sec=dur,
            auxiliary_video_frame_count=len(frames) if frame_n > 0 else 0,
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

    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(out_text)
    if args.out:
        Path(args.out).write_text(out_text + "\n", encoding="utf-8")
        print(f"\n✅ 已写入 {args.out}", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="根据 critique JSON 生成 Seedance 2.0 编辑用英文 prompt（第二轮 Gemini）。"
    )
    p.add_argument("--video", type=Path, required=True, help="待编辑原视频（与下游传入 Ark 的成片一致）")
    p.add_argument("--critique-json", type=Path, required=True, help="gemini_video_prompt_critique.py 输出的 JSON")
    p.add_argument("--prompt-file", type=Path, default=None, help="原始用户文案（如 batch_in_gpt_pencil/.../prompt.txt）")
    p.add_argument("--reference-image", type=Path, default=None, help="人物参考图；不设则用 --pencil-root + --id 自动查找")
    p.add_argument(
        "--pencil-root",
        type=Path,
        default=DEFAULT_PENCIL_ROOT,
        help=f"含 id*/pencil.png 等的根目录；默认 {DEFAULT_PENCIL_ROOT}（与成片 same-crop 对齐）",
    )
    p.add_argument("--id", type=str, default=None, help="与目录名一致，如 id010")
    p.add_argument(
        "--frame-screenshots",
        type=int,
        default=int(os.environ.get("EDIT_PROMPT_FRAME_SCREENSHOTS", "24")),
        metavar="N",
        help="默认 24：除参考图外再附 N 张均匀抽帧；0=改为附整段视频（云雾可能不稳定）",
    )
    p.add_argument("--out", type=Path, default=None)
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

    if not args.critique_json.is_file():
        raise SystemExit(f"找不到 critique: {args.critique_json}")
    if args.prompt_file and not args.prompt_file.is_file():
        raise SystemExit(f"找不到 prompt 文件: {args.prompt_file}")
    if args.frame_screenshots < 0:
        raise SystemExit("--frame-screenshots 不能为负数")

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
