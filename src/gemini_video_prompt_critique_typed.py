#!/usr/bin/env python3
"""
在 gemini_video_prompt_critique 基础上，为每条 difference 增加 issue_type 分类（独立脚本，不修改原版）。

issue_type 取值（英文键，便于下游脚本）：
  - missing_visual_element  缺失视觉元素（画面里根本没有该人/物/场景元素）
  - motion_state            运动/姿态/运镜的**终态或结果状态**与文本不符（可修关键帧作参考）
  - motion_process          **终态已基本正确**，仅速度/节奏/过程/运镜动态与文本不符（仅视频编辑 prompt）
  - other                   其它

两种用法：

1) 从头生成带类型的 critique（等同原版 + 分类字段）：
  python3 gemini_video_prompt_critique_typed.py \\
    --video .../id014.mp4 \\
    --prompt-file .../prompt.txt \\
    --out .../id014_gemini_critique_typed.json

2) 仅对已有 critique JSON 补标注（不重写 point，适合已有 idXXX_gemini_critique.json）：
  python3 gemini_video_prompt_critique_typed.py \\
    --critique-json .../id014_gemini_critique.json \\
    --video .../id014.mp4 \\
    --out .../id014_gemini_critique_typed.json

认证与抽帧逻辑同 gemini_video_prompt_critique.py（默认 YUNWU_API_KEY + 24 帧）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gemini_video_prompt_critique as gvc  # noqa: E402
from critique_issue_types import (  # noqa: E402
    ISSUE_TYPE_ZH,
    ISSUE_TYPES,
    normalize_issue_type,
)
from public_safety import public_path  # noqa: E402

CLASSIFY_GUIDE = """
【issue_type 分类规则（每条 difference 必须选一个）】

1) missing_visual_element（缺失视觉元素）
   - 文本明确需要出现在画面中的**人、物体、场景元素、UI、文字牌、动物**等，但抽帧/画面中完全看不到或无法辨认。
   - 例：文本要与职员对视交涉，画面只有女子直视镜头、没有职员 → missing_visual_element
   - 例：文本要求全息界面里的赛博格，画面里没有 → missing_visual_element
   - 注意：若元素存在但样子不对（如衣服颜色错），不算缺失，可用 other 或在 point 里说明。

2) motion_state（运动终态不符 · 需修帧参考）
   - **成片在相关时段结束时的姿态/站位/朝向/构图结果/运镜终态**与文本要求**不一致**；
     仅靠文字描述「加快/放慢」无法纠正，需要一张**目标状态的关键帧**供后续视频编辑对齐。
   - 包括：文本要求站着但人一直坐着；要求向右平移/拉开镜头但画面构图终态不对；
     要求转头看向左方但片末仍低头；要求举手/开门/拿起物体但画面中从未出现该姿态结果。
   - **不属于本类**：终态已基本正确，仅动作**快慢、节奏、中间过程**不同 → 用 motion_process。
   - **左右方向**：未写明「人物自身视角」时，按**观众视角**（画面左/右）比对。

3) motion_process（运动过程不符 · 仅视频编辑）
   - **关键姿态/场景结果在片末或该时段已与文本一致**，差异主要在**速度、幅度强弱、节奏、停顿、表情变化过程、运镜加减速**等时间维度。
   - 例：文本要求「快速低头」，画面是「缓慢低头」且最终都已低头 → motion_process
   - 例：文本要求 dolly-zoom，画面有推拉但速度/曲线不对、终态构图已接近 → motion_process
   - 反例（勿记为差异）：文本「头向左转」且画面头已到画面左侧 → 一致，不记入 differences

4) other（其它）
   - 全片艺术风格、画质标签、与上述三类边界不清的差异。

【勿再使用旧标签 "motion"；必须在 motion_state 与 motion_process 中二选一。】
"""


def build_user_prompt_typed(
    description: str,
    has_image: bool,
    duration_sec: float | None,
    frame_screenshots: int,
) -> str:
    base = gvc.build_user_prompt(
        description, has_image, duration_sec, frame_screenshots
    )
    allowed = " | ".join(f'"{t}"' for t in ISSUE_TYPES)
    extra = (
        "\n"
        + CLASSIFY_GUIDE.strip()
        + "\n\n"
        "在 differences 每一项中** additionally ** 增加：\n"
        f'- "issue_type"：字符串，仅允许 {allowed}。\n'
        '- "issue_type_zh"：与 issue_type 对应的中文短标签。\n'
        '- "issue_type_rationale"：一句中文，说明为何归入该类型（不超过 40 字）。\n'
        "仍须保留 point、approx_time_span_sec、basis_snapshot_range（若有）。不要其它说明文字。"
    )
    return base + extra


def normalize_differences(data: dict) -> list[dict]:
    raw = data.get("differences") or []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"point": item.strip()})
        elif isinstance(item, dict):
            out.append(dict(item))
        else:
            out.append({"point": str(item).strip()})
    return [x for x in out if (x.get("point") or "").strip()]


def merge_typed_with_source(typed: list[dict], source: list[dict]) -> list[dict]:
    """仅补分类时：以原 critique 为准保留 point/时间字段，只覆盖类型三字段。"""
    out: list[dict] = []
    for i, src in enumerate(source):
        t = typed[i] if i < len(typed) else {}
        merged = dict(src)
        it = normalize_issue_type(t.get("issue_type"))
        merged["issue_type"] = it
        merged["issue_type_zh"] = ISSUE_TYPE_ZH.get(it, "其它")
        merged["issue_type_rationale"] = str(
            t.get("issue_type_rationale") or merged.get("issue_type_rationale") or ""
        ).strip()
        out.append(merged)
    return validate_and_enrich_typed_items(out)


def validate_and_enrich_typed_items(items: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for item in items:
        raw = str(item.get("issue_type") or "").strip().lower()
        t = normalize_issue_type(raw)
        if raw and raw not in ISSUE_TYPES and raw != "motion":
            item.setdefault(
                "issue_type_rationale",
                "模型未返回合法 issue_type，已归为 other",
            )
        item["issue_type"] = t
        item["issue_type_zh"] = ISSUE_TYPE_ZH.get(t, "其它")
        if not item.get("issue_type_rationale"):
            item["issue_type_rationale"] = ""
        enriched.append(item)
    return enriched


def build_classify_only_prompt(
    *,
    description: str,
    diff_items: list[dict],
    duration_sec: float | None,
    frame_screenshots: int,
) -> str:
    lines = [
        "下面已有「文本 vs 视频」差异列表（point 等字段已写好）。",
        "请**仅**根据所附视频抽帧与文本，为每条补充 issue_type 分类。",
        "不要改写 point、approx_time_span_sec、basis_snapshot_range。",
        "",
        CLASSIFY_GUIDE.strip(),
        "",
        "【文本描述】",
        description.strip() or "（无）",
        "",
        "【已有 differences】",
        json.dumps(diff_items, ensure_ascii=False, indent=2),
        "",
    ]
    if duration_sec is not None and frame_screenshots > 0:
        T, N = duration_sec, frame_screenshots
        lines.append(
            f"时间轴：全片约 {T:.1f} 秒；{N} 张抽帧按时间均匀编号 1..{N}。"
        )
        lines.append("")
    allowed = " | ".join(f'"{t}"' for t in ISSUE_TYPES)
    lines.append(
        "只输出一个 JSON 对象：\n"
        '- "differences"：与输入条数、顺序一致；每条保留原 point 与时间字段，'
        f"并增加 issue_type（{allowed}）、issue_type_zh、issue_type_rationale。\n"
        '- "summary"：可沿用原 summary 或给一句更新后的中文总结。\n'
        "不要其它说明文字。"
    )
    return "\n".join(lines)


async def _call_llm(
    user_prompt: str,
    media_paths: list[str],
    args: argparse.Namespace,
) -> dict | None:
    import ace_i2v_qwen35_397b_a17b_track1_seedance2_hoi as ace

    return await ace.call_qwen_json(
        user_prompt,
        media_paths=media_paths,
        temperature=float(args.temperature),
        max_retries=args.max_retries,
        timeout=float(args.timeout),
        max_tokens=int(args.max_output_tokens),
        yunwu_text_first=True,
    )


async def _prepare_media(
    args: argparse.Namespace,
    video_path: Path,
) -> tuple[list[str], tempfile.TemporaryDirectory | None, int]:
    frame_n = max(0, int(args.frame_screenshots))
    tmp = (
        tempfile.TemporaryDirectory(prefix="critique_typed_frames_")
        if frame_n > 0
        else None
    )
    tdir = Path(tmp.__enter__()) if tmp else None
    media_paths: list[str] = []
    if args.image:
        media_paths.append(str(Path(args.image).expanduser().resolve()))
    if frame_n > 0:
        assert tdir is not None
        frames = gvc.extract_video_frames_jpg(video_path, frame_n, tdir)
        media_paths.extend(str(p) for p in frames)
        print(f"[info] 抽帧：共 {len(frames)} 张 JPEG", file=sys.stderr)
    else:
        media_paths.append(str(video_path))
    return media_paths, tmp, frame_n


async def _run_full_typed(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    prompt_body = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    dur = gvc.probe_video_duration_sec(video_path)
    if dur is not None:
        print(f"[info] video duration ≈ {dur:.1f}s", file=sys.stderr)

    tmp: tempfile.TemporaryDirectory | None = None
    try:
        media_paths, tmp, frame_n = await _prepare_media(args, video_path)
        user_prompt = build_user_prompt_typed(
            prompt_body,
            has_image=bool(args.image),
            duration_sec=dur,
            frame_screenshots=frame_n,
        )
        result = await _call_llm(user_prompt, media_paths, args)
    finally:
        if tmp is not None:
            tmp.__exit__(None, None, None)

    return _finish(result, args)


async def _run_classify_only(args: argparse.Namespace) -> int:
    critique_path = Path(args.critique_json).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    data = json.loads(critique_path.read_text(encoding="utf-8"))
    diff_items = normalize_differences(data)
    if not diff_items:
        print("❌ critique JSON 中 differences 为空", file=sys.stderr)
        return 1

    description = ""
    if args.prompt_file:
        pf = Path(args.prompt_file).expanduser().resolve()
        if pf.is_file():
            description = pf.read_text(encoding="utf-8").strip()

    dur = gvc.probe_video_duration_sec(video_path)
    if dur is not None:
        print(f"[info] video duration ≈ {dur:.1f}s", file=sys.stderr)

    tmp: tempfile.TemporaryDirectory | None = None
    try:
        media_paths, tmp, frame_n = await _prepare_media(args, video_path)
        user_prompt = build_classify_only_prompt(
            description=description,
            diff_items=diff_items,
            duration_sec=dur,
            frame_screenshots=frame_n,
        )
        result = await _call_llm(user_prompt, media_paths, args)
    finally:
        if tmp is not None:
            tmp.__exit__(None, None, None)

    if result and "differences" in result:
        result["differences"] = merge_typed_with_source(
            normalize_differences(result),
            diff_items,
        )
        if not result.get("summary") and data.get("summary"):
            result["summary"] = data["summary"]
        result["source_critique_json"] = public_path(critique_path)
        result["classification_mode"] = "annotate_existing_critique"

    return _finish(result, args)


def _finish(result: dict | None, args: argparse.Namespace) -> int:
    if not result:
        print("❌ 未得到 JSON（检查密钥与网络）", file=sys.stderr)
        return 1

    if "differences" in result:
        result["differences"] = validate_and_enrich_typed_items(
            normalize_differences(result)
        )
    if not args.critique_json:
        result["classification_mode"] = "full_typed_critique"

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\n✅ 已写入 {out_path}", file=sys.stderr)
    return 0


async def _run(args: argparse.Namespace) -> int:
    if args.critique_json:
        return await _run_classify_only(args)
    if not args.prompt_file:
        raise SystemExit("请提供 --prompt-file，或使用 --critique-json 仅补分类")
    return await _run_full_typed(args)


def main() -> None:
    p = argparse.ArgumentParser(
        description="视频+文本 critique，并为每条 difference 标注 issue_type。"
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="全文案；从头生成时必需；仅 --critique-json 模式时可选（辅助分类）",
    )
    p.add_argument(
        "--critique-json",
        type=Path,
        default=None,
        help="已有 gemini_video_prompt_critique 输出；仅补 issue_type，不重写 point",
    )
    p.add_argument("--image", type=Path, default=None)
    p.add_argument(
        "--frame-screenshots",
        type=int,
        default=gvc._default_frame_screenshots_from_env(),
        metavar="N",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.environ.get("CRITIQUE_TYPED_MAX_OUTPUT_TOKENS", "8192")),
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"找不到视频: {args.video}")
    if args.prompt_file and not args.prompt_file.is_file():
        raise SystemExit(f"找不到 prompt 文件: {args.prompt_file}")
    if args.critique_json and not args.critique_json.is_file():
        raise SystemExit(f"找不到 critique JSON: {args.critique_json}")
    if args.image and not args.image.is_file():
        raise SystemExit(f"找不到参考图: {args.image}")
    if args.frame_screenshots < 0:
        raise SystemExit("--frame-screenshots 不能为负数")

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
