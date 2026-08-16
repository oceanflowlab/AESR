"""Shared issue_type constants for typed video–prompt critique and HOI edit pipeline."""

from __future__ import annotations

ISSUE_TYPES = (
    "missing_visual_element",
    "motion_state",
    "motion_process",
    "other",
)

# 需要走 GPT-Image 修关键帧 → edited_frame 作为 Seedance 附图
FRAME_FIX_ISSUE_TYPES = frozenset({"missing_visual_element", "motion_state"})

# 仅写入 Seedance 编辑 prompt（速度/节奏/过程），不修帧
MOTION_PROMPT_ONLY_TYPES = frozenset({"motion_process"})

# 旧版 critique JSON 中的 motion → 保守视为过程问题（与改前行为一致）
LEGACY_ISSUE_TYPE_ALIASES: dict[str, str] = {
    "motion": "motion_process",
}

ISSUE_TYPE_ZH: dict[str, str] = {
    "missing_visual_element": "缺失视觉元素",
    "motion_state": "运动终态不符",
    "motion_process": "运动过程不符",
    "other": "其它",
}


def normalize_issue_type(raw: str | None) -> str:
    t = str(raw or "").strip().lower()
    if t in LEGACY_ISSUE_TYPE_ALIASES:
        t = LEGACY_ISSUE_TYPE_ALIASES[t]
    if t in ISSUE_TYPES:
        return t
    return "other"


def needs_frame_fix(issue_type: str | None) -> bool:
    return normalize_issue_type(issue_type) in FRAME_FIX_ISSUE_TYPES


def is_motion_prompt_only(issue_type: str | None) -> bool:
    return normalize_issue_type(issue_type) in MOTION_PROMPT_ONLY_TYPES
