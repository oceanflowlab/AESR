"""
Doubao Seedance 2.0 提示词指南（编辑视频 / 多模态参考）句式模板。

来源：火山方舟《Doubao Seedance 2.0 系列提示词指南》PDF（编辑视频章节）及
项目内 gemini_seedance_edit_prompt_from_critique.SEEDANCE20_EDIT_GUIDE。

用法：assemble_seedance_edit_prompt_with_frame_fixes_v2.py 按 operation 选用模板。

注意：下列 EN/ZH 为**交给 Seedance/方舟的正文**模板，勿写入 GPT-Image、issue_XXX、
keyframe 等流水线元数据（那些只留在 JSON 元信息里）。
"""

from __future__ import annotations

# --- 成片风格（步骤 3 写入 Seedance 正文；步骤 2 修帧图仍为铅笔）---
PHOTOREAL_VIDEO_RULE_EN = (
    "Image 1 (pencil) and Image 2..N edited keyframes (pencil, if any) are layout/identity "
    "references only. The edited Video 1 output MUST be photorealistic live-action with natural "
    "lighting and realistic skin, fabric, and materials—never pencil sketch, line art, cartoon, "
    "or illustration unless the user caption explicitly demands that aesthetic."
)
PHOTOREAL_VIDEO_RULE_ZH = (
    "@图片1（铅笔）及@图片2..N修帧图（若有，亦为铅笔）仅作身份/构图/姿态参考；"
    "编辑后的@视频1成片必须是逼真实拍电影感（自然光照、真实肤质与材质），"
    "不得输出铅笔素描、线稿、卡通或插画风格（除非用户原文明确要求该画风）。"
)

# --- 任务开场：主体锚点 + 明确「编辑视频」而非参考生视频 ---
OPENING_EN = (
    "Referencing Image 1 for the subject identity and appearance, strictly edit Video 1. "
    "Do not treat this as reference-to-video generation; edit the existing Video 1 only. "
    + PHOTOREAL_VIDEO_RULE_EN
)
OPENING_ZH = (
    "参考@图片1中人物的身份与外观，严格编辑@视频1（勿当作参考生视频，仅在原片上修改）。"
    + PHOTOREAL_VIDEO_RULE_ZH
)

# --- 编辑视频 · 官方推荐句式（PDF）---
ADD_SEGMENT_EN = (
    "During {span}s, add to Video 1 the missing visual elements shown in {image_label}: {target}. "
    "Integrate naturally with Video 1 lighting, perspective, and scene continuity. "
    "Render as photorealistic live-action in Video 1, not pencil or illustration."
)
ADD_SEGMENT_ZH = (
    "在{span}s内，对@视频1增加元素，画面参考{image_label}（铅笔仅为构图参考）：{target}。"
    "与@视频1原有光照、透视自然融合；成片为逼真实拍，非铅笔/插画。"
)

MODIFY_SEGMENT_EN = (
    "During {span}s, strictly edit Video 1 and change the incorrect visuals to match {image_label}: {target}. "
    "Output photorealistic live-action in Video 1, not pencil-sketch style."
)
MODIFY_SEGMENT_ZH = (
    "在{span}s内，严格编辑@视频1，将其中的错误画面修改为与{image_label}（铅笔构图参考）一致：{target}。"
    "成片为逼真实拍，非铅笔素描。"
)

DELETE_SEGMENT_EN = (
    "During {span}s, remove {target} from Video 1; keep all other unmentioned content unchanged."
)

MOTION_SEGMENT_EN = (
    "During {span}s, modify Video 1 motion: change the subject's closed lips to natural, "
    "continuous speaking mouth movement with subtle, smooth articulation. "
    "Keep photorealistic live-action rendering."
)
MOTION_SEGMENT_ZH = (
    "在{span}s内，修改@视频1中的动作/运动：将人物闭合的嘴唇改为自然、连续的说话口型，"
    "幅度适中、过渡平滑；保持逼真实拍画质。"
)

CLOSING_EN = (
    "Preserve Video 1's original single primary camera movement, scene layout, lighting, "
    "and color grading for all unmentioned content. "
    "Final Video 1 must be photorealistic live-action cinematic quality—no pencil sketch, "
    "line art, cartoon, or illustration. No subtitles, no logos, no watermarks."
)
CLOSING_ZH = (
    "未提及部分保持@视频1原有单一主运镜、场景布局与光照色调。"
    "成片必须为逼真实拍电影感，禁止铅笔素描/线稿/卡通/插画画风。"
    "避免生成字幕、Logo、水印。"
)

ARK_MEDIA_NOTE = (
    "Downstream Ark call (default): text + Image 1 (pencil reference_image, identity only) + Video 1. "
    "GPT-Image edited keyframes (Image 2..N) are pencil in the pipeline for pose/composition reference; "
    "the Seedance prompt must still require photorealistic video output. "
    "Optional upload of Image 2..N via seedance2_edit_from_prompt_json_v2.py."
)

# Gemini 步骤 3 系统说明（勿整段抄进 seedance_edit_prompt 正文）
STEP3_LLM_PHOTOREAL_INSTRUCTION = """
【成片风格 · 硬性要求（步骤 3 写 Seedance 指令时须遵守）】
- 步骤 2 产出的 edited_frame.png 为**铅笔素描**，仅用于理解构图、姿态、缺失元素布局。
- 你写的 seedance_edit_prompt_en / seedance_edit_prompt_zh 必须要求：**编辑后的视频 1 为逼真实拍**，
  与 Video 1 原片的光影、材质一致；参考图不得把成片画成铅笔/线稿/卡通。
- 从铅笔修帧图提炼的是**真实世界应出现的物体、动作、构图**，正文中用自然语言描述实拍画面，
  并显式写 photorealistic / live-action / 逼真实拍 等约束；勿写「保持铅笔风格」「素描画风」。
"""

# 写入 Gemini 任务说明（非 Seedance 正文）
SEEDANCE_PROMPT_FORBIDDEN_PHRASES = (
    "Do NOT put pipeline jargon in seedance_edit_prompt_en or seedance_edit_prompt_zh, e.g. "
    "GPT-Image, keyframe fix, issue_000_edited, aligned near, manifest, edit_prompt_en. "
    "Only use Video 1, Image 1, Image 2…, time spans, and plain visual descriptions."
)
