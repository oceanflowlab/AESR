"""
Track1 变体（基于 ace_i2v_qwen35_397b_a17b_IPT2V.py）：人物 ID 保持的图生视频 ACE 流水线。

推荐两阶段（同一 `--playbook-file` 工作副本）：
1) 全量更新 Playbook：`--mode warmup_only --tasks-from track1_seedance_pencil`（不要加 `--limit`），
   使用 `out_seedance2_videos_same_crop` 下全部「有 video_prompt + 有 mp4」的 id，与 `out_gpt_image2_pencil_same_crop/id*/pencil.png` 对齐后逐条分析失败样本并 merge 进 Playbook。
2) 增强前 20 个 id 的 prompt：`--mode enhance_prompt_only --limit 20`（id 按数字序取前 20），
   结果写入各 `id*/video_prompt_ace_enhanced.txt`。

- 默认 Playbook：`playbooks/playbook_final.json`（由 HOI+document 初始版在 Track 1 上迭代得到的比赛最终版本）。
- `enhance_prompt_only` + `track1_seedance_pencil`：增强结果写入各 `id*/video_prompt_ace_enhanced.txt`（不覆盖 `video_prompt.txt`）。
- `--tasks-from track1_seedance_pencil`：instruction = `out_seedance2_videos_same_crop/id*/video_prompt.txt`；参考图 = `out_gpt_image2_pencil_same_crop/id*/`（默认优先 `pencil.png`）。
- Epoch0 参考 mp4：与 prompt 同根的 `id*/id*.mp4`；请将 `--original-video-dir` 指向 `out_seedance2_videos_same_crop`。

test50 仍可用：`--tasks-from test50_selected` 等与原脚本一致。默认 ACE/I2V 与环境变量行为同 IPT2V 原版。
"""
import os
import json
import mimetypes
import time
import re
import shutil
import base64
import requests
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from http import HTTPStatus
from urllib.parse import urlparse
from tqdm import tqdm

# --- 导入 API 库 ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

import dashscope
from dashscope import VideoSynthesis, MultiModalConversation

# --- Gemini：直连 Google 用 GEMINI_MODEL_NAME；经云雾时 chat 模型 id 与 VLM 批处理脚本一致 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_gemini_model_env = os.getenv("GEMINI_MODEL_NAME", "").strip()
GEMINI_MODEL_NAME = _gemini_model_env or "gemini-3.1-pro-preview"
YUNWU_ACE_CHAT_MODEL = (
    os.getenv("VLM_SCORER_MODEL", "").strip() or _gemini_model_env or "gemini-3.1-pro-preview"
).strip()
gemini_client: Optional[Any] = None
if genai and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as _e:
        gemini_client = None
        print(f"❗ 无法初始化 Gemini（Google）客户端: {_e}")

# --- 配置 Qwen 多模态 API (角色 1, 2, 3) ---
# 模型名：DashScope 与云雾后台「可用模型」列表需一致；云雾若 slug 不同请设环境变量 QWEN_MODEL_NAME。
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3.5-397b-a17b")
QWEN_VIDEO_FPS = 2
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# --- 云雾 YUNWU（OpenAI 兼容 /v1/chat/completions）：Qwen、或 Gemini（ACE_QWEN_BACKEND=gemini 且 ACE_GEMINI_TRANSPORT=yunwu）---
# YUNWU_API_KEY；图生视频仍见 ACE_I2V_BACKEND（wan=DASHSCOPE，seedance=ARK）。
YUNWU_API_KEY = os.getenv("YUNWU_API_KEY", "").strip()
_YUNWU_BASE_URL = (
    os.getenv("YUNWU_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://yunwu.ai/v1"
).strip()


def normalize_yunwu_base_url(base_url: str) -> str:
    """与 batch_gpt_image2_pencil_copy 一致：站点根目录自动补 /v1。"""
    if not base_url:
        return "https://yunwu.ai/v1"
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if parsed.netloc == "yunwu.ai" and path == "":
        return "https://yunwu.ai/v1"
    return base_url.rstrip("/")


YUNWU_BASE_URL = normalize_yunwu_base_url(_YUNWU_BASE_URL)

# 运行时可由 main() 中的 argparse 覆盖（先于流水线执行）
_RUNTIME_QWEN_BACKEND = os.getenv("ACE_QWEN_BACKEND", "gemini").strip().lower()
_RUNTIME_YUNWU_BASE_URL = YUNWU_BASE_URL
_gt_raw = os.getenv("ACE_GEMINI_TRANSPORT", "yunwu").strip().lower()
_RUNTIME_GEMINI_TRANSPORT = _gt_raw if _gt_raw in ("yunwu", "google") else "yunwu"

# --- I2V 视频生成：Wan（DashScope）或 Seedance 2.0（火山方舟 Ark，与 batch_seedance2_r2v.py 一致）---
# ACE_I2V_BACKEND=seedance 时使用 ARK_API_KEY；=wan 时使用 DASHSCOPE_API_KEY + wan2.2-i2v-flash
_RUNTIME_I2V_BACKEND = os.getenv("ACE_I2V_BACKEND", "seedance").strip().lower()
ARK_API_KEY = os.getenv("ARK_API_KEY", "").strip()
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
SEEDANCE_MODEL = os.getenv("SEEDANCE_MODEL", "doubao-seedance-2-0-260128").strip()
SEEDANCE_PROMPT_MODE = os.getenv("SEEDANCE_PROMPT_MODE", "caption_first").strip()
_RUNTIME_ARK_API_KEY: Optional[str] = None  # main() 中可由 --ark-api-key 覆盖

# --- 配置 DashScope API (Wan 图生视频或 DashScope Qwen) ---
# 北京地域url
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
# 视频下载目录 - 现在在 main() 中动态设置
# GENERATED_VIDEOS_DIR = "generated_videos" 

if _RUNTIME_QWEN_BACKEND == "gemini":
    if _RUNTIME_GEMINI_TRANSPORT == "google" and not GEMINI_API_KEY:
        print("❗ 提示：ACE_GEMINI_TRANSPORT=google 时需 GEMINI_API_KEY（直连 Google）。")
    if _RUNTIME_GEMINI_TRANSPORT == "yunwu" and not YUNWU_API_KEY:
        print("❗ 提示：Gemini 经云雾时需 YUNWU_API_KEY（默认 ACE_GEMINI_TRANSPORT=yunwu）。")
if _RUNTIME_QWEN_BACKEND == "yunwu" and not YUNWU_API_KEY:
    print("❗ 提示：YUNWU_API_KEY 未设置（ACE_QWEN_BACKEND=yunwu 时必需）。")
if _RUNTIME_QWEN_BACKEND == "dashscope" and not DASHSCOPE_API_KEY:
    print("❗ 提示：DASHSCOPE_API_KEY 未设置（ACE_QWEN_BACKEND=dashscope 时必需）。")
if _RUNTIME_I2V_BACKEND == "wan" and not DASHSCOPE_API_KEY:
    print("❗ 提示：DASHSCOPE_API_KEY 未设置（ACE_I2V_BACKEND=wan 图生视频需要）。")
dashscope.api_key = DASHSCOPE_API_KEY
if _RUNTIME_QWEN_BACKEND == "gemini":
    _llm_model_disp = (
        YUNWU_ACE_CHAT_MODEL
        if _RUNTIME_GEMINI_TRANSPORT == "yunwu"
        else GEMINI_MODEL_NAME
    )
else:
    _llm_model_disp = QWEN_MODEL_NAME
_gem_route = ""
if _RUNTIME_QWEN_BACKEND == "gemini":
    _gem_route = f" | Gemini 通道: {_RUNTIME_GEMINI_TRANSPORT}"
print(
    f"✅ 脚本已加载 | ACE LLM 后端: {_RUNTIME_QWEN_BACKEND} | 模型: {_llm_model_disp}{_gem_route} | "
    f"I2V 默认: {_RUNTIME_I2V_BACKEND}（seedance=ARK / wan=DASHSCOPE）"
)

# --- 路径配置 (在 main 中设置) ---
PLAYBOOK_FILE: str = "id_preserving_i2v_playbook_qwen35_397b_a17b.json"  # 将在 main 中被设置为绝对路径

# --- 实用工具函数 ---

def guess_mime_type(file_path):
    """根据文件扩展名推测 MIME 类型。"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if file_path.endswith(('.mp4', '.mov', '.avi', '.webm')):
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "video/mp4"
    return mime_type or "image/jpeg"


def build_gemini_media_part(media_path: str) -> Optional[Any]:
    """图像或视频 → Gemini Part（仅 ACE_GEMINI_TRANSPORT=google 时使用）。"""
    if types is None:
        return None
    if not os.path.exists(media_path):
        print(f"   ❗ 媒体文件不存在: {media_path}")
        return None
    try:
        with open(media_path, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"   ❗ 无法读取媒体文件 {media_path}: {e}")
        return None
    mime_type = guess_mime_type(media_path)
    return types.Part(inline_data=types.Blob(data=data, mime_type=mime_type))


def build_gemini_response_schema(prompt_text: str) -> Dict[str, Any]:
    """按 ACE 提示词片段选择结构化 JSON schema（与 ace_i2v.py 对齐）。"""
    response_schema: Dict[str, Any] = {
        "type": "OBJECT",
        "properties": {
            "reasoning": {"type": "STRING"},
        },
    }
    if "ace_generator_prompt" in prompt_text:
        response_schema["properties"]["enhanced_prompt"] = {"type": "STRING"}
    elif "ace_analysis_prompt" in prompt_text:
        response_schema["properties"]["success"] = {"type": "BOOLEAN"}
        response_schema["properties"]["critique"] = {"type": "STRING"}
    elif "ace_reflector_prompt" in prompt_text:
        response_schema["properties"]["root_cause"] = {"type": "STRING"}
        response_schema["properties"]["key_insight"] = {"type": "STRING"}
    elif "ace_curator_prompt" in prompt_text:
        response_schema["properties"]["operations"] = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING"},
                    "section": {"type": "STRING"},
                    "content": {"type": "STRING"},
                },
            },
        }
    elif "seedance_edit_prompt_schema" in prompt_text:
        return {
            "type": "OBJECT",
            "properties": {
                "seedance_edit_prompt_en": {"type": "STRING"},
                "seedance_edit_prompt_zh": {"type": "STRING"},
                "editing_rationale": {"type": "STRING"},
            },
        }
    return response_schema


def build_dashscope_media_item(media_path: str, video_fps: int = QWEN_VIDEO_FPS) -> Optional[Dict[str, Any]]:
    """将图像或视频文件转换为 DashScope MultiModalConversation 所需的 content item。"""
    if not os.path.exists(media_path):
        print(f"   ❗ 媒体文件不存在: {media_path}")
        return None
    resolved = str(Path(media_path).expanduser().resolve())
    file_uri = f"file://{resolved}"
    mime_type = guess_mime_type(resolved)
    if mime_type.startswith("video/"):
        return {"video": file_uri, "fps": video_fps}
    if mime_type.startswith("image/"):
        return {"image": file_uri}
    print(f"   ❗ 不支持的媒体类型: {media_path} (MIME: {mime_type})")
    return None

def extract_dashscope_text(response: Any) -> str:
    """兼容对象/字典两种返回结构，提取文本。"""
    try:
        content = response.output.choices[0].message.content
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return (first.get("text") or "").strip()
            return str(first).strip()
        return str(content).strip()
    except Exception:
        pass

    try:
        content = response.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return (first.get("text") or "").strip()
            return str(first).strip()
    except Exception:
        pass
    return ""

def file_to_data_url_for_chat(file_path: str) -> Optional[str]:
    """任意图像/视频文件 → data:{mime};base64,...（用于云雾 OpenAI 兼容多模态）。"""
    if not os.path.exists(file_path):
        print(f"   ❗ 媒体文件不存在: {file_path}")
        return None
    resolved = str(Path(file_path).expanduser().resolve())
    mime_type = guess_mime_type(resolved)
    try:
        with open(resolved, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except OSError as e:
        print(f"   ❗ 无法读取文件: {resolved} ({e})")
        return None
    return f"data:{mime_type};base64,{b64}"


def build_yunwu_user_content(
    media_paths: Optional[List[str]],
    prompt_text: str,
    *,
    text_first: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """OpenAI 风格 user message content：image_url / video_url + text。

    text_first=True 时先文本后媒体（部分网关/模型对「先说明任务再贴图」更稳）。
    """
    media_blocks: List[Dict[str, Any]] = []
    for path in media_paths or []:
        data_url = file_to_data_url_for_chat(path)
        if not data_url:
            return None
        resolved = str(Path(path).expanduser().resolve())
        mime_type = guess_mime_type(resolved)
        if mime_type.startswith("image/"):
            media_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        elif mime_type.startswith("video/"):
            media_blocks.append({"type": "video_url", "video_url": {"url": data_url}})
        else:
            print(f"   ❗ 不支持的媒体类型（云雾）: {path} (MIME: {mime_type})")
            return None
    text_block = {"type": "text", "text": prompt_text}
    if text_first:
        return [text_block] + media_blocks
    return media_blocks + [text_block]


def extract_openai_chat_text(resp_json: Dict[str, Any]) -> str:
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
                elif "text" in block:
                    texts.append(str(block["text"]))
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts).strip()
    return str(content).strip() if content else ""


def call_qwen_yunwu_chat_sync(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_content: List[Dict[str, Any]],
    temperature: float,
    timeout: float,
    max_tokens: int = 8192,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """OpenAI 兼容 POST；默认不使用 HTTP(S)_PROXY（避免本地失效代理导致 ProxyError）。
    若必须通过系统代理访问云雾：export ACE_QWEN_HTTP_TRUST_ENV=1
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    trust_proxy_env = os.getenv("ACE_QWEN_HTTP_TRUST_ENV", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        session = requests.Session()
        session.trust_env = trust_proxy_env
        r = session.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, str(e)
    try:
        data = r.json()
    except json.JSONDecodeError:
        return None, f"HTTP {r.status_code}, body (non-json): {r.text[:300]!r}"
    if r.status_code != 200:
        err = data.get("error") if isinstance(data, dict) else data
        return None, f"HTTP {r.status_code}: {err}"
    return data, None


def encode_file_base64(file_path: str) -> str:
    """(DashScope) 格式为 data:{MIME_type};base64,{base64_data}"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

def download_video(video_url: str, save_path: Path) -> bool:
    """(DashScope) 从URL下载视频到本地"""
    try:
        print(f'  ... 正在下载视频到: {save_path}')
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = save_path.stat().st_size / (1024 * 1024)  # MB
        print(f'  ✓ 视频下载完成，大小: {file_size:.2f} MB')
        return True
    except Exception as e:
        print(f'  ✗ 下载视频失败: {str(e)}')
        return False


def _find_seedance_video_urls(value: Any) -> List[str]:
    """与 batch_seedance2_r2v.find_video_urls 一致，从 Ark 返回结构中提取 mp4 URL。"""
    urls: List[str] = []
    if isinstance(value, str):
        if value.startswith(("http://", "https://")) and any(
            token in value.lower() for token in [".mp4", ".mov", "video"]
        ):
            urls.append(value)
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(_find_seedance_video_urls(item))
        return urls
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                if "video" in key_lower or any(ext in item.lower() for ext in [".mp4", ".mov"]):
                    urls.append(item)
            urls.extend(_find_seedance_video_urls(item))
    return urls


def build_seedance_video_prompt(caption: str, prompt_mode: str) -> str:
    """与 batch_seedance2_r2v.build_video_prompt 一致；caption 此处为 ACE 增强后的英文 prompt。"""
    if prompt_mode == "raw":
        return caption
    if prompt_mode == "strong_reference":
        return (
            "Based on Image 1, generate a photorealistic live-action video (cinematic natural lighting, realistic skin and fabric); never output cartoon, anime, cel-shaded, or pencil-sketch aesthetics unless the caption explicitly asks for that style. "
            "Use the reference image as the character appearance reference, preserving the person's identity and facial features as much as possible. "
            "Unless the caption explicitly requires a different head pose (profile, looking away, etc.), keep face yaw and pitch broadly similar to the reference—face-embedding metrics (e.g. ArcFace) favor pose consistency with the reference image. "
            "Motion should be smooth and physically plausible; avoid jittery or inconsistent limb/face deformation across frames. "
            "If the reference is a sketch or illustration, render the character as a real human in live footage; do not retain line-art or comic rendering in the final video. "
            "The character's action, clothing, role identity, scene, and atmosphere should follow this caption exactly: "
            f"{caption}"
        )
    if prompt_mode == "caption_first":
        return (
            "Use image 1 as the character appearance reference, preserving the person's facial features and identity. "
            "Output must be photorealistic live-action video (natural lighting, realistic textures). Do not produce cartoon, anime, simplified illustration, or pencil-sketch look unless the caption explicitly requests that aesthetic. "
            "Unless the caption explicitly requires profile view, turning away, or back shots, keep head pose and face orientation broadly similar to the reference image (helps ArcFace-style identity similarity). "
            "You may change framing, background, and narrative motion; avoid freezing the entire clip as a single static frame. "
            "Keep motion smooth and coherent over time; prefer natural human movement without severe jitter or limb-face warping unless the caption asks for stylized motion. "
            "If the reference is a sketch or stylized drawing, still render the subject as a believable real person on camera. Generate a continuous video following this caption. "
            f"{caption}"
        )
    raise ValueError(f"Unknown Seedance prompt mode: {prompt_mode}")


def _seedance_call_with_retries(label: str, func, max_retries: int, retry_wait: int):
    attempt = 0
    wait_seconds = retry_wait
    while True:
        try:
            return func()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                print(f"   ✗ {label}: 已达最大重试 {max_retries} 次，最后错误: {type(exc).__name__}: {exc}")
                raise
            print(
                f"   [retry] {label}: {type(exc).__name__}: {exc} | "
                f"wait {wait_seconds}s ({attempt}/{max_retries})"
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2


def run_video_generation_seedance(image_path: str, enhanced_prompt: str, output_dir: Path) -> Optional[str]:
    """
    Seedance 2.0（火山方舟 Ark content_generation），与 batch_seedance2_r2v.py 对齐。
    参考图使用本地文件 base64 data URL（无需 GitHub raw）。
    """
    print("   ...(真实) 视频生成 (Seedance 2.0 / Ark) 启动...")
    try:
        from volcenginesdkarkruntime import Ark
    except ModuleNotFoundError:
        print(
            "   ✗ 缺少依赖：请先执行 conda activate aesr，并运行 python -m pip install -r requirements.txt"
        )
        return None

    api_key = (_RUNTIME_ARK_API_KEY or ARK_API_KEY).strip()
    if not api_key:
        print("   ✗ ARK_API_KEY 未设置（或为 Seedance 传入 --ark-api-key）")
        return None

    base_url = os.getenv("ARK_BASE_URL", ARK_BASE_URL).strip() or ARK_BASE_URL
    model = os.getenv("SEEDANCE_MODEL", SEEDANCE_MODEL).strip() or SEEDANCE_MODEL
    prompt_mode = os.getenv("SEEDANCE_PROMPT_MODE", SEEDANCE_PROMPT_MODE).strip() or "caption_first"
    ratio = os.getenv("SEEDANCE_RATIO", "16:9")
    resolution = os.getenv("SEEDANCE_RESOLUTION", "480p")
    duration = int(os.getenv("SEEDANCE_DURATION", "4"))
    generate_audio = os.getenv("SEEDANCE_GENERATE_AUDIO", "").lower() in ("1", "true", "yes")
    watermark = os.getenv("SEEDANCE_WATERMARK", "").lower() in ("1", "true", "yes")
    poll_interval = int(os.getenv("SEEDANCE_POLL_INTERVAL", "30"))
    timeout_sec = int(os.getenv("SEEDANCE_TIMEOUT", "1800"))
    max_retries = int(os.getenv("SEEDANCE_MAX_RETRIES", "5"))
    retry_wait = int(os.getenv("SEEDANCE_RETRY_WAIT", "30"))

    try:
        image_url = encode_file_base64(image_path)
        video_prompt = build_seedance_video_prompt(enhanced_prompt.strip(), prompt_mode)
        print(f'   ... 提交 Seedance - 图像: {Path(image_path).name}')
        print(f'   ... Prompt 前缀模式: {prompt_mode} | 片段: {video_prompt[:120]}...')

        client = Ark(base_url=base_url, api_key=api_key)

        def _create():
            return client.content_generation.tasks.create(
                model=model,
                content=[
                    {"type": "text", "text": video_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "role": "reference_image",
                    },
                ],
                generate_audio=generate_audio,
                ratio=ratio,
                resolution=resolution,
                duration=duration,
                watermark=watermark,
            )

        create_result = _seedance_call_with_retries(
            "seedance create",
            _create,
            max_retries,
            retry_wait,
        )
        task_id = create_result.id
        print(f'   ... task_id: {task_id}，轮询等待完成...')

        started = time.time()
        while True:
            result = _seedance_call_with_retries(
                f"seedance get {task_id}",
                lambda: client.content_generation.tasks.get(task_id=task_id),
                max_retries,
                retry_wait,
            )
            status = getattr(result, "status", None)
            print(f"   ... status={status}")
            if status == "succeeded":
                if hasattr(result, "model_dump"):
                    plain = result.model_dump()
                elif hasattr(result, "dict"):
                    plain = result.dict()
                elif isinstance(result, dict):
                    plain = result
                else:
                    plain = getattr(result, "__dict__", {})
                video_urls = _find_seedance_video_urls(plain)
                if not video_urls:
                    print("   ✗ 任务成功但未解析到视频 URL")
                    return None
                video_url = video_urls[0]
                print(f"   ✓ video_url: {video_url[:80]}...")
                stem_out = video_export_stem_from_image_path(image_path)
                safe_tid = str(task_id).replace("/", "_")[:16]
                video_filename = f"{stem_out}_{safe_tid}.mp4"
                video_local_path = output_dir / video_filename
                if download_video(video_url, video_local_path):
                    return str(video_local_path)
                return None
            if status == "failed":
                err = getattr(result, "error", result)
                print(f"   ✗ Seedance 任务失败: {err}")
                return None
            if time.time() - started > timeout_sec:
                print(f"   ✗ 等待 Seedance 超时（{timeout_sec}s）")
                return None
            time.sleep(poll_interval)

    except Exception as e:
        print(f"   ✗ Seedance 视频生成出错: {e}")
        return None


def run_video_generation_wan(image_path: str, enhanced_prompt: str, output_dir: Path) -> Optional[str]:
    """DashScope wan2.2-i2v-flash（原逻辑）。"""
    print("   ...(真实) 视频生成 (DashScope Wan) 启动...")
    try:
        img_url_base64 = encode_file_base64(image_path)

        prompt = (
            "The reference image defines which person to preserve: keep the same identity "
            "(face, body proportions, hairstyle, and outfit unless the caption explicitly changes attire or setting). "
            "Follow the caption for actions, events, and scene; camera angle and framing are flexible unless the caption asks for a specific shot. "
            "For face-embedding evaluation (e.g. ArcFace-style), similarity is higher when the face yaw/pitch in the video stays close to the reference; "
            "unless the caption demands profile view, turning away, or back shots, prefer keeping a similar head pose and visible frontal or three-quarter face as in the reference. "
            "Prefer smooth, temporally coherent motion with plausible human movement; avoid contradictory actions that cause jitter or unnatural warping. "
            "Photorealistic live-action output only: natural lighting and textures; do not render as cartoon, anime, pencil-sketch, or illustration unless the caption explicitly demands that style. "
            + enhanced_prompt.strip()
            + " Avoid switching to a different identifiable person mid-video."
        )
        print(f'   ... 提交任务 - 图像: {Path(image_path).name}')
        print(f'   ... 最终 Prompt: {prompt[:70]}...')

        rsp = VideoSynthesis.async_call(
            model='wan2.2-i2v-flash',
            prompt=prompt,
            img_url=img_url_base64,
            resolution="720P",
            duration=5,
            prompt_extend=True,
            watermark=False,
            negative_prompt=(
                "cartoon, anime, illustration, pencil sketch, line art, line drawing, "
                "cel shading, 2D, comic book, water color painting, flat vector"
            ),
        )

        if rsp.status_code != HTTPStatus.OK:
            print(f"   ✗ 提交任务失败: {rsp.code}, {rsp.message}")
            return None

        task_id = rsp.output.task_id
        print(f'   ... 任务已提交，task_id: {task_id}，等待完成...')

        rsp = VideoSynthesis.wait(rsp)

        if rsp.status_code == HTTPStatus.OK:
            video_url = rsp.output.video_url
            print(f"   ✓ 任务成功！video_url: {video_url}")

            videos_dir = output_dir
            stem_out = video_export_stem_from_image_path(image_path)
            video_filename = f"{stem_out}_{task_id[:8]}.mp4"
            video_local_path = videos_dir / video_filename

            if download_video(video_url, video_local_path):
                return str(video_local_path)
            print("   ✗ 视频下载失败。")
            return None
        print(f'   ✗ 等待任务完成失败: {rsp.code}, {rsp.message}')
        return None

    except Exception as e:
        print(f"   ✗ 视频生成过程中出错: {str(e)}")
        return None


def run_video_generation(image_path: str, enhanced_prompt: str, output_dir: Path) -> Optional[str]:
    """根据 ACE_I2V_BACKEND 选择 Wan 或 Seedance 2.0。"""
    if _RUNTIME_I2V_BACKEND == "seedance":
        return run_video_generation_seedance(image_path, enhanced_prompt, output_dir)
    return run_video_generation_wan(image_path, enhanced_prompt, output_dir)

async def call_qwen_json(
    prompt_text: str,
    media_paths: Optional[List[str]] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    timeout: float = 120.0,
    max_tokens: int = 8192,
    yunwu_text_first: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    调用 ACE 多模态 LLM 并返回 JSON 结果。

    - ACE_QWEN_BACKEND=gemini：默认经云雾（ACE_GEMINI_TRANSPORT=yunwu，与 Qwen 同一条 OpenAI 接口）；
      设 ACE_GEMINI_TRANSPORT=google 时直连 Google（需 GEMINI_API_KEY、google-genai）。
    - yunwu / dashscope：同前（Qwen）。
    - max_tokens：云雾 chat 输出上限；长 JSON 评判可调大（如 32768）以免截断断尾。
    - yunwu_text_first：仅云雾 OpenAI 兼容路径；为 True 时 user content 为「文本块在前、媒体块在后」。
    """
    backend = _RUNTIME_QWEN_BACKEND
    use_gemini_google = backend == "gemini" and _RUNTIME_GEMINI_TRANSPORT == "google"
    use_yunwu_openai = (backend == "yunwu") or (
        backend == "gemini" and _RUNTIME_GEMINI_TRANSPORT == "yunwu"
    )
    api_label = "LLM API"
    if use_yunwu_openai:
        api_label = "云雾 API"
    elif backend == "dashscope":
        api_label = "DashScope API"
    elif use_gemini_google:
        api_label = "Gemini API（Google）"

    if backend == "yunwu":
        if not YUNWU_API_KEY:
            print("   ❗ YUNWU_API_KEY 未设置（ACE_QWEN_BACKEND=yunwu 时需要）")
            return None
    elif backend == "dashscope":
        if not DASHSCOPE_API_KEY:
            print("   ❗ DASHSCOPE_API_KEY 未设置")
            return None
    elif backend == "gemini":
        if _RUNTIME_GEMINI_TRANSPORT == "yunwu":
            if not YUNWU_API_KEY:
                print("   ❗ YUNWU_API_KEY 未设置（Gemini 经云雾时需要）")
                return None
        else:
            if not GEMINI_API_KEY:
                print("   ❗ GEMINI_API_KEY 未设置（ACE_GEMINI_TRANSPORT=google 时需要）")
                return None
            if gemini_client is None:
                print("   ❗ Gemini 客户端未初始化（请安装 google-genai 并设置 GEMINI_API_KEY）")
                return None
    else:
        print(f"   ❗ 未知 ACE_QWEN_BACKEND: {backend}")
        return None

    content_items: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []
    yunwu_user_content: Optional[List[Dict[str, Any]]] = None
    yunwu_chat_model: Optional[str] = None
    gemini_contents: Optional[List[Any]] = None
    gemini_config: Optional[Any] = None

    if backend == "yunwu":
        yunwu_user_content = build_yunwu_user_content(
            media_paths, prompt_text, text_first=yunwu_text_first
        )
        if yunwu_user_content is None:
            return None
        yunwu_chat_model = QWEN_MODEL_NAME
    elif backend == "gemini" and _RUNTIME_GEMINI_TRANSPORT == "yunwu":
        yunwu_user_content = build_yunwu_user_content(
            media_paths, prompt_text, text_first=yunwu_text_first
        )
        if yunwu_user_content is None:
            return None
        yunwu_chat_model = YUNWU_ACE_CHAT_MODEL
    elif backend == "dashscope":
        if media_paths:
            for path in media_paths:
                media_item = build_dashscope_media_item(path)
                if media_item:
                    content_items.append(media_item)
                else:
                    return None
        content_items.append({"text": prompt_text})
        messages = [{"role": "user", "content": content_items}]
    elif use_gemini_google:
        if types is None or gemini_client is None:
            print("   ❗ google-genai 未安装或客户端未初始化")
            return None
        parts = [types.Part(text=prompt_text)]
        if media_paths:
            for path in media_paths:
                part = build_gemini_media_part(path)
                if part:
                    parts.append(part)
                else:
                    return None
        gemini_contents = [types.Content(role="user", parts=parts)]
        gemini_config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=build_gemini_response_schema(prompt_text),
        )

    # 重试机制
    for attempt in range(max_retries):
        try:
            text_content = ""
            if yunwu_user_content is not None and yunwu_chat_model is not None:
                def _chat_yunwu():
                    return call_qwen_yunwu_chat_sync(
                        api_key=YUNWU_API_KEY,
                        base_url=_RUNTIME_YUNWU_BASE_URL,
                        model=yunwu_chat_model,
                        user_content=yunwu_user_content,
                        temperature=temperature,
                        timeout=timeout,
                        max_tokens=max_tokens,
                    )

                response_data, err = await asyncio.wait_for(
                    asyncio.to_thread(_chat_yunwu),
                    timeout=timeout + 30.0,
                )
                if err:
                    raise RuntimeError(err)
                text_content = extract_openai_chat_text(response_data or {})
            elif use_gemini_google and gemini_contents is not None and gemini_config is not None:

                def _gen_gemini():
                    return gemini_client.models.generate_content(
                        model=GEMINI_MODEL_NAME,
                        contents=gemini_contents,
                        config=gemini_config,
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(_gen_gemini),
                    timeout=timeout,
                )
                text_content = (response.text or "").strip()
            elif backend == "dashscope":

                def _chat():
                    return MultiModalConversation.call(
                        api_key=DASHSCOPE_API_KEY,
                        model=QWEN_MODEL_NAME,
                        messages=messages,
                        temperature=temperature,
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(_chat),
                    timeout=timeout
                )

                text_content = extract_dashscope_text(response)
            else:
                raise RuntimeError(f"未实现的 ACE LLM 后端: {backend}")

            if not text_content:
                print(f"   ❗ {api_label} 返回为空")
                return None

            if text_content.startswith("```"):
                text_content = re.sub(r"```(json)?\n", "", text_content, 1)
                text_content = re.sub(r"\n```$", "", text_content, 1)

            return json.loads(text_content)

        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避：2秒, 4秒, 6秒
                print(f"   ⚠️ 请求超时（尝试 {attempt + 1}/{max_retries}），{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                print(f"   ❗ {api_label} 请求超时（已重试 {max_retries} 次）")
                return None
        except json.JSONDecodeError as e:
            snippet = text_content[:200].replace("\n", " ") if "text_content" in locals() else ""
            print(f"   ❗ 解析 LLM 响应失败: {e} | 片段: {snippet!r}")
            lead = snippet.lstrip()
            if lead and lead[0] not in "{[":
                print(
                    "   💡 响应不像 JSON（多为自然语言）。若 Gemini 经云雾且模型称「看不到视频」，"
                    "网关可能未转发 video_url；可试 ACE_GEMINI_TRANSPORT=google，"
                    "或对 gemini_video_prompt_critique 使用 --frame-screenshots。"
                )
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(
                    f"   ⚠️ 多为 JSON 被截断或未转义换行；{wait_time}s 后重试 "
                    f"(尝试 {attempt + 1}/{max_retries}，max_tokens={max_tokens})…"
                )
                await asyncio.sleep(wait_time)
                continue
            print("   ❗ 已达最大重试；可调大 call_qwen_json(..., max_tokens=) 或缩短模型输出要求。")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"   ⚠️ 调用 {api_label} 失败（尝试 {attempt + 1}/{max_retries}）: {e}，{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                print(f"   ❗ 调用 {api_label} 失败（已重试 {max_retries} 次）: {e}")
                return None

    return None

# --- Playbook 管理 (ACE 论文中的非 LLM 逻辑) ---

def load_playbook(filepath: str = PLAYBOOK_FILE) -> Dict[str, List[str]]:
    """加载 Playbook，如果不存在则创建默认结构。"""
    default_playbook = {
        "strategies": [
            "[strategy-001]: 视频 prompt 必须先重申参考人物的稳定外观锚点（脸型、发型、体态），再展开文本描述中的动作与事件。",
            "[strategy-002]: 若文案未要求侧身/扭头/背影等大角度变化，应写明保持与参考图相近的脸部朝向与可见五官角度（ArcFace 等人脸嵌入评测在 pose 接近参考时更易得高分）。",
            "[strategy-003]: 写清动作的节拍与时序（先做什么、再做什么），并约束「连贯、符合物理与人体常识的运动」：优先平滑的位移与姿态过渡，避免含糊动词导致模型生成抖动或肢体不合理形变。",
            "[strategy-004]: 成片默认真实影像风格：在 enhanced_prompt 中明确要求 photorealistic / live-action / natural lighting and skin texture；若参考图是素描/线稿，必须写明「真人实拍质感、不得保留线稿或卡通渲染」。",
        ],
        "templates": [
            "[template-001]: 与参考图同一人物；按下列描述发生：{事件与时间顺序}；外观除文案明确要求外与参考一致；脸部朝向尽量延续参考图的朝向除非文案另有要求；动作连贯、运动幅度与节奏合理；画面为真实风格实拍影像（非卡通、非简笔画）。"
        ],
        "pitfalls": [
            "[pitfall-001]: 陷阱：文字太笼统导致模型擅自换人或换场景。反思：把「谁、做什么、何时」写具体，并重复身份锚点词。",
            "[pitfall-002]: 陷阱：prompt 鼓励大幅度转头或长时间侧脸/背影，在人脸嵌入评测中易与参考 mis-align。反思：除非用户描述需要，否则约束头部以小幅度运动为主并多为正对或微侧。",
            "[pitfall-003]: 陷阱：堆砌特效词导致面部拉丝、肢体扭曲或非物理抖动。反思：明确「smooth natural motion」「temporally coherent」，限制同时发生的互相矛盾的动作指令。",
            "[pitfall-004]: 陷阱：未禁止非真实画风导致成片仍为素描边线、赛璐璐或插画质感。反思：显式写 no cartoon / anime / pencil-sketch look unless user asks；强调 cinematic photoreal。",
        ]
    }
    if not os.path.exists(filepath):
        print(f"   📘 Playbook 文件未找到，在 '{filepath}' 创建新的。")
        save_playbook(filepath, default_playbook)
        return default_playbook
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"   ❗ Playbook 文件 '{filepath}' 损坏，将使用默认 Playbook。")
        return default_playbook
    except Exception as e:
        print(f"   ❗ 加载 Playbook 失败: {e}")
        return default_playbook

def save_playbook(filepath: str, playbook: Dict[str, List[str]]):
    """将 Playbook 保存到文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(playbook, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   ❗ 保存 Playbook 失败: {e}")

_PLAYBOOK_SECTIONS = frozenset({"strategies", "templates", "pitfalls"})


def merge_playbook(playbook: Dict[str, List[str]], delta_update: Dict[str, Any]) -> bool:
    """(角色 4: Merger) ACE 增量更新，防止“上下文崩塌”。"""
    updated = False
    if "operations" not in delta_update:
        return False

    for op in delta_update.get("operations", []):
        if not isinstance(op, dict):
            continue
        op_type = op.get("type")
        if op_type != "ADD":
            if op_type is not None:
                print(f"   ⚠️ Curator 使用了未支持的 type={op_type!r}（仅支持 ADD），已跳过。")
            continue

        section = op.get("section")
        content = op.get("content")
        if isinstance(section, str):
            section = section.strip()
            if section not in _PLAYBOOK_SECTIONS:
                section_key = section.lower()
                if section_key in _PLAYBOOK_SECTIONS:
                    section = section_key
        if not section or section not in _PLAYBOOK_SECTIONS:
            print(
                "   ⚠️ Curator 的 ADD 缺少合法 section（须为 strategies / templates / pitfalls），已跳过本条。"
            )
            continue
        if not content or not str(content).strip():
            print("   ⚠️ Curator 的 ADD 缺少 content，已跳过本条。")
            continue
        content = str(content).strip()

        if section not in playbook:
            print(f"   ❗ 当前 Playbook 缺少板块 {section!r}，无法写入。")
            continue
        if content not in playbook[section]:
            playbook[section].append(content)
            print(f"   📘 Playbook 已更新 (板块: {section}): {content[:80]}...")
            updated = True
        else:
            print(f"   📘 (跳过冗余更新: {content[:50]}...)")

    return updated

# --- ACE 角色实现 ---

async def ace_generator(
    image_path: str, 
    instruction: str, 
    playbook: Dict[str, List[str]]
) -> Optional[Dict[str, Any]]:
    """(角色 1: Generator) 读取 Playbook 和用户输入，生成增强的 prompt。"""
    # print("--- 角色 1: Generator 启动 ---") # 减少日志噪音
    playbook_text = json.dumps(playbook, indent=2, ensure_ascii=False)
    
    prompt = f"""
    ace_generator_prompt
    任务：为**人物 ID 保持类参考图生视频**（reference image-to-video）写一条可直接给图生视频模型用的**英文** enhanced prompt。
    已附上人物参考图；下面是用户的简单视频描述与 Playbook。请综合三者：人物身份与参考图一致、视频内容与用户描述一致；写法与约束优先遵从 Playbook。
    若 Playbook 中条目冲突，以方括号内**编号更大**的条目为准（视为较新、后更新的规则）。

    [人物参考图像]: (已附上)
    [用户视频描述 / 简单指令]: "{instruction}"
    [Playbook 手册]:
    {playbook_text}

    请以 JSON 返回：必须包含字段 enhanced_prompt（最终英文视频 prompt）；其余字段可任选，用于简要记录推理。
    """
    
    return await call_qwen_json(
        prompt, 
        media_paths=[image_path], 
        temperature=0.5,
        timeout=180.0  # 处理图像需要更长时间，增加到3分钟
    )

async def ace_reflector(
    analysis_report: Dict[str, Any], 
    failed_prompt: str
) -> Optional[Dict[str, Any]]:
    """(角色 2b: Reflector) 将“特定的”失败报告“抽象”为“通用的”洞察。"""
    # print("--- 角色 2b: Reflector 启动 ---") # 减少日志噪音
    report_text = json.dumps(analysis_report, indent=2, ensure_ascii=False)
    
    prompt = f"""
    ace_reflector_prompt
    你是人物 ID 保持视频任务的分析师：从失败案例中提炼**可写入 Playbook 的通用原则**（身份一致性、与文本描述对齐、人脸朝向与嵌入评测、运动合理性、**成片须为真实实拍风格而非素描/卡通**）。
    
    [背景]:
    Generator 用参考人物图与文本描述生成了视频；评估模型给出了失败分析。

    [相关的 Prompt（可能不完美）]:
    "{failed_prompt}"

    [评估模型的分析报告（具体失败）]:
    {report_text}

    [你的任务]:
    1.  阅读报告，弄清失败属于「身份漂移 / 换脸换人」「未执行描述中的动作或事件」「擅自添加或遗漏关键语义」「人脸相对参考图朝向偏离过大（文案却未要求该姿态）」「运动不合理」「画面仍为简笔画、线稿、卡通、动漫或插画等非实拍风格（用户未要求）」等哪类问题。
    2.  **抽象**：背后的通用原则是什么？下次写 prompt 或 Playbook 该怎样避免？
    3.  输出中**不要**出现可识别的具体人物外貌细节或固定道具、具体地名（保持泛化）。
    4.  给出 root_cause 与 key_insight（JSON 字段名沿用这两项）。

    [示例]:
    - 具体失败: "视频中人物五官与参考图明显不是同一人。"
    - 抽象结果:
      - root_cause: "生成 prompt 未持续重申身份锚点，模型默认生成匿名面孔。"
      - key_insight: "在叙事展开前先并列列出若干身份锚点形容词，并在句末再次重复「同一人」约束。"

    请以 JSON 格式返回你的分析。
    """
    
    return await call_qwen_json(prompt, temperature=0.3)

async def ace_curator(
    insights: Dict[str, Any], 
    playbook: Dict[str, List[str]]
) -> Optional[Dict[str, Any]]:
    """(角色 3: Curator) 将“通用的”洞察转化为“增量的” Playbook 更新。"""
    # print("--- 角色 3: Curator 启动 ---") # 减少日志噪音
    insights_text = json.dumps(insights, indent=2, ensure_ascii=False)
    playbook_text = json.dumps(playbook, indent=2, ensure_ascii=False)

    prompt = f"""
    ace_curator_prompt
    你是一个知识库的“策展人”。你的工作是维护一个 Playbook，确保它只包含高质量、非冗余的条目。

    [输入洞察]:
    {insights_text}

    [当前 Playbook]:
    {playbook_text}

    [你的任务]:
    1.  阅读“key_insight”，是一条关于「身份一致性」「与描述对齐」「脸部朝向与嵌入」「运动连贯」或「强制 photoreal、禁止卡通/线稿风」的通用建议（勿收录纯机位号、景别堆砌；与 pose、motion、style、embeddings 相关的可操作条目应保留）。
    2.  **检查冗余**：仔细阅读“当前 Playbook”，判断该洞察是否已存在同类表述？
    3.  **若为新洞察**：
        a.  改写为简洁、可执行的条目。
        b.  归入 strategies / templates / pitfalls **三者之一**（不得使用其它板块名）。
        c.  仅使用下面规定的 **ADD** 操作格式（不要输出 DELETE/REPLACE 等其它 type）。
    4.  **若冗余**："operations" 必须为 []。

    [重要]: 仅输出增量 Delta Update，绝不重写整本 Playbook。

    [输出 JSON 必须满足]:
    - 顶层可含 "reasoning"（字符串）与 "operations"（数组）。
    - "operations" 中**每一项**只能是下面结构（section 三选一，必须小写英文）:
      {{"type": "ADD", "section": "strategies", "content": "单条新条目全文"}}
      或 section 为 "templates" / "pitfalls"（同上结构，仅 section 取值不同）。
    - "section" 只能是这三个英文小写键名之一，**不得**为 null、空串、中文或别名字符串。
    - 无增量时: "operations": []。

    请只输出一个 JSON 对象，不要其它说明文字。
    """
    
    return await call_qwen_json(prompt, temperature=0.1)

async def run_automated_analysis(
    image_path: str,
    instruction: str, 
    enhanced_prompt: str, 
    video_path: str
) -> Optional[Dict[str, Any]]:
    """
    (真实角色 2a) 您的 LLM 分析模型 (Qwen3.5)。
    """
    # print("--- 角色 2a: 自动分析模型 (Qwen3.5) 启动 ---") # 减少日志噪音
    
    prompt = f"""
    ace_analysis_prompt
    你是人物 ID 保持类图生视频的评审。判定视频是否在**身份一致性**、**与用户意图对齐**、**运动与时序合理性**、**视觉风格是否符合真实实拍**四方面综合达标。
    不要求与原图背景、构图、机位一致；除非描述明确要求，否则不因「换了背景」判失败。
    **人脸嵌入评测提示**：常见管线（如 ArcFace）对「视频中人脸朝向与参考图的差异」敏感——朝向相差越大，嵌入相似度往往越低。评审时请考虑：若用户描述**未要求**侧身、扭头、背影、大俯仰角，而视频中长时间呈现与参考差异很大的脸角度或大量不可见正脸，可作为身份一致性维度的负面因素（在 critique 中说明）。
    **视觉风格**：默认要求成片为**真实影像 / 实拍质感**（自然光影、材质可信）。若参考图是素描或插画，视频仍应呈现真人实拍感而非延续线稿。**若用户描述明确要求**动漫、插画、实验风格等，则以文案为准。

    [上下文]:
    1.  [人物参考图像]: (已附上) 用于比对「是否仍是同一人」及大致的脸部朝向。
    2.  [用户描述 / 原始指令]: "{instruction}"
    3.  [增强 Prompt]: "{enhanced_prompt}"
    4.  [待评视频]: (已附上)

    [你的任务]:
    - **身份一致性**：视频中主角是否与参考图像指向同一人（五官、体态、发型等）；是否明显换脸、顶替或漂移。**若文案未要求改变视角**，视频中人脸朝向是否与参考过于不一致（不利 ArcFace 类指标）可作为扣分考量。
    - **描述对齐**：视频是否落实视频描述（及增强 prompt）中的动作、事件或场景语义；是否严重跑题、静止无叙事、或与文案矛盾。，则以文案为准，不因与参考朝向不同判失败。视频相机运镜动作是否与描述一致
    - **运动合理性**：在时间维度上动作是否连贯；是否符合人体运动与简单物理直觉；是否出现严重抖动、面部或肢体异常扭曲、融化拉丝、穿模、节奏混乱等非自然 motion。
    - **视觉风格（实拍）**：整体观感是否为真实世界影像。**若用户描述未要求**卡通、动漫、简笔画线稿、赛璐璐或平面插画风，而成片明显保留此类非实拍风格，则本维度判为不达标；参考图为素描时，成片仍应以实拍人物呈现而非描线填色风。
    四方面均无明显硬伤则 success 为 true；任一维度严重不达标则为 false。

    请以 JSON 返回：success（布尔）、critique（简短中文或英文说明侧重点与失败类型）。
    """
    
    return await call_qwen_json(
        prompt, 
        media_paths=[image_path, video_path], # 发送原始图像和"生成的视频"
        temperature=0.1, # 分析任务需要高确定性
        timeout=180.0  # 处理图像+视频需要更长时间，增加到3分钟
    )

# --- 【新】数据加载 ---

def load_tasks_from_json(json_path: Path) -> List[Dict[str, str]]:
    """从您提供的 JSON 文件中加载任务列表。"""
    if not json_path.exists():
        print(f"❗ 错误：JSON 任务文件未找到: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tasks = []
        for image_name, meta in data.items():
            instruction = meta.get('instruction')
            if instruction: # 只处理有指令的条目
                tasks.append({
                    'image_name': image_name,
                    'instruction': instruction
                })
        
        print(f"✅ 从 {json_path.name} 加载了 {len(tasks)} 个任务。")
        return tasks
    except Exception as e:
        print(f"❗ 加载 JSON 任务失败: {e}")
        return []


def video_export_stem_from_image_path(image_path: str) -> str:
    """导出 mp4 文件名前缀：id004/pencil.png → id004_pencil，避免多 id 仅用笔名冲突。"""
    p = Path(image_path).resolve()
    if p.parent.name and re.match(r"^id\d+$", p.parent.name):
        return f"{p.parent.name}_{p.stem}"
    return p.stem


def discover_id_subdirs(root: Path) -> List[str]:
    """列出根目录下 id001、id004 等子目录名。"""
    if not root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and re.match(r"^id\d+$", p.name):
            out.append(p.name)
    return out


def resolve_video_in_id_folder(folder: Path) -> Optional[Path]:
    """在同一 id 目录内解析 mp4：优先 {id}.mp4，否则取任意 .mp4。"""
    if not folder.is_dir():
        return None
    stem = folder.name
    exact = folder / f"{stem}.mp4"
    if exact.is_file():
        return exact
    mp4s = sorted(folder.glob("*.mp4"))
    if mp4s:
        return mp4s[0]
    return None


def find_reference_image_under_id(
    source_prompt_root: Path, id_name: str, image_candidates: Tuple[str, ...]
) -> Optional[Path]:
    """在 source_prompt_root/idXXX/ 下按候选文件名找第一张存在的参考图。"""
    id_dir = source_prompt_root / id_name
    if not id_dir.is_dir():
        return None
    for cand in image_candidates:
        p = id_dir / cand
        if p.is_file():
            return p
    return None


def instruction_from_selected_entry(meta: Dict[str, Any]) -> Optional[str]:
    """从 selected_prompts.json 单条 entries[id] 取出用于训练的原始文案。"""
    pfiles = meta.get("prompt_files") or []
    if not pfiles:
        return None
    pm = meta.get("prompts") or {}
    text = (pm.get(pfiles[0]) or "").strip()
    return text or None


def load_tasks_from_selected_prompts(
    source_prompt_root: Path,
    selected_prompts_json: Path,
    image_candidates: Tuple[str, ...],
) -> List[Dict[str, str]]:
    """
    从 test_50_mixed/selected_prompts.json 构建任务：instruction 为选定原始 prompt；
    image_name 为相对 source_prompt_root 的路径（如 id004/pencil.png）。
    """
    if not selected_prompts_json.is_file():
        print(f"❗ selected_prompts.json 未找到: {selected_prompts_json}")
        return []
    try:
        with open(selected_prompts_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❗ 读取 selected_prompts 失败: {e}")
        return []

    entries = data.get("entries") or {}
    tasks: List[Dict[str, str]] = []
    for id_name in sorted(entries.keys()):
        meta = entries[id_name]
        instruction = instruction_from_selected_entry(meta)
        if not instruction:
            continue
        img_path = find_reference_image_under_id(source_prompt_root, id_name, image_candidates)
        if not img_path:
            print(f"   ⚠️ {id_name}: 未找到参考图（候选 {image_candidates}），跳过")
            continue
        try:
            rel = img_path.relative_to(source_prompt_root.resolve())
        except ValueError:
            rel = Path(id_name) / img_path.name
        tasks.append({"image_name": str(rel).replace("\\", "/"), "instruction": instruction})

    print(f"✅ 从 selected_prompts 加载 {len(tasks)} 个任务（图根目录: {source_prompt_root}）")
    return tasks


def _track1_id_numeric_sort_key(id_name: str) -> Tuple[int, str]:
    """id001 / id020 → 按数字 1、20 排序，避免纯字符串序异常。"""
    m = re.match(r"^id(\d+)$", id_name)
    if m:
        return (int(m.group(1)), id_name)
    return (10**12, id_name)


def load_tasks_track1_seedance_prompt_pencil_image(
    prompt_video_root: Path,
    pencil_image_root: Path,
    image_candidates: Tuple[str, ...],
) -> List[Dict[str, str]]:
    """
    Track1：每个 id 目录下，用 video_prompt.txt 作为 instruction；
    参考图在 pencil_image_root/id*/ 下按 image_candidates 选第一张存在的文件。
    image_name 为相对 pencil_image_root 的路径（如 id162/pencil.png）。
    """
    if not prompt_video_root.is_dir():
        print(f"❗ prompt 根目录不存在: {prompt_video_root}")
        return []
    if not pencil_image_root.is_dir():
        print(f"❗ 参考图根目录不存在: {pencil_image_root}")
        return []

    ids_prompt = set(discover_id_subdirs(prompt_video_root))
    ids_img = set(discover_id_subdirs(pencil_image_root))
    common = sorted(ids_prompt & ids_img, key=_track1_id_numeric_sort_key)
    missing_vid = sorted(ids_img - ids_prompt, key=_track1_id_numeric_sort_key)
    missing_img = sorted(ids_prompt - ids_img, key=_track1_id_numeric_sort_key)
    if missing_vid:
        print(f"   ℹ️ 有 {len(missing_vid)} 个 id 仅有铅笔图、无 Seedance prompt 目录，已跳过（示例: {missing_vid[:3]}…）")
    if missing_img:
        print(f"   ℹ️ 有 {len(missing_img)} 个 id 仅有 prompt 目录、无铅笔图目录，已跳过（示例: {missing_img[:3]}…）")

    tasks: List[Dict[str, str]] = []
    for id_name in common:
        vp = prompt_video_root / id_name / "video_prompt.txt"
        if not vp.is_file():
            continue
        instruction = vp.read_text(encoding="utf-8").strip()
        if not instruction:
            print(f"   ⚠️ {id_name}: video_prompt.txt 为空，跳过")
            continue
        img_path = find_reference_image_under_id(pencil_image_root, id_name, image_candidates)
        if not img_path:
            print(f"   ⚠️ {id_name}: 未找到参考图（候选 {image_candidates}），跳过")
            continue
        try:
            rel = img_path.relative_to(pencil_image_root.resolve())
        except ValueError:
            rel = Path(id_name) / img_path.name
        tasks.append({"image_name": str(rel).replace("\\", "/"), "instruction": instruction})

    print(
        f"✅ Track1 任务 {len(tasks)} 条（id 按数字序；prompt: {prompt_video_root.name}/id*/video_prompt.txt | 图: {pencil_image_root.name}）"
    )
    return tasks


def parse_skip_ids(skip_ids_str: Optional[str]) -> Set[str]:
    if not skip_ids_str or not str(skip_ids_str).strip():
        return set()
    return {p.strip() for p in str(skip_ids_str).split(",") if p.strip()}


def parse_comma_separated_paths(raw: Optional[str], base: Path) -> List[Path]:
    """逗号分隔路径列表，相对路径相对 base。"""
    if not raw or not str(raw).strip():
        return []
    out: List[Path] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        q = Path(p).expanduser()
        if not q.is_absolute():
            q = (base / q).resolve()
        else:
            q = q.resolve()
        out.append(q)
    return out


async def run_epoch_0_warmup_multi_roots(
    playbook: Dict[str, List[str]],
    source_prompt_root: Path,
    selected_prompts_json: Path,
    warmup_roots: List[Path],
    image_candidates: Tuple[str, ...],
    limit: Optional[int],
) -> Dict[str, List[str]]:
    """
    从多个「每 id 一文件夹」的视频库中学习 Playbook：同一原始 prompt（selected_prompts）
    + test_50_mixed 参考图 + 各库中的 mp4 与 video_prompt.txt。
    不生成新视频。
    """
    if not selected_prompts_json.is_file():
        print(f"❗ 需要 selected_prompts.json: {selected_prompts_json}")
        return playbook
    try:
        with open(selected_prompts_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❗ 读取 selected_prompts 失败: {e}")
        return playbook

    entries = data.get("entries") or {}
    print(f"\n--- 🚀 Epoch 0 多目录预热（{len(warmup_roots)} 个视频根）---")
    print(f"   原始 prompt / 图: {source_prompt_root} + {selected_prompts_json.name}")

    processed = 0
    for ref_root in warmup_roots:
        if not ref_root.is_dir():
            print(f"   ⚠️ 跳过不存在的目录: {ref_root}")
            continue
        label = ref_root.name
        print(f"   📂 参考库: {label}")
        for id_name in tqdm(
            discover_id_subdirs(ref_root), desc=f"Epoch0 [{label}]", leave=False
        ):
            if id_name not in entries:
                continue
            instruction = instruction_from_selected_entry(entries[id_name])
            if not instruction:
                continue
            img_path = find_reference_image_under_id(source_prompt_root, id_name, image_candidates)
            if not img_path:
                continue
            folder = ref_root / id_name
            video_path = resolve_video_in_id_folder(folder)
            if not video_path:
                continue

            vp = folder / "video_prompt.txt"
            enhanced_line = (
                vp.read_text(encoding="utf-8").strip()
                if vp.is_file()
                else "N/A (Original Video)"
            )

            analysis_report = await run_automated_analysis(
                str(img_path),
                instruction,
                enhanced_line,
                str(video_path),
            )
            if not analysis_report:
                continue
            if analysis_report.get("success", False):
                continue
            insights = await ace_reflector(analysis_report, enhanced_line)
            if not insights or "key_insight" not in insights:
                continue
            delta_update = await ace_curator(insights, playbook)
            if not delta_update or "operations" not in delta_update:
                continue
            merge_playbook(playbook, delta_update)

            processed += 1
            if limit is not None and processed >= limit:
                break
        if limit is not None and processed >= limit:
            break

    print(f"--- ✅ Epoch 0 多目录预热完成（已处理失败样本学习约 {processed} 条）---")
    save_playbook(PLAYBOOK_FILE, playbook)
    print(f"   ✅ Playbook V0 已保存到 {PLAYBOOK_FILE}")
    return playbook


def has_existing_video(image_name: str, video_dir: Optional[Path]) -> bool:
    """检查指定目录中是否已经为该图像生成过视频。"""
    if not video_dir or not video_dir.exists():
        return False
    ip = Path(image_name)
    stem_key = (
        f"{ip.parent.name}_{ip.stem}"
        if ip.parent.name and re.match(r"^id\d+$", ip.parent.name)
        else ip.stem
    )
    return any(video_dir.glob(f"{stem_key}_*.mp4"))


def resolve_reference_video(image_name: str, original_video_dir: Path) -> Optional[Path]:
    """
    在参考视频目录中定位与图像对应的 mp4：
    - 若 image_name 为 id162/pencil.png 且 original_video_dir 下存在 id162/ 子目录，
      则在 id162/ 内解析 mp4（与 out_seedance2_videos_same_crop 布局一致）。
    - 否则：优先 {stem}.mp4，否则匹配 {stem}_*.mp4（与 wan2.2 等带 hash 后缀的导出一致）。
    """
    if not original_video_dir.is_dir():
        return None
    ip = Path(image_name)
    if ip.parent.name and re.match(r"^id\d+$", ip.parent.name):
        nested = original_video_dir / ip.parent.name
        v_nested = resolve_video_in_id_folder(nested)
        if v_nested is not None:
            return v_nested
    stem = ip.stem
    exact = original_video_dir / f"{stem}.mp4"
    if exact.is_file():
        return exact
    matches = sorted(original_video_dir.glob(f"{stem}_*.mp4"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"   ⚠️ 多个参考视频匹配 {stem}，使用 {matches[0].name}")
        return matches[0]
    return None

async def generate_videos_from_playbook(
    tasks: List[Dict[str, str]],
    playbook: Dict,
    image_dir: Path,
    output_video_dir: Path,
    *,
    exclude_video_dir: Optional[Path] = None,
    skip_if_exists: bool = True,
) -> Dict[str, int]:
    """使用已有的 Playbook 直接生成视频，不对 Playbook 进行更新。"""
    print(f"\n--- 🚀 使用现有 Playbook 直接生成视频 (共 {len(tasks)} 个任务) 🚀 ---")
    print(f"   (输出目录: {output_video_dir})")
    if exclude_video_dir:
        print(f"   (排除已有视频目录: {exclude_video_dir})")

    output_video_dir.mkdir(parents=True, exist_ok=True)

    summary = {"generated": 0, "skipped": 0, "failed": 0}

    for task in tqdm(tasks, desc="Direct Generation"):
        image_name = task["image_name"]
        instruction = task["instruction"]
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            summary["failed"] += 1
            continue

        if exclude_video_dir and has_existing_video(image_name, exclude_video_dir):
            summary["skipped"] += 1
            continue

        if skip_if_exists and has_existing_video(image_name, output_video_dir):
            summary["skipped"] += 1
            continue

        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            summary["failed"] += 1
            continue

        enhanced_prompt = generator_output["enhanced_prompt"]
        video_path = run_video_generation(str(image_path), enhanced_prompt, output_video_dir)
        if not video_path:
            print(f"   (视频生成失败: {image_name})")
            summary["failed"] += 1
            continue

        summary["generated"] += 1

    print(f"--- ✅ 直接生成完成 ---")
    print(f"   已生成: {summary['generated']}")
    print(f"   已跳过: {summary['skipped']}")
    print(f"   失败: {summary['failed']}")
    return summary


def _extract_enhanced_prompt_field(payload: Dict[str, Any]) -> Optional[str]:
    """Generator 返回 JSON 中取出增强 prompt（兼容多种字段名）。"""
    if not payload:
        return None
    for key in ("enhanced_prompt", "final_prompt", "prompt", "video_prompt"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _discover_enhance_id_folders(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and re.match(r"^id\d+$", p.name):
            out.append(p)
    return out


def _find_enhance_reference_image(folder: Path, names: Tuple[str, ...]) -> Optional[Path]:
    for name in names:
        cand = folder / name
        if cand.is_file():
            return cand
    return None


async def run_enhance_prompt_only_folder(
    root: Path,
    playbook: Dict[str, List[str]],
    instruction_file: str,
    image_candidates: Tuple[str, ...],
    force: bool,
    output_txt_name: str,
    output_json_name: str,
    limit: Optional[int],
    skip_ids: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """每个 id* 子目录：参考图 + instruction 文件 → 写入增强 prompt（无视频）。"""
    skip_ids = skip_ids or set()
    folders = sorted(
        _discover_enhance_id_folders(root),
        key=lambda p: _track1_id_numeric_sort_key(p.name),
    )
    if limit is not None:
        folders = folders[:limit]
    stats = {"ok": 0, "skipped": 0, "failed": 0}
    if not folders:
        print(f"❗ 在 {root} 下未发现 id* 子目录")
        return stats

    for folder in tqdm(folders, desc="Enhance prompt (folders)"):
        if folder.name in skip_ids:
            stats["skipped"] += 1
            continue
        out_txt = folder / output_txt_name
        out_json = folder / output_json_name
        if not force and out_txt.is_file() and out_txt.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        img = _find_enhance_reference_image(folder, image_candidates)
        if not img:
            print(f"   (跳过 {folder.name}: 未找到参考图 {image_candidates})")
            stats["failed"] += 1
            continue

        instr_path = folder / instruction_file
        if not instr_path.is_file():
            print(f"   (跳过 {folder.name}: 缺少 {instruction_file})")
            stats["failed"] += 1
            continue

        instruction = instr_path.read_text(encoding="utf-8").strip()
        if not instruction:
            stats["failed"] += 1
            continue

        raw = await ace_generator(str(img.resolve()), instruction, playbook)
        if not raw:
            stats["failed"] += 1
            continue

        enhanced = _extract_enhanced_prompt_field(raw)
        out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        if not enhanced:
            print(f"   (⚠️ {folder.name}: 返回 JSON 无 enhanced_prompt 等字段，已保存 {output_json_name})")
            stats["failed"] += 1
            continue

        out_txt.write_text(enhanced + "\n", encoding="utf-8")
        stats["ok"] += 1

    print(
        f"--- ✅ Prompt 增强完成（目录模式）--- ok={stats['ok']} skipped={stats['skipped']} failed={stats['failed']} ---"
    )
    return stats


async def run_enhance_prompt_only_tasks(
    tasks: List[Dict[str, str]],
    playbook: Dict[str, List[str]],
    image_dir: Path,
    output_dir: Optional[Path],
    force: bool,
    label: str,
) -> Dict[str, int]:
    """JSON 任务列表：每张图 + instruction → 写出增强 prompt。"""
    stats = {"ok": 0, "skipped": 0, "failed": 0}
    for task in tqdm(tasks, desc=f"Enhance prompt ({label})"):
        image_name = task["image_name"]
        instruction = task["instruction"]
        image_path = image_dir / image_name
        stem_key = (
            video_export_stem_from_image_path(str(image_path))
            if image_path.is_file()
            else Path(image_name).stem
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_txt = output_dir / f"{stem_key}_enhanced_i2v_prompt.txt"
            out_json = output_dir / f"{stem_key}_enhanced_i2v_generator.json"
        else:
            out_txt = image_dir / f"{stem_key}_enhanced_i2v_prompt.txt"
            out_json = image_dir / f"{stem_key}_enhanced_i2v_generator.json"

        if not force and out_txt.is_file() and out_txt.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        if not image_path.is_file():
            print(f"   (跳过 {image_name}: 图像不存在)")
            stats["failed"] += 1
            continue

        raw = await ace_generator(str(image_path.resolve()), instruction, playbook)
        if not raw:
            stats["failed"] += 1
            continue

        out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        enhanced = _extract_enhanced_prompt_field(raw)
        if not enhanced:
            print(f"   (⚠️ {image_name}: 无 enhanced_prompt 字段，已保存 generator JSON)")
            stats["failed"] += 1
            continue

        out_txt.write_text(enhanced + "\n", encoding="utf-8")
        stats["ok"] += 1

    print(
        f"--- ✅ Prompt 增强完成 [{label}] --- ok={stats['ok']} skipped={stats['skipped']} failed={stats['failed']} ---"
    )
    return stats


async def run_enhance_track1_tasks_write_under_prompt_root(
    tasks: List[Dict[str, str]],
    playbook: Dict[str, List[str]],
    image_dir: Path,
    prompt_root: Path,
    force: bool,
    *,
    txt_name: str = "video_prompt_ace_enhanced.txt",
    json_name: str = "video_prompt_ace_enhanced.generator.json",
    skip_ids: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """
    Track1：增强结果写入 prompt_root/id*/ ，与 video_prompt.txt 同目录（不覆盖原文件）。
    """
    skip_ids = skip_ids or set()
    stats = {"ok": 0, "skipped": 0, "failed": 0}
    for task in tqdm(tasks, desc="Enhance prompt (track1 → prompt dir)"):
        image_name = task["image_name"]
        instruction = task["instruction"]
        image_path = image_dir / image_name
        id_name = Path(image_name).parent.name
        if id_name in skip_ids:
            stats["skipped"] += 1
            continue
        if not (id_name and re.match(r"^id\d+$", id_name)):
            print(f"   (跳过 {image_name}: 无法解析 id 子目录)")
            stats["failed"] += 1
            continue
        out_dir = prompt_root / id_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_txt = out_dir / txt_name
        out_json = out_dir / json_name

        if not force and out_txt.is_file() and out_txt.stat().st_size > 0:
            stats["skipped"] += 1
            continue
        if not image_path.is_file():
            print(f"   (跳过 {image_name}: 图像不存在)")
            stats["failed"] += 1
            continue

        raw = await ace_generator(str(image_path.resolve()), instruction, playbook)
        if not raw:
            stats["failed"] += 1
            continue
        out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        enhanced = _extract_enhanced_prompt_field(raw)
        if not enhanced:
            print(f"   (⚠️ {image_name}: 无 enhanced_prompt 字段，已保存 {json_name})")
            stats["failed"] += 1
            continue
        out_txt.write_text(enhanced + "\n", encoding="utf-8")
        stats["ok"] += 1

    print(
        f"--- ✅ Track1 Prompt 增强（写入各 id 目录）--- ok={stats['ok']} skipped={stats['skipped']} failed={stats['failed']} ---"
    )
    return stats


# --- 【新】多轮 (Multi-Epoch) 编排 ---

async def run_epoch_0_warmup(tasks: List[Dict[str, str]], playbook: Dict, image_dir: Path, original_video_dir: Path) -> Dict:
    """第 0 轮：从您已有的视频中学习，构建 Playbook V0。"""
    print(f"\n--- 🚀 开始 第 0 轮 (预热, {len(tasks)} 个样本) 🚀 ---")
    print(f"   (从已有视频中学习: {original_video_dir})")
    
    for task in tqdm(tasks, desc="Epoch 0 (Warmup)"):
        image_name = task['image_name']
        instruction = task['instruction']
        
        image_path = image_dir / image_name
        video_path = resolve_reference_video(image_name, original_video_dir)

        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            continue
        if not video_path:
            print(f"   (跳过 {image_name}: 原始视频未找到)")
            continue

        # 1. (角色 2a) 分析现有视频
        analysis_report = await run_automated_analysis(
            str(image_path), 
            instruction, 
            "N/A (Original Video)", # 原始视频没有增强 prompt
            str(video_path)
        )

        if not analysis_report:
            print(f"   (分析失败: {image_name})")
            continue
        
        # 2. 如果失败，则学习
        if not analysis_report.get("success", False):
            # print(f"   ⚠️ 发现失败案例: {image_name}")
            # 3. (角色 2b) 抽象洞察
            insights = await ace_reflector(analysis_report, "N/A (Original Video)")
            if not insights or "key_insight" not in insights:
                continue
            
            # 4. (角色 3) 策划更新
            delta_update = await ace_curator(insights, playbook)
            if not delta_update or "operations" not in delta_update:
                continue

            # 5. (角色 4) 合并 (在内存中)
            merge_playbook(playbook, delta_update)
    
    print(f"--- ✅ 第 0 轮 (预热) 完成 ---")
    save_playbook(PLAYBOOK_FILE, playbook)
    print(f"   ✅ Playbook V0 已保存到 {PLAYBOOK_FILE}")
    return playbook

async def run_epoch_1_learn(tasks: List[Dict[str, str]], playbook: Dict, image_dir: Path, output_video_dir: Path) -> Dict:
    """第 1 轮：使用 V0 Playbook 生成 V1 视频，并学习 V1 的失败，构建 Playbook V1。"""
    print(f"\n--- 🚀 开始 第 1 轮 (学习, {len(tasks)} 个样本) 🚀 ---")
    print(f"   (新视频将保存到: {output_video_dir})")
    output_video_dir.mkdir(parents=True, exist_ok=True)
    
    for task in tqdm(tasks, desc="Epoch 1 (Learn)"):
        image_name = task['image_name']
        instruction = task['instruction']
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            continue

        # 1. (角色 1) 生成增强 Prompt
        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            continue
        enhanced_prompt = generator_output["enhanced_prompt"]

        # 2. (真实) 生成 V1 视频
        video_path_v1 = run_video_generation(str(image_path), enhanced_prompt, output_video_dir)
        if not video_path_v1:
            print(f"   (视频生成失败: {image_name})")
            continue

        # 3. (角色 2a) 分析 V1 视频
        analysis_report = await run_automated_analysis(
            str(image_path), 
            instruction, 
            enhanced_prompt, 
            video_path_v1
        )
        if not analysis_report:
            print(f"   (分析失败: {image_name})")
            continue

        # 4. 如果 V1 失败，则学习
        if not analysis_report.get("success", False):
            # print(f"   ⚠️ 发现失败案例 (V1): {image_name}")
            insights = await ace_reflector(analysis_report, enhanced_prompt)
            if not insights or "key_insight" not in insights:
                continue
            
            delta_update = await ace_curator(insights, playbook)
            if not delta_update or "operations" not in delta_update:
                continue
            
            # 实时更新 Playbook (在内存中)
            merge_playbook(playbook, delta_update)

    print(f"--- ✅ 第 1 轮 (学习) 完成 ---")
    save_playbook(PLAYBOOK_FILE, playbook)
    print(f"   ✅ Playbook V1 已保存到 {PLAYBOOK_FILE}")
    return playbook

async def run_epoch_2_final(tasks: List[Dict[str, str]], playbook: Dict, image_dir: Path, output_video_dir: Path):
    """第 2 轮：使用 V1 Playbook 生成最终的 V2 视频。"""
    print(f"\n--- 🚀 开始 第 2 轮 (最终生成, {len(tasks)} 个样本) 🚀 ---")
    print(f"   (最终视频将保存到: {output_video_dir})")
    output_video_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for task in tqdm(tasks, desc="Epoch 2 (Final)"):
        image_name = task['image_name']
        instruction = task['instruction']
        image_path = image_dir / image_name
        
        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            fail_count += 1
            continue
            
        # 1. (角色 1) 生成最终 Prompt
        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            fail_count += 1
            continue
        enhanced_prompt = generator_output["enhanced_prompt"]

        # 2. (真实) 生成 V2 视频
        video_path_v2 = run_video_generation(str(image_path), enhanced_prompt, output_video_dir)
        if not video_path_v2:
            print(f"   (视频生成失败: {image_name})")
            fail_count += 1
            continue
        
        success_count += 1
        # print(f"   ✓ 最终视频已生成: {video_path_v2}") # 减少日志噪音

    print(f"--- ✅ 第 2 轮 (最终生成) 完成 ---")
    print(f"   成功: {success_count} / {len(tasks)}")
    print(f"   失败: {fail_count} / {len(tasks)}")

# --- 【新】主函数 (替换旧的 __main__) ---

def _pipeline_groups_from_args(args, json_path: Path, image_dir: Path, original_video_dir: Path) -> List[Dict[str, Any]]:
    """构建一组或多组 (json, 图像目录, 参考视频目录)。若传入 --l3-* / --l1l2-* 则仅用这些组，否则为单一默认组。"""
    groups: List[Dict[str, Any]] = []
    if args.l3_json:
        if not args.l3_images or not args.l3_ref_videos:
            raise ValueError("使用 --l3-json 时必须同时指定 --l3-images 与 --l3-ref-videos")
        groups.append({
            "label": "l3",
            "json": Path(args.l3_json).expanduser().resolve(),
            "image_dir": Path(args.l3_images).expanduser().resolve(),
            "ref_videos": Path(args.l3_ref_videos).expanduser().resolve(),
        })
    if args.l1l2_json:
        if not args.l1l2_images or not args.l1l2_ref_videos:
            raise ValueError("使用 --l1l2-json 时必须同时指定 --l1l2-images 与 --l1l2-ref-videos")
        groups.append({
            "label": "l1l2",
            "json": Path(args.l1l2_json).expanduser().resolve(),
            "image_dir": Path(args.l1l2_images).expanduser().resolve(),
            "ref_videos": Path(args.l1l2_ref_videos).expanduser().resolve(),
        })
    if groups:
        return groups
    return [{
        "label": "default",
        "json": json_path,
        "image_dir": image_dir,
        "ref_videos": original_video_dir,
    }]


async def main():
    """主编排函数"""
    default_base_dir = Path(__file__).resolve().parent
    repo_root = default_base_dir.parent
    track1_prompt_default = (repo_root / "examples" / "data" / "draft_videos").resolve()
    track1_pencil_default = (repo_root / "examples" / "data" / "references").resolve()
    merged_playbook_default = (repo_root / "playbooks" / "playbook_final.json").resolve()
    parser = argparse.ArgumentParser(
        description="Track1 ACE：铅笔参考图 + Seedance video_prompt；默认 Playbook 为 seedance2+HOI 合并版。"
    )
    parser.add_argument(
        "--mode",
        choices=["full_pipeline", "final_only", "warmup_only", "enhance_prompt_only"],
        default="full_pipeline",
        help="full_pipeline: ACE 完整多轮；final_only: Playbook+图生成视频；warmup_only: 仅 Epoch0；"
        " enhance_prompt_only: 仅用 Playbook+参考图+现有 prompt 做增强，无 ACE 迭代、无视频生成。",
    )
    parser.add_argument(
        "--json-path",
        default=str((repo_root / "examples/data/tasks.json").resolve()),
        help="任务列表 JSON 文件路径。",
    )
    parser.add_argument(
        "--image-dir",
        default=str((repo_root / "examples/data/images").resolve()),
        help="人物参考图像目录（按任务 JSON 中的文件名）。",
    )
    parser.add_argument(
        "--original-video-dir",
        default=str(track1_prompt_default),
        help="已有视频目录（Epoch0）：Track1 下为各 id 子文件夹内的 mp4（默认 out_seedance2_videos_same_crop）。",
    )
    parser.add_argument(
        "--epoch1-video-dir",
        default=str((repo_root / "runs/epoch_1_videos").resolve()),
        help="Epoch 1 输出目录。",
    )
    parser.add_argument(
        "--epoch2-video-dir",
        default=str((repo_root / "runs/epoch_2_videos").resolve()),
        help="Epoch 2 输出目录。",
    )
    parser.add_argument(
        "--direct-output-dir",
        default=str((repo_root / "runs/epoch_3_videos").resolve()),
        help="使用现有 Playbook 直接生成视频的输出目录。",
    )
    parser.add_argument(
        "--exclude-dir",
        default=None,
        help="如果指定，则跳过在该目录下已经生成过的视频任务。",
    )
    parser.add_argument(
        "--playbook-file",
        default=str(merged_playbook_default),
        help="Playbook 文件路径（默认 playbooks/playbook_final.json）。",
    )
    parser.add_argument(
        "--tasks-from",
        choices=["json", "test50_selected", "track1_seedance_pencil"],
        default="track1_seedance_pencil",
        help="任务来源：json=--json-path；test50_selected=selected_prompts.json；"
        "track1_seedance_pencil=各 id 的 video_prompt.txt（Seedance 目录）+ 铅笔图根目录。",
    )
    parser.add_argument(
        "--source-prompt-root",
        default=None,
        help="test50_selected：含 id004/ 等子目录的根（默认 <脚本目录>/test_50_mixed）。",
    )
    parser.add_argument(
        "--selected-prompts-json",
        default=None,
        help="test50_selected：selected_prompts.json 路径（默认 test_50_mixed/selected_prompts.json）。",
    )
    parser.add_argument(
        "--track1-prompt-root",
        default=None,
        help="track1：含 id*/video_prompt.txt 的根；默认 IPVG2026-Test-Track1/out_seedance2_videos_same_crop。",
    )
    parser.add_argument(
        "--track1-image-root",
        default=None,
        help="track1：含 id*/pencil.png 的根；默认 IPVG2026-Test-Track1/out_gpt_image2_pencil_same_crop。",
    )
    parser.add_argument(
        "--warmup-ref-roots",
        default=None,
        help="Epoch0 多库学习：逗号分隔目录列表（各含 id*/视频），相对脚本目录；与 --test50-warmup-defaults 二选一或并用（显式列表优先）。",
    )
    parser.add_argument(
        "--test50-warmup-defaults",
        action="store_true",
        help="Epoch0 使用四个默认库：test_50_mixed_seedance2_enhanced_i2v_videos、face_captions_videos、prompt_same_crop_pencil_videos、seedance2_videos。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个任务，用于调试或分批执行。",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="默认跳过输出目录中已存在的视频，启用该选项可强制重新生成。",
    )
    parser.add_argument(
        "--l3-json",
        default=None,
        help="v6 L3 任务 JSON；与 --l3-images、--l3-ref-videos 一起启用 L3 组（可与 L1L2 组串联）。",
    )
    parser.add_argument(
        "--l3-images",
        default=None,
        help="L3 数据源图像目录（如 v6_images_L3）。",
    )
    parser.add_argument(
        "--l3-ref-videos",
        default=None,
        help="L3 参考视频目录（如 wan22_qwen_playbook_v6_l3），Epoch0 从此学习。",
    )
    parser.add_argument(
        "--l1l2-json",
        default=None,
        help="v6 L1L2 任务 JSON；与 --l1l2-images、--l1l2-ref-videos 一起启用 L1L2 组。",
    )
    parser.add_argument(
        "--l1l2-images",
        default=None,
        help="L1L2 数据源图像目录（如 v6_images_L1L2）。",
    )
    parser.add_argument(
        "--l1l2-ref-videos",
        default=None,
        help="L1L2 参考视频目录（如 wan22_qwen_playbook_v6_l1l2）。",
    )
    parser.add_argument(
        "--qwen-backend",
        choices=["dashscope", "yunwu", "gemini"],
        default=None,
        help="ACE 多模态 LLM：gemini（默认；经云雾时模型 id 为 VLM_SCORER_MODEL→GEMINI_MODEL_NAME→gemini-3.1-pro-preview，与 batch_vlm_score_seedance2_methods 一致）、yunwu（仅 Qwen）、dashscope。"
        " 默认读 ACE_QWEN_BACKEND（未设则为 gemini）。",
    )
    parser.add_argument(
        "--gemini-transport",
        choices=["yunwu", "google"],
        default=None,
        help="ACE_QWEN_BACKEND=gemini 时：yunwu=云雾 OpenAI 兼容（默认，需 YUNWU_API_KEY）；"
        " google=直连 Google（需 GEMINI_API_KEY）。默认 ACE_GEMINI_TRANSPORT。",
    )
    parser.add_argument(
        "--yunwu-base-url",
        default=None,
        help="云雾 OpenAI 兼容 Base URL，须含 /v1，例如 https://yunwu.ai/v1。"
        " 默认读 YUNWU_BASE_URL 或 OPENAI_BASE_URL。",
    )
    parser.add_argument(
        "--i2v-backend",
        choices=["wan", "seedance"],
        default=None,
        help="图生视频后端：wan=DashScope wan2.2-i2v-flash；seedance=火山方舟 Seedance 2.0（与 batch_seedance2_r2v.py 一致）。"
        " 默认读环境变量 ACE_I2V_BACKEND（未设则为 seedance）。",
    )
    parser.add_argument(
        "--ark-api-key",
        default=None,
        help="火山方舟 API Key（ACE_I2V_BACKEND=seedance 时）。默认读 ARK_API_KEY 环境变量。",
    )
    parser.add_argument(
        "--enhance-input-root",
        default=None,
        help="（mode=enhance_prompt_only）含 id001/id002… 的数据根目录；指定后按子目录批量增强。"
        " 不指定则使用 --json-path + --image-dir（及 L3/L1L2 组）加载任务。",
    )
    parser.add_argument(
        "--enhance-instruction-file",
        default="prompt.txt",
        help="（目录模式）每个 id 子目录内作为「现有 prompt」的文本文件名。",
    )
    parser.add_argument(
        "--enhance-image-candidates",
        default="pencil.png,pencil_full_body.png,image.png,reference.png",
        help="Track1 / 目录模式：逗号分隔，按优先级选用第一张存在的参考图（默认 pencil.png 优先）。",
    )
    parser.add_argument(
        "--enhance-output-txt",
        default="enhanced_i2v_prompt.txt",
        help="（目录模式）写入各 id 目录下的增强 prompt 文件名。",
    )
    parser.add_argument(
        "--enhance-output-json",
        default="enhanced_i2v_generator.json",
        help="（目录模式）写入各 id 目录下的 Generator 完整 JSON 文件名。",
    )
    parser.add_argument(
        "--enhance-output-dir",
        default=None,
        help="（JSON 任务模式）增强结果保存目录；不指定则写在每张图同目录或默认命名。",
    )
    parser.add_argument(
        "--enhance-force",
        action="store_true",
        help="（enhance_prompt_only）覆盖已存在的增强输出文件。",
    )
    parser.add_argument(
        "--skip-ids",
        default=None,
        help="（enhance_prompt_only）逗号分隔的 id 列表（如 id001,id002），这些 id 不做增强。",
    )

    args = parser.parse_args()

    track1_prompt_root = (
        Path(args.track1_prompt_root).expanduser().resolve()
        if args.track1_prompt_root
        else track1_prompt_default
    )
    track1_image_root = (
        Path(args.track1_image_root).expanduser().resolve()
        if args.track1_image_root
        else track1_pencil_default
    )

    global _RUNTIME_QWEN_BACKEND, _RUNTIME_YUNWU_BASE_URL, PLAYBOOK_FILE
    global _RUNTIME_I2V_BACKEND, _RUNTIME_ARK_API_KEY, _RUNTIME_GEMINI_TRANSPORT
    if args.qwen_backend is not None:
        _RUNTIME_QWEN_BACKEND = args.qwen_backend
    if args.gemini_transport is not None:
        _RUNTIME_GEMINI_TRANSPORT = args.gemini_transport.strip().lower()
    if args.yunwu_base_url:
        _RUNTIME_YUNWU_BASE_URL = normalize_yunwu_base_url(args.yunwu_base_url.strip())
    if args.i2v_backend is not None:
        _RUNTIME_I2V_BACKEND = args.i2v_backend.strip().lower()
    if args.ark_api_key:
        _RUNTIME_ARK_API_KEY = args.ark_api_key.strip()

    # 1. 检查 API Key（ACE LLM + I2V）
    if _RUNTIME_QWEN_BACKEND == "gemini":
        if _RUNTIME_GEMINI_TRANSPORT == "yunwu":
            if not YUNWU_API_KEY:
                print("❗ Gemini 经云雾时需 YUNWU_API_KEY（默认 ACE_GEMINI_TRANSPORT=yunwu）。")
                return
        else:
            if not GEMINI_API_KEY:
                print("❗ ACE_GEMINI_TRANSPORT=google 时需 GEMINI_API_KEY（直连 Google）。")
                return
            if gemini_client is None:
                print("❗ Gemini 客户端未初始化（请安装 google-genai 并设置有效的 GEMINI_API_KEY）。")
                return
    if _RUNTIME_QWEN_BACKEND == "yunwu" and not YUNWU_API_KEY:
        print("❗ ACE_QWEN_BACKEND / --qwen-backend=yunwu 时需要环境变量 YUNWU_API_KEY（云雾控制台密钥）。")
        return
    if _RUNTIME_QWEN_BACKEND == "dashscope" and not DASHSCOPE_API_KEY:
        print("❗ 使用 DashScope 调用 Qwen 时需要 DASHSCOPE_API_KEY。")
        return

    _needs_i2v_keys = args.mode not in ("enhance_prompt_only", "warmup_only")
    if _needs_i2v_keys:
        if _RUNTIME_I2V_BACKEND == "seedance":
            if not (_RUNTIME_ARK_API_KEY or ARK_API_KEY):
                print(
                    "❗ ACE_I2V_BACKEND=seedance 时需要 ARK_API_KEY（或传入 --ark-api-key）。"
                    " 参见 batch_seedance2_r2v.py 使用的火山方舟 SDK。"
                )
                return
        else:
            if not DASHSCOPE_API_KEY:
                print("❗ ACE_I2V_BACKEND=wan 时需要 DASHSCOPE_API_KEY（wan2.2-i2v-flash）。")
                return

    if _RUNTIME_QWEN_BACKEND == "gemini":
        if _RUNTIME_GEMINI_TRANSPORT == "yunwu":
            print(
                f"✅ 本次 ACE LLM → Gemini 经云雾 {_RUNTIME_YUNWU_BASE_URL}（chat 模型: {YUNWU_ACE_CHAT_MODEL}）"
            )
        else:
            print(f"✅ 本次 ACE LLM → Gemini（{GEMINI_MODEL_NAME}）直连 Google")
    elif _RUNTIME_QWEN_BACKEND == "yunwu":
        print(f"✅ 本次 ACE LLM → 云雾 {_RUNTIME_YUNWU_BASE_URL}（模型名须与控制台一致: {QWEN_MODEL_NAME}）")
    else:
        print(f"✅ 本次 ACE LLM → DashScope 直连（{QWEN_MODEL_NAME}）")
    if args.mode == "enhance_prompt_only":
        print("✅ 本次模式：仅增强 Prompt（不调用图生视频）")
    elif _RUNTIME_I2V_BACKEND == "seedance":
        print(f"✅ 本次 I2V → Seedance 2.0（{os.getenv('SEEDANCE_MODEL', SEEDANCE_MODEL)} @ Ark）")
    else:
        print("✅ 本次 I2V → DashScope wan2.2-i2v-flash")

    # 2. 解析路径
    json_path = Path(args.json_path).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    original_video_dir = Path(args.original_video_dir).expanduser().resolve()
    epoch1_video_dir = Path(args.epoch1_video_dir).expanduser().resolve()
    epoch2_video_dir = Path(args.epoch2_video_dir).expanduser().resolve()
    direct_output_dir = Path(args.direct_output_dir).expanduser().resolve()
    exclude_dir = Path(args.exclude_dir).expanduser().resolve() if args.exclude_dir else None

    source_prompt_root = (
        Path(args.source_prompt_root).expanduser().resolve()
        if args.source_prompt_root
        else (repo_root / "examples/data/input").resolve()
    )
    selected_prompts_json_path = (
        Path(args.selected_prompts_json).expanduser().resolve()
        if args.selected_prompts_json
        else (repo_root / "examples/data/tasks.json").resolve()
    )

    test50_warmup_quartet = (
        "test_50_mixed_seedance2_enhanced_i2v_videos",
        "test_50_mixed_seedance2_face_captions_videos",
        "test_50_mixed_seedance2_prompt_same_crop_pencil_videos",
        "test_50_mixed_seedance2_videos",
    )
    warmup_roots = parse_comma_separated_paths(args.warmup_ref_roots, default_base_dir)
    if not warmup_roots and args.test50_warmup_defaults:
        warmup_roots = [(default_base_dir / n).resolve() for n in test50_warmup_quartet]

    names_tuple = tuple(x.strip() for x in args.enhance_image_candidates.split(",") if x.strip())

    # Playbook 路径必须为全局变量，以兼容现有函数
    PLAYBOOK_FILE = str(Path(args.playbook_file).expanduser().resolve())

    Path(PLAYBOOK_FILE).parent.mkdir(parents=True, exist_ok=True)

    playbook_v0 = load_playbook(PLAYBOOK_FILE)

    if args.mode == "enhance_prompt_only":
        print("\n--- 📝 enhance_prompt_only：Playbook + 图 + 现有 prompt → 增强（无 ACE 迭代、无视频）---")
        names = tuple(x.strip() for x in args.enhance_image_candidates.split(",") if x.strip())
        force = args.enhance_force
        skip_ids_enhance = parse_skip_ids(args.skip_ids)
        if skip_ids_enhance:
            print(f"   跳过 {len(skip_ids_enhance)} 个 id（--skip-ids）")
        if args.enhance_input_root:
            root = Path(args.enhance_input_root).expanduser().resolve()
            if not root.is_dir():
                print(f"❗ --enhance-input-root 不是目录: {root}")
                return
            await run_enhance_prompt_only_folder(
                root,
                playbook_v0,
                args.enhance_instruction_file,
                names,
                force,
                args.enhance_output_txt,
                args.enhance_output_json,
                args.limit,
                skip_ids_enhance,
            )
            return
        if args.tasks_from == "track1_seedance_pencil":
            tasks_t1 = load_tasks_track1_seedance_prompt_pencil_image(
                track1_prompt_root, track1_image_root, names_tuple
            )
            if args.limit:
                tasks_t1 = tasks_t1[: args.limit]
            if not tasks_t1:
                print("❗ track1_seedance_pencil：未加载到任务（检查两目录下 id* 对齐与 video_prompt.txt / 参考图）。")
                return
            await run_enhance_track1_tasks_write_under_prompt_root(
                tasks_t1,
                playbook_v0,
                track1_image_root,
                track1_prompt_root,
                args.enhance_force,
                skip_ids=skip_ids_enhance,
            )
            return
        if args.tasks_from == "test50_selected":
            tasks_sel = load_tasks_from_selected_prompts(
                source_prompt_root, selected_prompts_json_path, names_tuple
            )
            if args.limit:
                tasks_sel = tasks_sel[: args.limit]
            if not tasks_sel:
                print(
                    "❗ test50_selected：未加载到任务（检查 --source-prompt-root、--selected-prompts-json 与参考图）。"
                )
                return
            out_dir_enh = (
                Path(args.enhance_output_dir).expanduser().resolve()
                if args.enhance_output_dir
                else None
            )
            await run_enhance_prompt_only_tasks(
                tasks_sel,
                playbook_v0,
                source_prompt_root,
                out_dir_enh,
                args.enhance_force,
                "test50_selected",
            )
            return
        try:
            pipeline_groups_enh = _pipeline_groups_from_args(args, json_path, image_dir, original_video_dir)
        except ValueError as e:
            print(f"❗ {e}")
            return
        multi_enh = len(pipeline_groups_enh) > 1
        out_base = Path(args.enhance_output_dir).expanduser().resolve() if args.enhance_output_dir else None

        def _load_tasks_enh(g: Dict[str, Any]) -> List[Dict[str, str]]:
            t = load_tasks_from_json(g["json"])
            if args.limit:
                t = t[: args.limit]
            return t

        for g in pipeline_groups_enh:
            tasks_g = _load_tasks_enh(g)
            if not tasks_g:
                print(f"❗ 组 {g['label']} 没有任务，跳过。")
                continue
            od: Optional[Path] = None
            if out_base is not None:
                od = out_base / g["label"] if multi_enh else out_base
            await run_enhance_prompt_only_tasks(
                tasks_g, playbook_v0, g["image_dir"], od, force, g["label"]
            )
        return

    try:
        if args.tasks_from == "test50_selected":
            pipeline_groups = [
                {
                    "label": "test50",
                    "json": json_path,
                    "image_dir": source_prompt_root,
                    "ref_videos": original_video_dir,
                }
            ]
        elif args.tasks_from == "track1_seedance_pencil":
            pipeline_groups = [
                {
                    "label": "track1",
                    "json": json_path,
                    "image_dir": track1_image_root,
                    "ref_videos": original_video_dir,
                }
            ]
        else:
            pipeline_groups = _pipeline_groups_from_args(args, json_path, image_dir, original_video_dir)
    except ValueError as e:
        print(f"❗ {e}")
        return

    multi_group = len(pipeline_groups) > 1

    def _load_group_tasks(g: Dict[str, Any]) -> List[Dict[str, str]]:
        if args.tasks_from == "test50_selected":
            t = load_tasks_from_selected_prompts(
                source_prompt_root, selected_prompts_json_path, names_tuple
            )
        elif args.tasks_from == "track1_seedance_pencil":
            t = load_tasks_track1_seedance_prompt_pencil_image(
                track1_prompt_root, track1_image_root, names_tuple
            )
        else:
            t = load_tasks_from_json(g["json"])
        if args.limit:
            t = t[: args.limit]
        return t

    if args.mode == "final_only":
        for g in pipeline_groups:
            tasks_g = _load_group_tasks(g)
            if not tasks_g:
                print(f"❗ 组 {g['label']} 没有任务，跳过。")
                continue
            out_dir = direct_output_dir / g["label"] if multi_group else direct_output_dir
            await generate_videos_from_playbook(
                tasks_g,
                playbook_v0,
                g["image_dir"],
                out_dir,
                exclude_video_dir=exclude_dir,
                skip_if_exists=not args.no_skip_existing,
            )
        return

    if args.mode == "warmup_only":
        if warmup_roots:
            await run_epoch_0_warmup_multi_roots(
                playbook_v0,
                source_prompt_root,
                selected_prompts_json_path,
                warmup_roots,
                names_tuple,
                args.limit,
            )
        else:
            playbook = playbook_v0
            for g in pipeline_groups:
                tasks_g = _load_group_tasks(g)
                if not tasks_g:
                    print(f"❗ 组 {g['label']} 没有任务，跳过。")
                    continue
                playbook = await run_epoch_0_warmup(tasks_g, playbook, g["image_dir"], g["ref_videos"])
        return

    # 5. 执行多轮迭代（多组时依次累积同一 Playbook；视频输出到 epoch1/2 下 l3/、l1l2/ 子目录）
    try:
        epoch1_video_dir.mkdir(parents=True, exist_ok=True)
        epoch2_video_dir.mkdir(parents=True, exist_ok=True)

        playbook_acc = playbook_v0
        if warmup_roots:
            playbook_acc = await run_epoch_0_warmup_multi_roots(
                playbook_acc,
                source_prompt_root,
                selected_prompts_json_path,
                warmup_roots,
                names_tuple,
                args.limit,
            )
        elif args.tasks_from != "test50_selected":
            for g in pipeline_groups:
                tasks_g = _load_group_tasks(g)
                if not tasks_g:
                    print(f"❗ 组 {g['label']} 没有任务，跳过 Epoch 0。")
                    continue
                playbook_acc = await run_epoch_0_warmup(tasks_g, playbook_acc, g["image_dir"], g["ref_videos"])
        else:
            print(
                "⚠️ tasks_from=test50_selected 但未指定 --warmup-ref-roots / --test50-warmup-defaults，跳过 Epoch0，仅从 Epoch1 起用 Seedance 生成。"
            )

        for g in pipeline_groups:
            tasks_g = _load_group_tasks(g)
            if not tasks_g:
                print(f"❗ 组 {g['label']} 没有任务，跳过 Epoch 1。")
                continue
            out1 = epoch1_video_dir / g["label"] if multi_group else epoch1_video_dir
            playbook_acc = await run_epoch_1_learn(tasks_g, playbook_acc, g["image_dir"], out1)

        for g in pipeline_groups:
            tasks_g = _load_group_tasks(g)
            if not tasks_g:
                print(f"❗ 组 {g['label']} 没有任务，跳过 Epoch 2。")
                continue
            out2 = epoch2_video_dir / g["label"] if multi_group else epoch2_video_dir
            await run_epoch_2_final(tasks_g, playbook_acc, g["image_dir"], out2)

        print("\n--- 🚀 ACE 多轮迭代全部完成 🚀 ---")

    except Exception as e:
        print(f"\n--- ❗ 异步主循环中发生意外错误 ---")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行主异步函数
    asyncio.run(main())
