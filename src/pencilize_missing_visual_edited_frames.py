#!/usr/bin/env python3
"""
将 missing_visual_fix 下已有的 edited_frame.png 转为铅笔素描风格（供 Seedance 与 Image 1 一致）。

默认：若不存在 edited_frame_photo.png，先把当前 edited_frame.png 备份为 edited_frame_photo.png，
再覆盖 edited_frame.png 为铅笔版。

  export YUNWU_GPT_IMAGE_API_KEY=...

  python3 pencilize_missing_visual_edited_frames.py \\
    --missing-visual-fix-dir .../id014/missing_visual_fix \\
    --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from gpt_image2_edit_frame_api import (  # noqa: E402
    build_pencil_style_transfer_prompt,
    edit_frame_image,
    make_openai_client,
)


def iter_issue_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("issue_"):
            out.append(p)
    return out


def process_issue(
    issue_dir: Path,
    *,
    client,
    args: argparse.Namespace,
) -> bool:
    edited = issue_dir / "edited_frame.png"
    if not edited.is_file():
        print(f"[skip] {issue_dir.name}: 无 edited_frame.png", file=sys.stderr)
        return False

    photo = issue_dir / "edited_frame_photo.png"
    if args.dry_run:
        print(f"[dry-run] {issue_dir.name}: 将铅笔化 {edited.name}", file=sys.stderr)
        return True

    if not photo.is_file():
        if args.backup_photo or not args.no_backup_photo:
            shutil.copy2(edited, photo)
            print(f"  [{issue_dir.name}] 备份 → edited_frame_photo.png", file=sys.stderr)
    elif args.backup_photo and args.force:
        shutil.copy2(edited, photo)

    prompt = build_pencil_style_transfer_prompt()
    (issue_dir / "pencil_transfer_prompt_en.txt").write_text(prompt + "\n", encoding="utf-8")
    tmp = issue_dir / "edited_frame_pencil.tmp.png"
    edit_frame_image(
        client,
        image_path=photo if photo.is_file() else edited,
        prompt=prompt,
        output_path=tmp,
        image_model=args.image_model,
        size=args.size,
        quality=args.quality,
        input_fidelity=args.input_fidelity or None,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
    )
    tmp.replace(edited)
    print(f"  [{issue_dir.name}] ✅ edited_frame.png（铅笔）", file=sys.stderr)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="将 missing_visual 修帧图转为铅笔素描。")
    p.add_argument("--missing-visual-fix-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="即使已有 pencil_transfer 也重做")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-backup-photo", action="store_true", help="不备份原 edited_frame 为 edited_frame_photo.png")
    p.add_argument("--backup-photo", action="store_true", help="总是用当前 edited_frame 覆盖 photo 备份")
    p.add_argument("--image-model", default=os.environ.get("GPT_IMAGE_MODEL", "gpt-image-2"))
    p.add_argument("--size", default=os.environ.get("GPT_IMAGE_SIZE", "auto"))
    p.add_argument("--quality", default=os.environ.get("GPT_IMAGE_QUALITY", "medium"))
    p.add_argument("--input-fidelity", default=os.environ.get("GPT_IMAGE_INPUT_FIDELITY", "high"))
    p.add_argument("--image-api-key", default=None)
    p.add_argument("--image-api-base-url", default=None)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--retry-wait", type=int, default=30)
    p.add_argument("--image-api-timeout", type=float, default=600.0)
    args = p.parse_args()
    if args.input_fidelity == "":
        args.input_fidelity = None

    root = args.missing_visual_fix_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"找不到目录: {root}")

    client = None
    if not args.dry_run:
        client = make_openai_client(
            api_key=args.image_api_key,
            base_url=args.image_api_base_url,
            timeout=args.image_api_timeout,
        )

    n = 0
    for issue_dir in iter_issue_dirs(root):
        edited = issue_dir / "edited_frame.png"
        if not edited.is_file():
            continue
        if not args.force and (issue_dir / "pencil_transfer_prompt_en.txt").is_file():
            # 已铅笔化过（启发式）；仍可用 --force
            try:
                issue = json.loads((issue_dir / "issue.json").read_text(encoding="utf-8"))
                if issue.get("edited_frame_style") == "pencil":
                    print(f"[skip] {issue_dir.name}: 已是铅笔", file=sys.stderr)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if process_issue(issue_dir, client=client, args=args):
            n += 1

    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not args.dry_run:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["edited_frames_pencilized"] = True
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass

    print(f"\n✅ 铅笔化 {n} 个 issue → {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
