#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def open_rgb(path: Path):
    # 统一转成 RGB，避免 RGBA/P 模式混在一起出问题
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    # 如果是 RGBA，转 RGB（你也可以改成保留透明通道）
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img


def hstack_images(paths, resize_to="min_height"):
    """
    横向拼接多张图（同名来自不同文件夹）
    resize_to:
      - "min_height": 所有图缩放到最小高度
      - "max_height": 所有图缩放到最大高度
      - None: 不缩放（要求高度一致，否则会错位）
    """
    imgs = [open_rgb(p) for p in paths]

    if resize_to in ("min_height", "max_height"):
        heights = [im.height for im in imgs]
        target_h = min(heights) if resize_to == "min_height" else max(heights)

        resized = []
        for im in imgs:
            if im.height == target_h:
                resized.append(im)
                continue
            new_w = int(round(im.width * (target_h / im.height)))
            resized.append(im.resize((new_w, target_h), Image.LANCZOS))
        imgs = resized

    total_w = sum(im.width for im in imgs)
    max_h = max(im.height for im in imgs)

    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in imgs:
        y = (max_h - im.height) // 2  # 垂直居中
        canvas.paste(im, (x, y))
        x += im.width
    return canvas


def vstack_images(imgs):
    """纵向拼接一组横图"""
    widths = [im.width for im in imgs]
    target_w = max(widths)

    resized = []
    for im in imgs:
        if im.width == target_w:
            resized.append(im)
            continue
        new_h = int(round(im.height * (target_w / im.width)))
        resized.append(im.resize((target_w, new_h), Image.LANCZOS))

    total_h = sum(im.height for im in resized)
    canvas = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    y = 0
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def find_common_filenames(folders):
    """取所有文件夹的图片文件名交集（只按文件名匹配）"""
    sets = []
    for fd in folders:
        names = {p.name for p in list_images(fd)}
        sets.append(names)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def main(
    input_folders,
    output_dir,
    group_size=3,
    resize_to="min_height",
    output_ext=".jpg",
    jpg_quality=95,
):
    folders = [Path(x).expanduser().resolve() for x in input_folders]
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for fd in folders:
        if not fd.exists():
            raise FileNotFoundError(f"Input folder not found: {fd}")

    common_names = find_common_filenames(folders)
    if not common_names:
        raise RuntimeError("No common image filenames across all folders.")

    # 先为每个同名文件做横向拼接，得到一批横图
    h_imgs = []
    h_names = []
    for name in common_names:
        paths = [fd / name for fd in folders]
        if not all(p.exists() for p in paths):
            continue
        merged = hstack_images(paths, resize_to=resize_to)
        h_imgs.append(merged)
        h_names.append(Path(name).stem)

    if not h_imgs:
        raise RuntimeError("No images were merged. Check filenames and extensions.")

    # 再按 group_size 纵向堆叠输出
    for i in range(0, len(h_imgs), group_size):
        chunk_imgs = h_imgs[i : i + group_size]
        chunk_names = h_names[i : i + group_size]

        vmerged = vstack_images(chunk_imgs)

        # 输出文件名：起止索引 + 组内名字范围（便于定位）
        start = i + 1
        end = i + len(chunk_imgs)
        name_hint = f"{chunk_names[0]}__to__{chunk_names[-1]}" if chunk_names else "group"
        out_name = f"group_{start:04d}_{end:04d}__{name_hint}{output_ext}"
        out_path = out_dir / out_name

        if output_ext.lower() in (".jpg", ".jpeg"):
            vmerged.save(out_path, quality=jpg_quality, subsampling=0)
        else:
            vmerged.save(out_path)

        print(f"[OK] saved: {out_path}")

    print(f"Done. Total common images: {len(h_imgs)}, groups: {(len(h_imgs) + group_size - 1)//group_size}")


if __name__ == "__main__":
    # ====== 你只需要改这里 ======
    INPUT_FOLDERS = [
        "/SSD/wangyh/shadow/shadowdata/SBU-shadow/SBU-shadow/SBU-Test/ShadowImages",
        "/SSD/wangyh/shadow/shadowdata/SBU-shadow/SBU-shadow/SBU-Test/ShadowMasks",
        "/SSD/wangyh/shadowdino/ShadowDino26/results/sbu_logit/10000",
        # 还可以继续加，顺序就是横向拼接顺序
    ]
    OUTPUT_DIR = "/SSD/wangyh/shadowdino/ShadowDino26/results/concat_logit"

    main(
        input_folders=INPUT_FOLDERS,
        output_dir=OUTPUT_DIR,
        group_size=3,          # “最好 3 个图为一组” -> 每 3 张横图再纵向拼为一个大图
        resize_to="min_height",# 横拼时统一高度：min_height / max_height / None
        output_ext=".jpg",     # 输出格式：.jpg / .png
        jpg_quality=95,
    )