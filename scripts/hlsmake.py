#!/usr/bin/env python3
"""HLS под запись экрана: одна дорожка 720p сегментами по 6 секунд.

Отдельно от апстримного transcode.py, потому что тот не годится для этой задачи:
там лесенка из трёх качеств, и аудио дублируется в каждое. На двухчасовой записи
Google Meet это давало HLS ТЯЖЕЛЕЕ исходника — 640 МБ против 590. Исходник идёт
примерно на 630 кбит/с, он уже экономный, и обогнать его можно только одной
дорожкой с меньшим битрейтом.

Замерено на лекции: 324 МБ вместо 590 за 4 минуты, картинка на глаз не отличается.
tune=stillimage — потому что на экране статичный текст, а не движение.
"""
from __future__ import annotations

import os
import subprocess


def src_kbps(path: str) -> float:
    """Битрейт исходника, кбит/с. 0 если не определился."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate",
                        "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) / 1000
    except ValueError:
        return 0.0


def make_hls(src: str, dst_dir: str, bitrate: str = "350k", preset: str = "medium") -> str:
    """→ путь к master.m3u8. Одна дорожка, поэтому master и есть медиаплейлист."""
    target = int(bitrate.rstrip("k")) + 96          # видео + аудио
    have = src_kbps(src)
    if have and have < target * 1.15:
        print(f"    ! исходник и так {have:.0f} кбит/с, цель {target} — файл станет ТЯЖЕЛЕЕ.\n"
              f"      Сжатие имеет смысл от ~{int(target * 1.15)} кбит/с; для нарезки на "
              f"сегменты это всё равно может быть нужно.")
    hls = os.path.join(dst_dir, "hls")
    os.makedirs(hls, exist_ok=True)
    buf = f"{int(bitrate.rstrip('k')) * 2}k"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", src,
        "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", preset,
        "-tune", "stillimage", "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", buf,
        "-g", "48", "-keyint_min", "48", "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2",
        "-f", "hls", "-hls_time", "6", "-hls_playlist_type", "vod",
        "-hls_flags", "independent_segments",
        "-hls_segment_filename", os.path.join(hls, "seg%04d.ts"),
        os.path.join(hls, "master.m3u8"),
    ], check=True)
    return os.path.join(hls, "master.m3u8")


def dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, x))
               for d, _, fs in os.walk(path) for x in fs)
