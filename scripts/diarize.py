#!/usr/bin/env python3
"""Диаризация (кто когда говорит) через pyannote → проставляет speaker в timeline.json.

Локальный whisper даёт текст с таймкодами, но не говорящих. Здесь считается отдельный
слой «интервал → спикер», который затем пришивается к УЖЕ готовому транскрипту:
для каждой реплики берётся спикер с наибольшим перекрытием. Перегонять ASR не нужно.

Модель гейтед: прими условия на huggingface.co/pyannote/speaker-diarization-community-1
и положи токен в .env как HF_TOKEN.

Env: HF_TOKEN, PYANNOTE_MODEL, PYANNOTE_DEVICE (mps|cpu).

  python scripts/diarize.py видео.mp4 --timeline videro-out/x/timeline.json
  python scripts/diarize.py видео.mp4 --json turns.json --device cpu   # без записи в timeline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")


def to_wav(src: str, dst: str, start: float | None = None, dur: float | None = None) -> bool:
    """pyannote хочет 16 кГц моно; заодно позволяет вырезать кусок для проверок."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]
    return subprocess.run(cmd).returncode == 0 and os.path.getsize(dst) > 0


def diarize(wav: str, device: str = "auto", num_speakers: int | None = None,
            progress: bool = True) -> list[dict]:
    """→ [{start, end, speaker}], отсортировано по времени."""
    import torch
    from pyannote.audio import Pipeline

    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise SystemExit("нет HF_TOKEN — прими условия модели на HF и положи токен в .env")

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    pipe = Pipeline.from_pretrained(MODEL, token=token)
    if pipe is None:
        raise SystemExit(f"{MODEL} не отдался — условия репозитория приняты этим аккаунтом?")
    pipe.to(torch.device(device))

    kw = {"num_speakers": num_speakers} if num_speakers else {}
    if progress:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            out = pipe(wav, hook=hook, **kw)
    else:
        out = pipe(wav, **kw)

    # community-1 (4.x) отдаёт объект с несколькими разметками, 3.1 — сразу Annotation
    ann = getattr(out, "speaker_diarization", out)
    turns = [{"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk}
             for seg, _, spk in ann.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t["start"])
    return turns


def stamp(transcript: list[dict], turns: list[dict]) -> int:
    """Проставить speaker каждой реплике по максимальному перекрытию. → сколько проставлено."""
    n = 0
    for seg in transcript:
        best, best_ov = None, 0.0
        for t in turns:
            if t["start"] >= seg["end"]:
                break
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        seg["speaker"] = best
        n += best is not None
    return n


def main():
    ap = argparse.ArgumentParser(description="диаризация → speaker в timeline.json")
    ap.add_argument("media", help="видео или аудио")
    ap.add_argument("--timeline", default=None, help="timeline.json, куда проставить speaker")
    ap.add_argument("--json", default=None, help="куда сложить сырые интервалы спикеров")
    ap.add_argument("--device", default=os.getenv("PYANNOTE_DEVICE", "auto"),
                    choices=["auto", "mps", "cpu"])
    ap.add_argument("--num-speakers", type=int, default=None, help="если число говорящих известно")
    ap.add_argument("--start", type=float, default=None, help="вырезать кусок: начало, сек")
    ap.add_argument("--dur", type=float, default=None, help="вырезать кусок: длительность, сек")
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav = tf.name
    try:
        if not to_wav(args.media, wav, args.start, args.dur):
            sys.exit(f"ffmpeg не смог вытащить аудио из {args.media}")
        print(f"▶ {MODEL} на {args.device}")
        t0 = time.time()
        turns = diarize(wav, args.device, args.num_speakers)
        dt = time.time() - t0
    finally:
        os.path.exists(wav) and os.unlink(wav)

    span = turns[-1]["end"] if turns else 0
    speakers = sorted({t["speaker"] for t in turns})
    print(f"\n✔ {len(turns)} интервалов, {len(speakers)} спикеров за {dt:.0f}s "
          f"({span / dt if dt else 0:.1f}× реалтайма)")
    for s in speakers:
        own = [t for t in turns if t["speaker"] == s]
        talk = sum(t["end"] - t["start"] for t in own)
        print(f"  {s}: {len(own):4d} реплик, {talk / 60:6.1f} мин, первый выход {own[0]['start']:.1f}s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)
        print(f"  → {args.json}")

    if args.timeline:
        path = os.path.abspath(args.timeline)
        with open(path, encoding="utf-8") as f:
            tl = json.load(f)
        tr = tl.get("transcript") or []
        if not tr:
            sys.exit(f"в {path} пустой transcript — размечать нечего")
        n = stamp(tr, turns)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tl, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(f"  speaker проставлен у {n} из {len(tr)} реплик → {path}")


if __name__ == "__main__":
    main()
