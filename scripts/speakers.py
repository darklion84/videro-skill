#!/usr/bin/env python3
"""Список спикеров с примерами реплик → руками проставить имена → накатить на транскрипт.

Диаризация даёт безымянные метки SPEAKER_00/01/02. Опознать, кто есть кто, может
только человек. Поэтому:

  1. `speakers.py timeline.json`          → speakers.json: кто сколько говорит и
                                            где впервые слышен, с таймкодом и текстом
                                            реплики — чтобы открыть видео на этом месте
  2. руками вписать "name" в speakers.json
  3. `speakers.py timeline.json --apply`  → имена уходят в transcript[].speaker_name
                                            и в subtitles.srt

Исходные метки НЕ затираются: имя живёт в отдельном поле, так что шаг обратим,
а повторный прогон первого режима сохраняет уже вписанные имена.

  python scripts/speakers.py videro-out/x/timeline.json
  python scripts/speakers.py videro-out/x/timeline.json --apply
  python scripts/speakers.py videro-out/x/timeline.json --apply --no-srt-names
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze          # noqa: E402  — ради build_srt, чтобы не плодить свой формат

SAMPLES = 2          # сколько примеров реплик показывать на спикера
MIN_CHARS = 25       # «ага» и «секунду» для опознания бесполезны


def fmt_ts(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_roster(transcript: list[dict], keep: dict[str, str]) -> dict:
    """keep — уже вписанные имена, их нельзя терять при пересборке."""
    ids, unknown = {}, 0
    for seg in transcript:
        spk = seg.get("speaker")
        if not spk:
            unknown += 1
            continue
        ids.setdefault(spk, []).append(seg)

    total = sum(s["end"] - s["start"] for s in transcript) or 1
    out = []
    for spk, segs in sorted(ids.items(), key=lambda kv: -sum(s["end"] - s["start"] for s in kv[1])):
        talk = sum(s["end"] - s["start"] for s in segs)
        good = [s for s in segs if len(s.get("text") or "") >= MIN_CHARS] or segs
        out.append({
            "id": spk,
            "name": keep.get(spk, ""),
            "turns": len(segs),
            "talk_sec": round(talk, 1),
            "share": round(talk / total, 3),
            "samples": [{"at": round(s["start"], 1), "ts": fmt_ts(s["start"]),
                         "text": (s.get("text") or "")[:160]} for s in good[:SAMPLES]],
        })
    return {"speakers": out, "segments_without_speaker": unknown}


def roster_path(tl_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(tl_path)), "speakers.json")


def _load_timeline(tl_path: str) -> tuple[dict, list[dict]]:
    with open(tl_path, encoding="utf-8") as f:
        tl = json.load(f)
    transcript = tl.get("transcript") or []
    if not transcript:
        raise ValueError("в timeline.json пустой transcript")
    if not any(s.get("speaker") for s in transcript):
        raise ValueError("ни у одной реплики нет speaker — сначала прогони diarize.py --timeline")
    return tl, transcript


def make_roster(tl_path: str) -> dict:
    """Собрать/пересобрать speakers.json, сохранив уже вписанные имена."""
    _, transcript = _load_timeline(tl_path)
    rp = roster_path(tl_path)
    keep = {}
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            keep = {s["id"]: s.get("name", "") for s in json.load(f).get("speakers", [])}
    roster = build_roster(transcript, keep)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(roster, f, ensure_ascii=False, indent=2)
    return roster


def save_names(tl_path: str, names: dict[str, str]) -> dict:
    """Записать имена в speakers.json (создав реестр, если его ещё нет)."""
    rp = roster_path(tl_path)
    if not os.path.exists(rp):
        make_roster(tl_path)
    with open(rp, encoding="utf-8") as f:
        roster = json.load(f)
    for s in roster.get("speakers", []):
        if s["id"] in names:
            s["name"] = (names[s["id"]] or "").strip()
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(roster, f, ensure_ascii=False, indent=2)
    return roster


def apply_names(tl_path: str, srt_names: bool = True) -> dict:
    """Разнести имена из speakers.json по транскрипту и субтитрам. → статистика."""
    tl_path = os.path.abspath(tl_path)
    tl, transcript = _load_timeline(tl_path)
    rp = roster_path(tl_path)
    if not os.path.exists(rp):
        raise ValueError("нет speakers.json — сначала собери реестр")
    with open(rp, encoding="utf-8") as f:
        all_names = {s["id"]: (s.get("name") or "").strip()
                     for s in json.load(f).get("speakers", [])}
    named = {k: v for k, v in all_names.items() if v}
    if not named:
        raise ValueError("не заполнено ни одного имени")

    n = 0
    for seg in transcript:
        nm = named.get(seg.get("speaker") or "")
        if nm:
            seg["speaker_name"] = nm
            n += 1
        else:
            seg.pop("speaker_name", None)      # имя убрали — метка не должна остаться

    out_dir = os.path.dirname(tl_path)
    if srt_names:
        pref = [{**s, "text": (f"{s['speaker_name']}: {s['text']}"
                               if s.get("speaker_name") else s["text"])} for s in transcript]
        tl["srt"] = analyze.build_srt(pref)
        with open(os.path.join(out_dir, "subtitles.srt"), "w", encoding="utf-8") as f:
            f.write(tl["srt"])

    tmp = tl_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tl, f, ensure_ascii=False, indent=2)
    os.replace(tmp, tl_path)

    return {"named_segments": n, "total_segments": len(transcript), "names": named,
            "unnamed": [k for k in all_names if k not in named]}


def main():
    ap = argparse.ArgumentParser(description="реестр спикеров и подстановка имён")
    ap.add_argument("timeline", help="путь к timeline.json")
    ap.add_argument("--apply", action="store_true", help="накатить имена из speakers.json")
    ap.add_argument("--no-srt-names", action="store_true",
                    help="не добавлять «Имя:» в субтитры (по умолчанию добавляет)")
    args = ap.parse_args()

    tl_path = os.path.abspath(args.timeline)
    try:
        if not args.apply:
            roster = make_roster(tl_path)
            rp = roster_path(tl_path)
            print(f"✔ {len(roster['speakers'])} спикеров → {rp}")
            if roster["segments_without_speaker"]:
                print(f"  без спикера: {roster['segments_without_speaker']} реплик "
                      f"(речь вне интервалов диаризации)")
            print()
            for s in roster["speakers"]:
                named = f'  → «{s["name"]}»' if s["name"] else ""
                print(f"  {s['id']}  {s['talk_sec'] / 60:5.1f} мин  {s['share'] * 100:4.1f}%  "
                      f"{s['turns']:4d} реплик{named}")
                for smp in s["samples"]:
                    print(f"       {smp['ts']}  {smp['text'][:96]}")
                print()
            print("Впиши имена — либо кнопкой «Назвать спикеров» в плеере,")
            print(f"либо в поле \"name\" в {rp}, потом:")
            print(f"  ./run.sh speakers {args.timeline} --apply")
            return

        res = apply_names(tl_path, srt_names=not args.no_srt_names)
    except ValueError as e:
        sys.exit(str(e))

    if not args.no_srt_names:
        print(f"  субтитры с именами → {os.path.join(os.path.dirname(tl_path), 'subtitles.srt')}")
    print(f"✔ имена проставлены у {res['named_segments']} из {res['total_segments']} реплик "
          f"→ {tl_path}")
    for k, v in res["names"].items():
        print(f"  {k} → {v}")
    if res["unnamed"]:
        print(f"  без имени осталось: {', '.join(res['unnamed'])}")


if __name__ == "__main__":
    main()
