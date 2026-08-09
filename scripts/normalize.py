#!/usr/bin/env python3
"""Привести термины в транскрипте к написанию из glossary.txt.

Зачем именно постобработка, а не глоссарий в initial_prompt: проверено, что
initial_prompt на искажённые термины не влияет вообще. Whisper пишет «харнес»
или harness в зависимости от того, где прошла граница 30-секундного окна, —
на один и тот же звук даёт оба варианта. Управлять этим нельзя, а исправить
после можно: соответствие «харнес» → harness по глоссарию восстанавливается.

Чем отличается от апстримного fix_subtitles: тот опирается на OCR-контекст окна
и правит «что попало под руку», из-за чего однажды перенёс в транскрипт токен
с экрана. Здесь список терминов задан явно и ничего кроме них не трогается.

  python scripts/normalize.py videro-out/x/timeline.json
  python scripts/normalize.py videro-out/x/timeline.json --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import json
import os
import re
import sys

NUM = re.compile(r"^\s*\d+[.)]\s*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze          # noqa: E402  — ради build_srt
import asr_local        # noqa: E402  — ради load_glossary
import hub_retry        # noqa: E402
import redact           # noqa: E402

HTTP = hub_retry.session()
NDT_BASE = (os.getenv("ND_BASE_URL", "https://api.neuraldeep.ru/v1")).rstrip("/")
NDT_KEY = os.getenv("NEURALDEEP_API_KEY", "")
MODEL = os.getenv("NDT_NORMALIZE_MODEL", os.getenv("NDT_VISION_MODEL", "qwen3.6-fp8"))
WORKERS = int(os.getenv("ANALYZE_WORKERS", "4"))
BATCH = int(os.getenv("NORMALIZE_BATCH", "25"))

SCHEMA = {"type": "json_schema", "json_schema": {"name": "Normalized", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"fixed": {"type": "array", "items": {"type": "string"}}},
    "required": ["fixed"]}}}


def accept(old: str, new: str, terms_lower: list[str]) -> bool:
    """Пропускать правку, только если каждый изменённый фрагмент вводит слово из словаря.

    Модель, несмотря на инструкцию, правит и постороннее: «закроем» → «закроим».
    Просить её этого не делать бесполезно, а проверить механически — легко.
    """
    ow, nw = old.split(), new.split()
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, ow, nw).get_opcodes():
        if op == "equal":
            continue
        span = " ".join(nw[j1:j2]).lower()
        if not any(t in span for t in terms_lower):
            return False
    return True


def build_prompt(terms: str, lines: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    return (
        "Ты корректор транскрипта. Речь русская, но в ней звучат английские технические "
        "термины, и распознавание записало часть из них на слух кириллицей: «харнес» вместо "
        "harness, «клод» вместо Claude, «жира» вместо Jira.\n\n"
        "Ниже словарь правильных написаний и пронумерованные реплики. Приведи термины из "
        "словаря к правильному написанию.\n\n"
        "ЖЁСТКИЕ ПРАВИЛА:\n"
        "- верни РОВНО столько строк, сколько на входе, в том же порядке;\n"
        "- правь ТОЛЬКО термины из словаря и их искажённые формы. Всё остальное — "
        "слова, порядок, окончания, пунктуацию — оставь как есть;\n"
        "- русские склонения сохраняй: «куча харнесов» → «куча harness'ов», "
        "«в клоде» → «в Claude»;\n"
        "- если в реплике нет ничего из словаря — верни её байт в байт;\n"
        "- слово, лишь ПОХОЖЕЕ на термин из словаря, но означающее другое, НЕ ТРОГАЙ: "
        "«воркшоп» это workshop, а не workflow; «скрипт» это не skill. Сомневаешься — не правь;\n"
        "- ничего не добавляй и не удаляй, не переписывай стиль, не дополняй смысл;\n"
        "- нумерацию строк в ответ НЕ включай, только сам текст реплики.\n\n"
        f"=== Словарь ===\n{terms}\n\n"
        f"=== Реплики ({len(lines)}) ===\n{numbered}")


def call(prompt: str, n: int, retries: int = 3) -> list[str] | None:
    prompt, _ = redact.redact_text(prompt)
    for attempt in range(retries):
        try:
            r = HTTP.post(
                f"{NDT_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {NDT_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "max_tokens": 6000, "temperature": 0.1,
                      "chat_template_kwargs": {"enable_thinking": False},
                      "response_format": SCHEMA,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=300)
            r.raise_for_status()
            fixed = json.loads(r.json()["choices"][0]["message"]["content"]).get("fixed", [])
            if len(fixed) != n:                            # не совпало число строк — не верим
                return None
            return [NUM.sub("", x) for x in fixed]         # модель возвращает строки с нумерацией
        except Exception as e:
            if attempt == retries - 1:
                print(f"  батч не разобран: {e}", file=sys.stderr)
                return None


def main():
    ap = argparse.ArgumentParser(description="нормализация терминов по glossary.txt")
    ap.add_argument("timeline")
    ap.add_argument("--glossary", default=None)
    ap.add_argument("--dry-run", action="store_true", help="показать правки, ничего не писать")
    args = ap.parse_args()

    if not NDT_KEY:
        sys.exit("нет NEURALDEEP_API_KEY")
    terms = asr_local.load_glossary(args.glossary or "")
    if not terms:
        sys.exit("пустой glossary.txt — нечего нормализовать")

    path = os.path.abspath(args.timeline)
    with open(path, encoding="utf-8") as f:
        tl = json.load(f)
    tr = tl.get("transcript") or []
    if not tr:
        sys.exit("пустой transcript")

    batches = [(i, tr[i:i + BATCH]) for i in range(0, len(tr), BATCH)]
    print(f"▶ {len(tr)} реплик, {len(terms.split(', '))} терминов, "
          f"{len(batches)} батчей по {BATCH}\n  модель: {MODEL}")

    def do(b):
        i, segs = b
        fixed = call(build_prompt(terms, [s["text"] for s in segs]), len(segs))
        return i, fixed

    terms_lower = [t.strip().lower() for t in terms.split(", ") if t.strip()]
    changes, rejected = [], []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, fixed in ex.map(do, batches):
            if fixed is None:
                continue
            for k, new in enumerate(fixed):
                seg = tr[i + k]
                if new != seg["text"] and not accept(seg["text"], new, terms_lower):
                    rejected.append((seg["start"], seg["text"], new))
                    continue
                if new != seg["text"]:
                    changes.append((seg["start"], seg["text"], new))
                    if not args.dry_run:
                        seg.setdefault("text_asr", seg["text"])   # что было до нормализации
                        seg["text"] = new

    print(f"\n{'нашлось' if args.dry_run else 'исправлено'} реплик: {len(changes)} из {len(tr)}")
    for st, old, new in changes[:25]:
        m, s = divmod(int(st), 60)
        print(f"  {m}:{s:02d}\n     было:  {old[:92]}\n     стало: {new[:92]}")
    if len(changes) > 25:
        print(f"  … ещё {len(changes) - 25}")

    if rejected:
        print(f"\nотклонено фильтром: {len(rejected)} (правка трогала не термины из словаря)")
        for st, old, new in rejected[:8]:
            m, s = divmod(int(st), 60)
            print(f"  {m}:{s:02d}\n     было:  {old[:92]}\n     хотели:{new[:92]}")
        if len(rejected) > 8:
            print(f"  … ещё {len(rejected) - 8}")

    if args.dry_run or not changes:
        if args.dry_run:
            print("\nчтобы применить — повтори без --dry-run")
        return

    tl["srt"] = analyze.build_srt(
        [{**s, "text": (f"{s['speaker_name']}: {s['text']}" if s.get("speaker_name") else s["text"])}
         for s in tr])
    with open(os.path.join(os.path.dirname(path), "subtitles.srt"), "w", encoding="utf-8") as f:
        f.write(tl["srt"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tl, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"\n✔ {path}")


if __name__ == "__main__":
    main()
