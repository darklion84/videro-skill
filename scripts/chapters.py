#!/usr/bin/env python3
"""Второй проход: сцены → главы + отобранные хайлайты (без повторных vision-вызовов).

Зачем: analyze.py размечает каждую сцену независимо и параллельно, поэтому сцена
не знает тем соседей — единого словаря topic не возникает, и группировка по нему
даёт по главе на сцену. Здесь все сцены уходят в один текстовый запрос, где модель
видит их целиком и режет на главы, а заодно честно отбирает хайлайты вместо
«важно всё».

Вход:  timeline.json от videro.py
Выход: в него же дописывается "chapters", а "highlights" заменяется на отобранные.
       Оригинал один раз сохраняется рядом как timeline.raw.json.

Использование:
  python scripts/chapters.py videro-out/видео/timeline.json
  python scripts/chapters.py .../timeline.json --chapters 6 --max-highlights 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hub_retry        # noqa: E402
import redact           # noqa: E402

HTTP = hub_retry.session()

NDT_BASE = (os.getenv("ND_BASE_URL", "https://api.neuraldeep.ru/v1")).rstrip("/")
NDT_KEY = os.getenv("NEURALDEEP_API_KEY", "")
MODEL = os.getenv("NDT_CHAPTERS_MODEL", os.getenv("NDT_ASK_MODEL",
                  os.getenv("NDT_VISION_MODEL", "qwen3.6-fp8")))
OCR_CAP = int(os.getenv("CHAPTERS_OCR_CAP", "200"))
# 450, а не меньше: на 260 фраза рубится посередине, и глава про десктопное приложение
# получила название про VS Code — модель цеплялась за единственное уцелевшее имя продукта
SPEECH_CAP = int(os.getenv("CHAPTERS_SPEECH_CAP", "450"))

SCHEMA = {"type": "json_schema", "json_schema": {"name": "Chapters", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "properties": {
        "chapters": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "start_scene": {"type": "integer"},
                "title": {"type": "string"},
                "summary": {"type": "string"}},
            "required": ["start_scene", "title", "summary"]}},
        "highlights": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "scene": {"type": "integer"},
                "reason": {"type": "string"},
                "importance": {"type": "integer"}},
            "required": ["scene", "reason", "importance"]}}},
    "required": ["chapters", "highlights"]}}}


def fmt_ts(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def dominant_speakers(scenes: list[dict], transcript: list[dict]) -> list[str]:
    """Кто больше всех говорит в каждой сцене. Смена докладчика — сильный признак границы,
    но в описании сцены её нет: analyze.py пишет только картинку и текст с экрана."""
    out = []
    for s in scenes:
        agg: dict[str, float] = {}
        for t in transcript:
            if t["start"] >= s["end"]:
                break
            ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if ov > 0:
                who = t.get("speaker_name") or t.get("speaker")
                if who:
                    agg[who] = agg.get(who, 0.0) + ov
        out.append(max(agg, key=agg.get) if agg else "")
    return out


def scene_speech(scenes: list[dict], transcript: list[dict], cap: int) -> list[str]:
    """Что говорят в каждой сцене. Без этого глава называется по картинке: на экране
    может быть открыт редактор, пока обсуждают совсем другой инструмент."""
    out = []
    for s in scenes:
        parts = []
        for t in transcript:
            if t["start"] >= s["end"]:
                break
            if t["end"] > s["start"] and t.get("text"):
                parts.append(t["text"])
        out.append(" ".join(" ".join(parts).split())[:cap])
    return out


def speaker_boundaries(dom: list[str], min_run: int) -> set[int]:
    """Сцены, где слово переходит к другому человеку и он его удерживает.

    Просить об этом модель бесполезно: она держит цель по числу глав и подсказку
    игнорирует (на лекции границу получили 5 участков из 27). Поэтому список
    считается здесь и навязывается — модель отвечает только за название.
    """
    bounds, prev, i = set(), None, 0
    while i < len(dom):
        j = i
        while j + 1 < len(dom) and dom[j + 1] == dom[i]:
            j += 1
        if dom[i] and j - i + 1 >= min_run:
            if i > 0 and dom[i] != prev:
                bounds.add(i)
            prev = dom[i]                  # два куска подряд одного и того же — не граница
        i = j + 1
    return bounds


def build_prompt(scenes: list[dict], n_chapters: int, n_highlights: int, offset: int = 0,
                 prev_title: str = "", speakers: list[str] | None = None,
                 must_start: list[int] | None = None, speech: list[str] | None = None) -> str:
    """offset — глобальный номер первой сцены куска, чтобы индексы в ответе были сквозными.

    prev_title — глава, на которой оборвался предыдущий кусок. Без неё тема, идущая
    через границу, разрезается пополам; с ней модель повторяет заголовок дословно,
    и склейка делается уже детерминированно.
    """
    lines = []
    for k, s in enumerate(scenes):
        i = offset + k
        ocr = " ".join((s.get("on_screen_text") or "").split())[:OCR_CAP]
        who = (speakers[k] if speakers and k < len(speakers) else "")
        said = (speech[k] if speech and k < len(speech) else "")
        lines.append(
            f"#{i} [{fmt_ts(s['start'])}–{fmt_ts(s['end'])}] "
            f"({s.get('screen_type') or '?'}) {s.get('caption') or ''}\n"
            + (f"    говорит: {who}\n" if who else "")
            + (f"    речь: {said}\n" if said else "")
            + f"    что происходит: {s.get('action') or ''}\n"
            f"    на экране: {ocr or '(нет)'}")
    return (
        "Ты редактор видеокурса. Ниже — последовательные сцены одного видео, каждая размечена "
        "отдельно и независимо, поэтому темы у них при близком смысле расходятся в формулировках.\n\n"
        + (f"Это продолжение середины видео. Предыдущий кусок оборвался на главе «{prev_title}». "
           f"Если сцена #{offset} и дальше всё ещё про это, назови первую главу ДОСЛОВНО "
           f"«{prev_title}» — её склеят с предыдущей. Если тема сменилась, дай новое название.\n\n"
           if prev_title else "")
        + "Задача 1 — ОГЛАВЛЕНИЕ. Склей сцены в непрерывные главы:\n"
        f"- целься примерно в {n_chapters} глав, но приоритет у смысла, а не у числа;\n"
        "- глава = связный кусок работы («установка нод», «разбор графа», «примеры результатов»), "
        "а не одна сцена;\n"
        "- тему определяет поле «речь», а не картинка: на экране может быть открыт один "
        "инструмент, пока обсуждают совсем другой. Называй главу по тому, О ЧЁМ ГОВОРЯТ;\n"
        + (f"- ОБЯЗАТЕЛЬНО начни новую главу со сцен {', '.join('#' + str(b) for b in must_start)}: "
           "там слово переходит к другому человеку. Назови такую главу по тому, о чём говорит "
           "новый докладчик, а не по картинке на экране — она может не поменяться;\n"
           if must_start else "")
        + "- start_scene — номер ПЕРВОЙ сцены главы; главы идут подряд и не пересекаются; "
        f"первая глава обязана начинаться со сцены #{offset};\n"
        "- title: 3–7 слов, конкретно и по делу, без «Введение в тему» и прочей воды; "
        "названия глав не должны дублировать друг друга;\n"
        "- summary: 1–2 предложения о содержании главы.\n\n"
        f"Задача 2 — ХАЙЛАЙТЫ. Отбери НЕ БОЛЕЕ {n_highlights} действительно ключевых сцен:\n"
        "- бери момент, ради которого это видео открывают: результат, ключевая настройка, "
        "неочевидный приём, ошибка и её починка;\n"
        "- НЕ бери проходные, повторяющиеся и переходные сцены;\n"
        "- если по-настоящему ценных моментов меньше лимита — верни меньше, добивать не надо;\n"
        "- importance 1–5, и шкалой пользуйся целиком: 5 — единичные, 3 — просто полезное;\n"
        "- reason: чем именно ценно, конкретно, одна строка.\n\n"
        f"=== Сцены ({len(scenes)}) ===\n" + "\n".join(lines))


def call_model(prompt: str, retries: int = 3) -> dict | None:
    # страховка на случай старого, ещё не очищенного timeline.json: в промпт уходит
    # on_screen_text, а там живут токены с экрана. Чистим перед отправкой, а не после.
    prompt, leaked = redact.redact_text(prompt)
    if leaked:
        print(f"  ! из промпта вырезано секретов: {len(leaked)} "
              f"({', '.join(sorted(set(leaked)))}) — прогони ./run.sh redact на этом таймлайне",
              file=sys.stderr)

    for attempt in range(retries):
        try:
            r = HTTP.post(
                f"{NDT_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {NDT_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "max_tokens": 8000, "temperature": 0.2,
                      "chat_template_kwargs": {"enable_thinking": False},
                      "response_format": SCHEMA,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=300)
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == retries - 1:
                print(f"  запрос не удался: {e}", file=sys.stderr)
                return None


def force_splits(out: list[dict], scenes: list[dict], must: set[int]) -> list[dict]:
    """Дорезать главы там, где модель не поставила обязательную границу."""
    for b in sorted(must):
        for k, c in enumerate(out):
            if c["scene_from"] < b <= c["scene_to"]:
                tail = {"start": scenes[b]["start"], "end": c["end"],
                        "title": (scenes[b].get("caption") or "").strip() or "Другой докладчик",
                        "summary": (scenes[b].get("action") or "").strip(),
                        "scene_from": b, "scene_to": c["scene_to"]}
                c["end"] = scenes[b - 1]["end"]
                c["scene_to"] = b - 1
                out.insert(k + 1, tail)
                break
    return out


def make_chapters(raw: list[dict], scenes: list[dict], duration: float) -> list[dict]:
    """Индексы сцен от модели → непрерывные главы с таймкодами."""
    seen, starts = set(), []
    for ch in raw:
        i = ch.get("start_scene")
        if isinstance(i, int) and 0 <= i < len(scenes) and i not in seen:
            seen.add(i)
            starts.append((i, ch))
    starts.sort(key=lambda x: x[0])
    if not starts:
        return []
    if starts[0][0] != 0:                      # первая глава всегда от начала видео
        starts[0] = (0, starts[0][1])

    out = []
    for k, (i, ch) in enumerate(starts):
        end_scene = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(scenes) - 1
        title = (ch.get("title") or "").strip() or f"Глава {k + 1}"
        summary = (ch.get("summary") or "").strip()
        end = scenes[end_scene]["end"] if end_scene >= i else duration

        # тема, перешедшая через границу куска, приходит дважды под одним заголовком
        if out and out[-1]["title"].casefold() == title.casefold():
            out[-1]["end"] = end
            out[-1]["scene_to"] = end_scene
            if len(summary) > len(out[-1]["summary"]):
                out[-1]["summary"] = summary
            continue

        out.append({"start": scenes[i]["start"], "end": end, "title": title,
                    "summary": summary, "scene_from": i, "scene_to": end_scene})
    return out


def make_highlights(raw: list[dict], scenes: list[dict], limit: int) -> list[dict]:
    seen, out = set(), []
    for h in raw:
        i = h.get("scene")
        if not (isinstance(i, int) and 0 <= i < len(scenes)) or i in seen:
            continue
        seen.add(i)
        s = scenes[i]
        imp = h.get("importance")
        out.append({"start": s["start"], "end": s["end"], "caption": s.get("caption") or "",
                    "reason": (h.get("reason") or "").strip(),
                    "importance": imp if isinstance(imp, int) and 1 <= imp <= 5 else 3,
                    "scene": i})
    out.sort(key=lambda x: (-x["importance"], x["start"]))
    # importance решает, ЧТО попадёт в список; наружу отдаём по времени —
    # так список читается вдоль видео, и поиск текущего хайлайта по таймкоду
    # работает корректно (он рассчитан на возрастающие start)
    return sorted(out[:limit], key=lambda x: x["start"])


def main():
    ap = argparse.ArgumentParser(description="timeline.json → главы + отобранные хайлайты")
    ap.add_argument("timeline", help="путь к timeline.json")
    ap.add_argument("--chapters", type=int, default=None, help="целевое число глав")
    ap.add_argument("--max-highlights", type=int, default=None, help="потолок хайлайтов")
    ap.add_argument("--chunk", type=int, default=60,
                    help="сцен в одном запросе; длинное видео режется на куски (default 60)")
    ap.add_argument("--speaker-split", type=int, default=3, metavar="N",
                    help="ставить границу главы, где новый докладчик держит слово N+ сцен "
                         "(0 — не учитывать говорящих; default 3)")
    args = ap.parse_args()

    if not NDT_KEY:
        sys.exit("нет NEURALDEEP_API_KEY — см. README, раздел «1. Получить токен»")

    path = os.path.abspath(args.timeline)
    with open(path, encoding="utf-8") as f:
        tl = json.load(f)
    scenes = tl.get("scenes") or []
    if len(scenes) < 2:
        sys.exit(f"в {path} меньше двух сцен — группировать нечего")
    duration = float(tl.get("duration_sec") or scenes[-1]["end"])

    n_ch = args.chapters or max(3, min(40, round(duration / 120)))
    n_ch = min(n_ch, max(2, len(scenes) // 2))          # глав не больше, чем сцен пополам
    n_hl = args.max_highlights or max(3, min(30, round(duration / 90)))

    # длинное видео не влезает в один запрос — режем на куски по CHUNK сцен.
    # Нумерация сцен в промпте сквозная, поэтому индексы из ответов склеиваются без пересчёта.
    chunk = max(10, args.chunk)
    parts = [(i, scenes[i:i + chunk]) for i in range(0, len(scenes), chunk)] or []

    print(f"▶ {len(scenes)} сцен, {duration:.0f}s → цель: ~{n_ch} глав, ≤{n_hl} хайлайтов")
    print(f"  модель: {MODEL}")
    if len(parts) > 1:
        print(f"  {len(parts)} запроса(ов) по ≤{chunk} сцен")

    transcript = tl.get("transcript") or []
    spk = dominant_speakers(scenes, transcript)
    say = scene_speech(scenes, transcript, SPEECH_CAP)
    must = speaker_boundaries(spk, args.speaker_split) if any(spk) and args.speaker_split else set()
    if must:
        print(f"  диаризация: {len({s for s in spk if s})} говорящих, "
              f"{len(must)} смен слова станут границами глав")

    raw_ch, raw_hl, prev_title = [], [], ""
    for k, (offset, part) in enumerate(parts, 1):
        # цель по куску пропорциональна его размеру: последний кусок обычно короче,
        # и равная доля глав дробила бы его вдвое мельче остальных
        share = len(part) / len(scenes)
        per_ch = max(1, round(n_ch * share))
        per_hl = max(2, round(n_hl * share))
        if len(parts) > 1:
            print(f"  [{k}/{len(parts)}] сцены {offset}–{offset + len(part) - 1} "
                  f"→ ~{per_ch} глав, ≤{per_hl} хайлайтов", flush=True)
        in_part = sorted(b for b in must if offset < b < offset + len(part))
        data = call_model(build_prompt(part, per_ch, per_hl, offset, prev_title,
                                       spk[offset:offset + len(part)], in_part,
                                       say[offset:offset + len(part)]))
        if data is None:
            print(f"  кусок {k} не разобран — пропускаю", file=sys.stderr)
            prev_title = ""
            continue
        got = data.get("chapters") or []
        raw_ch += got
        raw_hl += data.get("highlights") or []
        prev_title = (max(got, key=lambda c: c.get("start_scene", 0)).get("title") or ""
                      ).strip() if got else ""

    if not raw_ch:
        sys.exit("модель не ответила ни по одному куску — timeline.json не тронут")

    chapters = force_splits(make_chapters(raw_ch, scenes, duration), scenes, must)
    highlights = make_highlights(raw_hl, scenes, n_hl)   # отбор лучших уже по всему видео
    if not chapters:
        sys.exit("модель не вернула ни одной валидной главы — timeline.json не тронут")

    picked = {h["scene"] for h in highlights}
    for i, s in enumerate(scenes):                       # сцены в согласии со списком хайлайтов
        s["highlight"] = i in picked
        s["highlight_reason"] = next((h["reason"] for h in highlights if h["scene"] == i), "")

    backup = os.path.join(os.path.dirname(path), "timeline.raw.json")
    if not os.path.exists(backup):                       # только исходник, повторы не затирают
        os.replace(path, backup)
        print(f"  оригинал сохранён: {backup}")

    tl["chapters"] = chapters
    tl["highlights"] = highlights
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tl, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

    print(f"\n✔ {len(chapters)} глав, {len(highlights)} хайлайтов → {path}\n")
    for c in chapters:
        print(f"  {fmt_ts(c['start'])}  {c['title']}  (сцены {c['scene_from']}–{c['scene_to']})")


if __name__ == "__main__":
    main()
