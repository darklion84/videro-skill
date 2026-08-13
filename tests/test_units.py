#!/usr/bin/env python3
"""Тесты на чистые функции — без сети, моделей и ключей.

Покрыто то, что за время работы реально ломалось: фильтр правок пропускал порчу
идентификаторов, очистка секретов задваивала маркер, границы глав терялись на
стыках кусков. Каждый тест здесь — след конкретной ошибки, а не «для покрытия».

  ./run.sh test
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import chapters      # noqa: E402
import diarize       # noqa: E402
import normalize     # noqa: E402
import redact        # noqa: E402
import speakers      # noqa: E402

TERMS = ["claude", "jira", "confluence", "harness", "skill", "vs code", "llm"]


class TestNormalizeFilter(unittest.TestCase):
    """normalize.accept — единственная защита от того, что модель испортит текст."""

    def ok(self, old, new):
        self.assertTrue(normalize.accept(old, new, TERMS), f"должно приниматься: {new}")

    def no(self, old, new):
        self.assertFalse(normalize.accept(old, new, TERMS), f"должно отклоняться: {new}")

    def test_real_calque_accepted(self):
        self.ok("объяснение термина харнесс", "объяснение термина harness")
        self.ok("агента Клод и копирование", "агента Claude и копирование")

    def test_identifier_case_change_rejected(self):
        # ловил реальную порчу: имя папки на диске — строчными
        self.no("папку 'confluence_access' в папку", "папку 'Confluence_access' в папку")
        self.no("лежит jira_access внутри", "лежит Jira_access внутри")

    def test_meaning_inversion_rejected(self):
        # модель добавляла отрицание и переворачивала смысл фразы
        self.no("будем скорее всего 99 процентов", "не будем скорее всего 99 процентов")

    def test_unrelated_edit_rejected(self):
        self.no("полностью закроем вопрос", "полностью закроим вопрос")

    def test_untouched_text_accepted(self):
        self.ok("обычная реплика без терминов", "обычная реплика без терминов")


class TestRedact(unittest.TestCase):
    """redact — утечку создаёт сам пайплайн, поэтому проверяем оба направления."""

    def test_known_formats(self):
        for s in ("JIRA_TOKEN=ATATT3xFfGF0abcdefghij1234567890",
                  "почта me@corp.com",
                  "Bearer abcdefghijklmnop1234567890"):
            out, hits = redact.redact_text(s)
            self.assertTrue(hits, f"не поймано: {s}")
            self.assertIn(redact.MARK, out)

    def test_unknown_format_blob(self):
        # код авторизации Claude Code: без префикса и без КЛЮЧ=, ловится по форме
        out, hits = redact.redact_text(
            "Paste this into Claude Code: qskQ48e1EWxTn4bRmZaLpQ7vHcYd2KfJ9sNgU3oIeAtBw6XrM5yVi")
        self.assertTrue(hits)
        self.assertIn(redact.MARK, out)

    def test_false_positives_left_alone(self):
        for s in ("Version: 1.132.0 Commit: df53daabb1c9e77f4a2b6d8e0c135792fa4bcde1",
                  "/Users/darklion/Documents/projects/videro/videro-skill",
                  "Name * jira_client_integration_prod",
                  "Дальше нам нужно авторизоваться, это делается одной командой"):
            _, hits = redact.redact_text(s)
            self.assertFalse(hits, f"ложное срабатывание: {s}")

    def test_variable_name_survives(self):
        out, _ = redact.redact_text("JIRA_TOKEN=ATATT3xFfGF0abcdefghij1234567890")
        self.assertTrue(out.startswith("JIRA_TOKEN="), out)

    def test_no_marker_doubling(self):
        # маркер содержит пробел и запятую — защита от повторной замены на этом ломалась
        out, _ = redact.redact_text("JIRA_TOKEN=ATATT3xFfGF0abcdefghij1234567890")
        self.assertEqual(out.count("⚠"), 2, out)

    def test_idempotent(self):
        once, _ = redact.redact_text("JIRA_TOKEN=ATATT3xFfGF0abcdefghij1234567890")
        twice, hits = redact.redact_text(once)
        self.assertEqual(once, twice)
        self.assertFalse(hits)

    def test_timeline_and_ocr(self):
        tl = {"transcript": [{"start": 0, "text": "JIRA_TOKEN=ATATT3xFfGF0abcdefghij1234567890"}],
              "scenes": [{"start": 0, "on_screen_text": "почта boss@corp.com"}], "srt": ""}
        found = redact.redact_timeline(tl)
        self.assertEqual(len(found), 2)
        self.assertNotIn("ATATT", tl["transcript"][0]["text"])
        self.assertNotIn("@corp.com", tl["scenes"][0]["on_screen_text"])


class TestSpeakerBoundaries(unittest.TestCase):
    """Границы глав по смене докладчика: просить модель бесполезно, считаем сами."""

    def test_new_speaker_holding_floor(self):
        dom = ["A"] * 5 + ["B"] * 4
        self.assertEqual(chapters.speaker_boundaries(dom, 3), {5})

    def test_short_interjection_ignored(self):
        dom = ["A"] * 5 + ["B"] * 2 + ["A"] * 5      # реплика из зала — не глава
        self.assertEqual(chapters.speaker_boundaries(dom, 3), set())

    def test_same_speaker_twice_not_a_boundary(self):
        dom = ["A"] * 4 + ["B"] * 4 + ["A"] * 4
        self.assertEqual(chapters.speaker_boundaries(dom, 3), {4, 8})

    def test_start_is_never_a_boundary(self):
        self.assertNotIn(0, chapters.speaker_boundaries(["A"] * 9, 3))


class TestChapterAssembly(unittest.TestCase):
    def scenes(self, n):
        return [{"start": i * 10.0, "end": (i + 1) * 10.0, "caption": f"кадр {i}",
                 "action": "", "highlight": False, "importance": 3} for i in range(n)]

    def test_continuous_and_covers_all(self):
        sc = self.scenes(10)
        raw = [{"start_scene": 0, "title": "Начало", "summary": ""},
               {"start_scene": 4, "title": "Середина", "summary": ""}]
        ch = chapters.make_chapters(raw, sc, 100.0)
        self.assertEqual(ch[0]["scene_from"], 0)
        self.assertEqual(ch[-1]["scene_to"], 9)
        for a, b in zip(ch, ch[1:]):
            self.assertEqual(a["scene_to"] + 1, b["scene_from"], "дыра между главами")
            self.assertAlmostEqual(a["end"], b["start"], places=6)

    def test_identical_adjacent_titles_merged(self):
        # тема, перешедшая через границу куска, приходит дважды под одним именем
        sc = self.scenes(10)
        raw = [{"start_scene": 0, "title": "Настройка", "summary": "a"},
               {"start_scene": 5, "title": "настройка", "summary": "подробнее"}]
        ch = chapters.make_chapters(raw, sc, 100.0)
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0]["summary"], "подробнее")   # берётся более длинное

    def test_garbage_indices_dropped(self):
        sc = self.scenes(5)
        raw = [{"start_scene": 99, "title": "мимо", "summary": ""},
               {"start_scene": 2, "title": "норм", "summary": ""}]
        ch = chapters.make_chapters(raw, sc, 50.0)
        self.assertTrue(ch)
        self.assertEqual(ch[0]["scene_from"], 0, "первая глава обязана начинаться с нуля")

    def test_force_split_when_model_ignored(self):
        sc = self.scenes(10)
        ch = chapters.make_chapters([{"start_scene": 0, "title": "Одна", "summary": ""}], sc, 100.0)
        self.assertEqual(len(ch), 1)
        ch = chapters.force_splits(ch, sc, {6})
        self.assertEqual([c["scene_from"] for c in ch], [0, 6])
        self.assertEqual(ch[0]["scene_to"], 5)

    def test_highlights_sorted_by_time(self):
        sc = self.scenes(10)
        raw = [{"scene": 7, "reason": "x", "importance": 5},
               {"scene": 2, "reason": "y", "importance": 3}]
        hl = chapters.make_highlights(raw, sc, 10)
        self.assertEqual([h["scene"] for h in hl], [2, 7], "список должен идти по времени")


class TestSpeakersRoster(unittest.TestCase):
    def transcript(self):
        return [{"start": 0, "end": 10, "text": "Длинная содержательная реплика для опознания",
                 "speaker": "SPEAKER_00"},
                {"start": 10, "end": 11, "text": "Ага", "speaker": "SPEAKER_01"},
                {"start": 11, "end": 20, "text": "Ещё одна длинная реплика того же человека",
                 "speaker": "SPEAKER_00"}]

    def test_sorted_by_talk_time(self):
        r = speakers.build_roster(self.transcript(), {})
        self.assertEqual([s["id"] for s in r["speakers"]], ["SPEAKER_00", "SPEAKER_01"])

    def test_names_preserved_on_rebuild(self):
        r = speakers.build_roster(self.transcript(), {"SPEAKER_00": "Лектор"})
        self.assertEqual(r["speakers"][0]["name"], "Лектор")

    def test_short_lines_fallback(self):
        r = speakers.build_roster(self.transcript(), {})
        short = [s for s in r["speakers"] if s["id"] == "SPEAKER_01"][0]
        self.assertTrue(short["samples"], "у кого нет длинных реплик — показываем что есть")

    def test_segments_without_speaker_counted(self):
        tr = self.transcript() + [{"start": 20, "end": 25, "text": "ничей", "speaker": None}]
        self.assertEqual(speakers.build_roster(tr, {})["segments_without_speaker"], 1)


class TestDiarizeStamp(unittest.TestCase):
    """Спикер приклеивается к готовым репликам по максимальному перекрытию."""

    def test_max_overlap_wins(self):
        tr = [{"start": 0.0, "end": 10.0, "text": "a"}]
        turns = [{"start": 0.0, "end": 3.0, "speaker": "A"},
                 {"start": 3.0, "end": 10.0, "speaker": "B"}]
        self.assertEqual(diarize.stamp(tr, turns), 1)
        self.assertEqual(tr[0]["speaker"], "B")

    def test_no_overlap_leaves_none(self):
        tr = [{"start": 0.0, "end": 5.0, "text": "a"}]
        self.assertEqual(diarize.stamp(tr, [{"start": 90.0, "end": 99.0, "speaker": "A"}]), 0)
        self.assertIsNone(tr[0]["speaker"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
