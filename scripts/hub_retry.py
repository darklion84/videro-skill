#!/usr/bin/env python3
"""Ретраи с backoff для запросов к Hub. Лечит потерю сцен и субтитров на 429.

Апстрим под лимитами разваливается тихо:
  - describe_scene не ретраится совсем — один 429 и сцена превращается в пустышку
    с текстом ошибки в action;
  - _fix_call ретраит трижды, но без пауз, поэтому все попытки попадают в то же
    окно лимита и коррекция субтитров молча пропадает целиком.

Чиним на уровне транспорта: analyze.requests подменяется на Session с urllib3-Retry.
Вызовы вида requests.post(...) внутри analyze продолжают работать как были, но
теперь сами пережидают 429/5xx, уважая Retry-After. Правок в analyze.py не нужно.
"""
from __future__ import annotations

import requests
from urllib3.util.retry import Retry

RETRY = Retry(
    total=5,
    backoff_factor=2.0,               # 2s, 4s, 8s, 16s, 32s
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
    respect_retry_after_header=True,
    raise_on_status=False,            # исчерпали попытки → обычный ответ, его разберёт raise_for_status
)


def session() -> requests.Session:
    """Session, которая сама пережидает 429/5xx."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=RETRY, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def apply(analyze_module) -> None:
    """Подменить analyze.requests на сессию с ретраями."""
    analyze_module.requests = session()
