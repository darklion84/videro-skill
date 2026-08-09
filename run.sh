#!/usr/bin/env bash
# Обёртка: скрипты videro читают только os.getenv и сами .env не подхватывают.
#   ./run.sh видео.mp4 [флаги]              → разметка, ASR через SpeechCore (scripts/videro.py)
#   ./run.sh local видео.mp4 [флаги]        → то же, но ASR локальным whisper (scripts/videro_local.py)
#   ./run.sh asr видео.mp4 --lang en        → только распознавание речи (scripts/asr_local.py)
#   ./run.sh chapters .../timeline.json     → главы + хайлайты (scripts/chapters.py)
#   ./run.sh ask .../timeline.json "вопрос" → Q&A (scripts/ask.py)
#   ./run.sh serve videro-out/имя [порт]    → поднять плеер и напечатать ссылку
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a

if [ "${1:-}" = serve ]; then
  dir="${2:?укажи папку прогона, например videro-out/krea2-lora-full}"
  port="${3:-8000}"
  [ -f "$dir/timeline.json" ] || { echo "нет $dir/timeline.json" >&2; exit 1; }
  echo "→ http://localhost:$port/player/?src=/${dir%/}/"
  exec .venv/bin/python scripts/serve.py . --port "$port"
fi

script=videro.py
case "${1:-}" in
  chapters|ask|videro) script="$1.py"; shift ;;
  local)               script=videro_local.py; shift ;;
  asr)                 script=asr_local.py; shift ;;
  diarize)             script=diarize.py; shift ;;
  speakers)            script=speakers.py; shift ;;
  redact)              script=redact.py; shift ;;
esac

if [ "$script" = videro.py ] && [ -z "${SPEECHCORE_TOKEN:-}" ]; then
  echo "warn: SPEECHCORE_TOKEN пуст — транскрипт и субтитры будут пустыми" >&2
fi

exec .venv/bin/python "scripts/$script" "$@"
