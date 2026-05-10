from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config import (
    CHUNK_SIZE,
    CURRENT_GAME_FILE,
    DEFAULT_CURRENT_GAME,
    GAMES,
    GAME_ALIASES,
    GAME_LABELS,
    LOGS_DIR,
    MODEL_PROVIDER,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    OPENAI_MODEL,
    OVERLAP,
    PROMPT_FILE,
    RUNTIME_DIR,
    TAIL_CHARS,
    TRANSCRIPTS_PROCESSED_DIR,
    TRANSCRIPTS_RAW_DIR,
)


APPEND_RE = re.compile(
    r"^APPEND\s+(?P<path>games/[A-Za-z0-9_-]+\.md)\s*$",
    re.MULTILINE,
)
CURRENT_GAME_RE = re.compile(r"^CURRENT_GAME\s*=\s*(?P<game>[A-Za-z0-9_-]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class AppendOperation:
    path: Path
    content: str


GENERAL_CONTEXT_MARKERS = (
    "no specific game",
    "no concrete game",
    "no specific board",
    "purely introductory",
    "introductory and contextual",
    "pre-round",
    "standings",
    "tournament context",
    "setting the stage",
)

CONCRETE_CHESS_MARKERS = (
    "move",
    "position",
    "line",
    "variation",
    "opening",
    "white",
    "black",
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "king",
    "castle",
    "castles",
    "e4",
    "d4",
    "c4",
    "nf3",
    "nc6",
)

PLAYER_HINTS = {
    "sindarov_esipenko": ("sindarov", "javokhir", "esipenko", "andrey"),
    "bluebaum_wei": ("bluebaum", "matthias", "wei yi"),
    "praggnanandhaa_giri": ("pragg", "praggnanandhaa", "giri", "anish"),
    "caruana_nakamura": ("fabi", "fabio", "fabiano", "caruana", "hikaru", "nakamura"),
    "divya_humpy": ("divya", "deshmukh", "humpy", "koneru"),
    "vaishali_bibisara": ("vaishali", "rameshbabu", "bibisara", "assaubayeva"),
    "goryachkina_lagno": ("goryachkina", "aleksandra", "lagno", "kateryna"),
    "zhu_tan": ("zhu", "jiner", "tan zhongyi", "zhongyi"),
}

PROMPT_SCAFFOLD_MARKERS = (
    "## 8...Re8",
    "21...Bd4 22.Nxd4 exd4 23.exd4",
    "This natural developing move was presented as the main practical choice",
)


def ensure_project_files() -> None:
    for path in [
        TRANSCRIPTS_RAW_DIR,
        TRANSCRIPTS_PROCESSED_DIR,
        RUNTIME_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for game_file in GAMES.values():
        game_file.parent.mkdir(parents=True, exist_ok=True)
        game_file.touch(exist_ok=True)

    if not CURRENT_GAME_FILE.exists():
        CURRENT_GAME_FILE.write_text(DEFAULT_CURRENT_GAME + "\n", encoding="utf-8")


def read_current_game() -> str:
    if not CURRENT_GAME_FILE.exists():
        return DEFAULT_CURRENT_GAME
    current_game = CURRENT_GAME_FILE.read_text(encoding="utf-8").strip()
    current_game = GAME_ALIASES.get(current_game, current_game)
    return current_game if current_game in GAMES else DEFAULT_CURRENT_GAME


def write_current_game(game: str) -> None:
    game = GAME_ALIASES.get(game, game)
    if game not in GAMES:
        raise ValueError(f"Unknown current game: {game}")
    CURRENT_GAME_FILE.write_text(game + "\n", encoding="utf-8")


def iter_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and less than chunk_size")

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


def file_tail(path: Path, tail_chars: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[-tail_chars:]


def build_file_tails(current_game: str) -> str:
    sections: list[str] = []
    games_to_include = ["setting_the_stage"]
    if current_game != "setting_the_stage":
        games_to_include.append(current_game)

    for game in games_to_include:
        game_file = GAMES[game]
        rel_path = Path("games") / f"{game}.md"
        tail = clean_tail_for_prompt(file_tail(game_file)).strip()
        sections.append(f"--- {rel_path.as_posix()} ---\n{tail or '(empty)'}")
    return "\n\n".join(sections)


def build_prompt(transcript_chunk: str, current_game: str) -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    return template.format(
        current_game=current_game,
        known_games="\n".join(f"- {game}: {GAME_LABELS[game]}" for game in GAMES),
        file_tails=build_file_tails(current_game),
        transcript_chunk=transcript_chunk,
    )


def infer_target_from_chunk(transcript_chunk: str, current_game: str) -> str:
    text = transcript_chunk.lower()
    has_concrete_chess = any(marker in text for marker in CONCRETE_CHESS_MARKERS)
    if not has_concrete_chess:
        return current_game

    for game, hints in PLAYER_HINTS.items():
        if any(hint in text for hint in hints):
            return game
    return current_game


def call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK first: pip install openai") from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before processing transcripts.")

    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    return response.output_text


def call_ollama(prompt: str) -> str:
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL),
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", OLLAMA_NUM_CTX)),
            "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", OLLAMA_NUM_PREDICT)),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        os.environ.get("OLLAMA_URL", OLLAMA_URL),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        timeout = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", OLLAMA_TIMEOUT_SECONDS))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start it with `ollama serve` and make sure the model is pulled."
        ) from exc

    return body.get("response", "").strip()


def call_model(prompt: str) -> str:
    provider = os.environ.get("MODEL_PROVIDER", MODEL_PROVIDER).strip().lower()
    if provider == "ollama":
        return call_ollama(prompt)
    if provider == "openai":
        return call_openai(prompt)
    raise RuntimeError(f"Unsupported MODEL_PROVIDER: {provider}")


def resolve_append_path(relative_path: str) -> Path:
    parts = Path(relative_path).parts
    if len(parts) != 2 or parts[0] != "games":
        raise ValueError(f"Invalid append path: {relative_path}")

    filename = parts[1]
    game = filename.removesuffix(".md")
    game = GAME_ALIASES.get(game, game)
    if game not in GAMES or filename != f"{game}.md":
        if game in GAMES:
            return GAMES[game]
        raise ValueError(f"Unsupported append target: {relative_path}")

    return GAMES[game]


def parse_model_output(text: str, fallback_current_game: str | None = None) -> tuple[list[AppendOperation], str]:
    matches = list(APPEND_RE.finditer(text))
    operations: list[AppendOperation] = []

    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        current_game_match = CURRENT_GAME_RE.search(text, content_start, content_end)
        if current_game_match:
            content_end = current_game_match.start()

        content = text[content_start:content_end].strip()
        if content:
            path = resolve_append_path(match.group("path"))
            content_lower = content.lower()
            if path != GAMES["setting_the_stage"] and any(marker in content_lower for marker in GENERAL_CONTEXT_MARKERS):
                path = GAMES["setting_the_stage"]
            operations.append(AppendOperation(path, content))

    current_game_matches = list(CURRENT_GAME_RE.finditer(text))
    if not current_game_matches:
        if fallback_current_game:
            fallback_current_game = GAME_ALIASES.get(fallback_current_game, fallback_current_game)
            if fallback_current_game in GAMES:
                return operations, fallback_current_game
        raise ValueError("Model output did not include CURRENT_GAME=<game_key>")

    current_game = current_game_matches[-1].group("game")
    current_game = GAME_ALIASES.get(current_game, current_game)
    if current_game not in GAMES:
        raise ValueError(f"Model returned unknown CURRENT_GAME: {current_game}")

    text_lower = text.lower()
    if (
        current_game != "setting_the_stage"
        and operations
        and all(operation.path == GAMES["setting_the_stage"] for operation in operations)
        and any(marker in text_lower for marker in GENERAL_CONTEXT_MARKERS)
    ):
        current_game = "setting_the_stage"

    return operations, current_game


def append_if_new(operation: AppendOperation) -> bool:
    operation.path.parent.mkdir(parents=True, exist_ok=True)
    existing_tail = file_tail(operation.path)
    normalized_content = operation.content.strip()
    if not normalized_content:
        return False
    if any(marker in normalized_content for marker in PROMPT_SCAFFOLD_MARKERS):
        return False
    if normalized_content in existing_tail:
        return False

    filtered_content = filter_duplicate_paragraphs(normalized_content, existing_tail)
    if not filtered_content:
        return False

    needs_blank = operation.path.exists() and operation.path.stat().st_size > 0
    with operation.path.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_blank:
            handle.write("\n")
        handle.write(filtered_content)
        handle.write("\n")
    return True


def normalize_for_duplicate_check(text: str) -> str:
    text = re.sub(r"[*_`#>\-]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_tail_for_prompt(text: str) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in split_paragraphs(text):
        if any(marker in paragraph for marker in PROMPT_SCAFFOLD_MARKERS):
            continue
        normalized = normalize_for_duplicate_check(paragraph)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(paragraph)
    return "\n\n".join(kept[-6:])


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def is_near_duplicate(paragraph: str, existing_paragraphs: list[str]) -> bool:
    normalized = normalize_for_duplicate_check(paragraph)
    if len(normalized) < 80:
        return normalized in {normalize_for_duplicate_check(item) for item in existing_paragraphs}

    for existing in existing_paragraphs:
        existing_normalized = normalize_for_duplicate_check(existing)
        if not existing_normalized:
            continue
        if normalized in existing_normalized or existing_normalized in normalized:
            return True
        if difflib.SequenceMatcher(None, normalized, existing_normalized).ratio() >= 0.86:
            return True
    return False


def filter_duplicate_paragraphs(content: str, existing_tail: str) -> str:
    existing_paragraphs = split_paragraphs(existing_tail)
    kept: list[str] = []
    for paragraph in split_paragraphs(content):
        comparison_pool = existing_paragraphs + kept
        if not is_near_duplicate(paragraph, comparison_pool):
            kept.append(paragraph)
    return "\n\n".join(kept).strip()


def process_chunk(transcript_chunk: str, dry_run: bool = False) -> tuple[int, str, str]:
    current_game = infer_target_from_chunk(transcript_chunk, read_current_game())
    prompt = build_prompt(transcript_chunk, current_game)
    output = call_model(prompt)
    operations, updated_game = parse_model_output(output, fallback_current_game=current_game)

    appended = 0
    if not dry_run:
        for operation in operations:
            if append_if_new(operation):
                appended += 1
        write_current_game(updated_game)

    return appended, updated_game, output


def process_transcript(path: Path, dry_run: bool = False) -> None:
    text = path.read_text(encoding="utf-8")
    for number, chunk in enumerate(iter_chunks(text), start=1):
        log_path = LOGS_DIR / f"{path.stem}.chunk-{number:04d}.txt"
        try:
            appended, current_game, output = process_chunk(chunk, dry_run=dry_run)
        except Exception as exc:
            error_path = LOGS_DIR / f"{path.stem}.chunk-{number:04d}.error.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            raise
        log_path.write_text(output, encoding="utf-8")
        print(f"chunk {number}: appended {appended} block(s), current_game={current_game}")

    if not dry_run:
        destination = TRANSCRIPTS_PROCESSED_DIR / path.name
        if path.resolve() != destination.resolve():
            shutil.move(str(path), str(destination))


def process_raw_transcripts(dry_run: bool = False) -> None:
    ensure_project_files()
    transcript_paths = sorted(TRANSCRIPTS_RAW_DIR.glob("*.txt"))
    if not transcript_paths:
        print(f"No .txt transcripts found in {TRANSCRIPTS_RAW_DIR}")
        return
    for path in transcript_paths:
        process_transcript(path, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process chess broadcast transcript chunks.")
    parser.add_argument("transcript", nargs="?", type=Path, help="Optional transcript file to process.")
    parser.add_argument("--dry-run", action="store_true", help="Call the model and log output without appending.")
    parser.add_argument("--init", action="store_true", help="Create runtime folders and game files.")
    args = parser.parse_args()

    ensure_project_files()

    if args.init:
        print("Project files are ready.")
        return

    if args.transcript:
        process_transcript(args.transcript, dry_run=args.dry_run)
    else:
        process_raw_transcripts(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
