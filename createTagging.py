from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    ROOT,
    TRANSCRIPTS_PROCESSED_DIR,
    TRANSCRIPTS_RAW_DIR,
)


TAGGED_DIR = ROOT / "transcripts" / "tagged"
DEFAULT_OUTPUT = TAGGED_DIR / "transcript2.txt"
TAGGING_LOG_DIR = ROOT / "logs" / "tagging"

WINDOW_LINES = 6
LOOKBACK_LINES = 8

STAGE_TAG = "stageSetting"
BREAK_TAG = "breakRelated"
UNCERTAIN_TAG = "uncertain"

GAME_TAGS = {
    "sindarov_esipenko": "Open: Javokhir Sindarov vs Andrey Esipenko. Hints: Sindarov, Javokhir, Esipenko, Andrey.",
    "bluebaum_wei": "Open: Matthias Bluebaum vs Wei Yi. Hints: Bluebaum, Matthias, Wei, Wei Yi.",
    "praggnanandhaa_giri": "Open: Praggnanandhaa R. vs Anish Giri. Hints: Pragg, Praggnanandhaa, Giri, Anish.",
    "caruana_nakamura": "Open: Fabiano Caruana vs Hikaru Nakamura. Hints: Fabi, Fabio, Fabiano, Caruana, Hikaru, Nakamura.",
    "divya_anna_muzychuk": "Women: Divya Deshmukh vs Anna Muzychuk. Hints: Divya, Deshmukh, Anna Muzychuk, Anna.",
    "vaishali_bibisara": "Women: Vaishali Rameshbabu vs Bibisara Assaubayeva. Hints: Vaishali, Rameshbabu, Bibisara, Assaubayeva.",
    "goryachkina_lagno": "Women: Aleksandra Goryachkina vs Kateryna Lagno. Hints: Goryachkina, Aleksandra, Lagno, Kateryna.",
    "zhu_tan": "Women: Zhu Jiner vs Tan Zhongyi. Hints: Zhu, Jiner, Tan, Zhongyi.",
}

ALL_TAGS = [STAGE_TAG, BREAK_TAG, UNCERTAIN_TAG, *GAME_TAGS.keys()]
SWITCH_RE = re.compile(r"\b(?:SWITCH_TO|TAG)\s*[:=]\s*(?P<tag>[A-Za-z0-9_]+)\b", re.IGNORECASE)
STAY_RE = re.compile(r"\bSTAY\b", re.IGNORECASE)


def default_input_path() -> Path:
    raw = TRANSCRIPTS_RAW_DIR / "transcript.txt"
    if raw.exists():
        return raw
    processed = TRANSCRIPTS_PROCESSED_DIR / "transcript.txt"
    if processed.exists():
        return processed
    return raw


def windows(lines: list[str], window_size: int):
    for start in range(0, len(lines), window_size):
        yield start, lines[start : start + window_size]


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
            "num_predict": int(os.environ.get("TAGGING_NUM_PREDICT", "24")),
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
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Ollama. Start it with `ollama serve`.") from exc

    return body.get("response", "").strip()


def normalize_tag(tag: str) -> str | None:
    return tag if tag in ALL_TAGS else None


def allowed_tags(stage_done: bool) -> list[str]:
    if not stage_done:
        return [STAGE_TAG, BREAK_TAG, *GAME_TAGS.keys()]
    return [BREAK_TAG, UNCERTAIN_TAG, *GAME_TAGS.keys()]


def build_switch_prompt(
    current_tag: str,
    stage_done: bool,
    previous_context: list[str],
    current_lines: list[str],
) -> str:
    allowed = allowed_tags(stage_done)
    allowed_text = ", ".join(allowed)
    games_text = "\n".join(f"- {tag}: {label}" for tag, label in GAME_TAGS.items())
    context = "\n".join(previous_context) if previous_context else "(none)"
    lines_text = "\n".join(current_lines)

    return f"""You are detecting topic switches in a chess broadcast transcript.

Current active tag: {current_tag}
Allowed next tags: {allowed_text}

Game tags:
{games_text}

Previous few lines, for continuity only:
{context}

Current 6 lines:
{lines_text}

Task:
Decide whether these current lines explicitly switch away from the current active tag.

Very important rules:
- Usually answer STAY.
- A switch means the live broadcast focus moved to a CURRENT ROUND board. It does not mean a player, opening, or old game was merely mentioned.

Rules when current tag is stageSetting:
- Stay stageSetting for previews, standings, player biographies, experience comparisons, predictions, prize money, format discussion, prior performance, head-to-head records, under-the-radar picks, and general psychology.
- Do NOT switch just because players are named.
- These are stageSetting, not game coverage: "Bluebaum has the best score against the field", "the great Bluebaum sweep", "Esipenko/Sindarov have never played", "which player is flying under the radar", "we will see how you can stop him".
- Switch out of stageSetting only when live games actually start or the broadcast unmistakably moves to a live board.
- Valid kickoff clues: "first move", "make the first move", "kickoff", "liftoff", "the games have started", "we have moves", "on the board", "he/she plays d4/e4/Nf3", "let's go to the first board", "let's look at the position after the first moves".
- Never use uncertain during stageSetting.

Rules when current tag is a game:
- Stay on the current game while commentators discuss plans, psychology, clock pressure, old games, preparation, opening history, or reference games related to the current position.
- Do NOT switch because another player's name appears in a comparison or historical reference.
- Switch to another game only on an explicit live-board transition: "let's look at X's game", "on the X-Y board", "meanwhile in X's game", "X has just played...", "now to the women's game between X and Y".
- Use SWITCH_TO=uncertain only if a live-board transition definitely happens but no board/player is identifiable, e.g. "let's look at this board" or "let's check the other boards" with no names.

Rules when current tag is uncertain:
- Stay uncertain until a live current-game clue identifies the board.
- If the next lines identify a player or move on a current board, switch directly to that game. Example: "Anna played d4" => divya_anna_muzychuk.
- Do not stay uncertain once the board is identifiable.

Rules when current tag is breakRelated:
- Stay breakRelated through break/interview/filler content.
- Leave breakRelated only on clear return to live chess such as "welcome back", "we are back", "back to the games", or direct board commentary.

Global cautions:
- Once stageSetting is over, stageSetting is not allowed.
- If there is any doubt whether live board coverage has started, answer STAY.
- If there is any doubt whether a board switch happened, answer STAY.
- Breaks start on clear telegraphs like "quick break", "take a break", "we'll be back", "after the break".

Output exactly one line:
STAY
or
SWITCH_TO=<one allowed tag>"""


def parse_switch_response(response: str) -> str | None:
    if STAY_RE.search(response):
        return None
    match = SWITCH_RE.search(response)
    if not match:
        return None
    return normalize_tag(match.group("tag"))


def choose_next_tag(
    current_tag: str,
    stage_done: bool,
    previous_context: list[str],
    current_lines: list[str],
    step_index: int,
    dry_run: bool,
    log_path: Path,
) -> str:
    if dry_run:
        return current_tag

    prompt = build_switch_prompt(current_tag, stage_done, previous_context, current_lines)
    response = call_ollama(prompt)
    TAGGING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== STEP {step_index:05d} =====\n")
        log.write(prompt)
        log.write("\n\n--- RESPONSE ---\n")
        log.write(response)
        log.write("\n--- END STEP ---\n")

    requested = parse_switch_response(response)
    if not requested:
        return current_tag
    if requested == STAGE_TAG and stage_done:
        return current_tag
    if requested not in allowed_tags(stage_done):
        return current_tag
    return requested


def tag_transcript(lines: list[str], window_size: int, log_path: Path, dry_run: bool = False) -> list[tuple[str, str]]:
    tagged: list[tuple[str, str]] = []
    current_tag = STAGE_TAG
    stage_done = False

    for step_index, (start, current_lines) in enumerate(windows(lines, window_size), start=1):
        previous_context = lines[max(0, start - LOOKBACK_LINES) : start]
        next_tag = choose_next_tag(current_tag, stage_done, previous_context, current_lines, step_index, dry_run, log_path)

        if current_tag == STAGE_TAG and next_tag != STAGE_TAG:
            stage_done = True
        current_tag = next_tag

        for line in current_lines:
            tagged.append((current_tag, line))

        first_line = start + 1
        last_line = start + len(current_lines)
        print(f"step {step_index}: lines {first_line}-{last_line}, tag={current_tag}")

    return tagged


def render_tagged_transcript(tagged_lines: list[tuple[str, str]]) -> str:
    output: list[str] = []
    current_tag: str | None = None

    for tag, line in tagged_lines:
        if tag != current_tag:
            output.append(f"<{tag}>")
            current_tag = tag
        output.append(line)

    return "\n".join(output) + "\n"


def line_time(line: str) -> str:
    match = re.match(r"^\[(?P<time>[^\]]+)\]", line)
    return match.group("time") if match else "unknown time"


def render_uncertain_report(tagged_lines: list[tuple[str, str]]) -> str:
    sections: list[str] = []
    current: list[tuple[int, str]] = []

    def flush() -> None:
        if not current:
            return
        start_line, start_text = current[0]
        end_line, end_text = current[-1]
        sections.append(
            f"Lines {start_line}-{end_line} | {line_time(start_text)} -> {line_time(end_text)}\n"
            + "\n".join(text for _, text in current)
        )
        current.clear()

    for index, (tag, text) in enumerate(tagged_lines, start=1):
        if tag == UNCERTAIN_TAG:
            current.append((index, text))
        else:
            flush()
    flush()

    if not sections:
        return "No uncertain sections found.\n"
    return "\n\n---\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create transcript2.txt with conservative switch tags.")
    parser.add_argument("transcript", nargs="?", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-lines", type=int, default=WINDOW_LINES)
    parser.add_argument("--uncertain-output", type=Path, default=TAGGED_DIR / "uncertain_sections.txt")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Do not call Ollama; keep the current tag.")
    args = parser.parse_args()

    if args.window_lines < 1:
        raise ValueError("--window-lines must be at least 1")
    if not args.transcript.exists():
        raise FileNotFoundError(f"Transcript not found: {args.transcript}")

    TAGGED_DIR.mkdir(parents=True, exist_ok=True)
    TAGGING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%H%M%S")
    log_path = args.log_file or TAGGING_LOG_DIR / f"{run_stamp}_tagging.log"
    log_path.write_text(
        f"Tagging run started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Transcript: {args.transcript}\n"
        f"Output: {args.output}\n"
        f"Window lines: {args.window_lines}\n"
        f"Dry run: {args.dry_run}\n",
        encoding="utf-8",
    )
    lines = args.transcript.read_text(encoding="utf-8").splitlines()
    tagged_lines = tag_transcript(lines, window_size=args.window_lines, log_path=log_path, dry_run=args.dry_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_tagged_transcript(tagged_lines), encoding="utf-8")
    args.uncertain_output.parent.mkdir(parents=True, exist_ok=True)
    args.uncertain_output.write_text(render_uncertain_report(tagged_lines), encoding="utf-8")
    print(f"Wrote tagged transcript: {args.output}")
    print(f"Wrote uncertain report: {args.uncertain_output}")
    print(f"Wrote tagging log: {log_path}")


if __name__ == "__main__":
    main()
