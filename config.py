from pathlib import Path


ROOT = Path(__file__).resolve().parent

TRANSCRIPTS_RAW_DIR = ROOT / "transcripts" / "raw"
TRANSCRIPTS_PROCESSED_DIR = ROOT / "transcripts" / "processed"
GAMES_DIR = ROOT / "games"
RUNTIME_DIR = ROOT / "runtime"
PROMPTS_DIR = ROOT / "prompts"
LOGS_DIR = ROOT / "logs"

CURRENT_GAME_FILE = RUNTIME_DIR / "current_game.txt"
PROMPT_FILE = PROMPTS_DIR / "process_chunk.txt"

CHUNK_SIZE = 1800
OVERLAP = 300
TAIL_CHARS = 1200

MODEL_PROVIDER = "ollama"
OPENAI_MODEL = "gpt-4.1-mini"
OLLAMA_MODEL = "qwen3.5:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_SECONDS = 120
OLLAMA_NUM_PREDICT = 700
OLLAMA_NUM_CTX = 4096
OLLAMA_KEEP_ALIVE = "0s"

# One append-only annotation file per broadcast section/game.
# Rename these keys to match the actual event pairings when known.
GAMES = {
    "setting_the_stage": GAMES_DIR / "setting_the_stage.md",
    "sindarov_esipenko": GAMES_DIR / "sindarov_esipenko.md",
    "bluebaum_wei": GAMES_DIR / "bluebaum_wei.md",
    "praggnanandhaa_giri": GAMES_DIR / "praggnanandhaa_giri.md",
    "caruana_nakamura": GAMES_DIR / "caruana_nakamura.md",
    "divya_humpy": GAMES_DIR / "divya_humpy.md",
    "vaishali_bibisara": GAMES_DIR / "vaishali_bibisara.md",
    "goryachkina_lagno": GAMES_DIR / "goryachkina_lagno.md",
    "zhu_tan": GAMES_DIR / "zhu_tan.md",
}

GAME_LABELS = {
    "setting_the_stage": "Setting the stage / standings / pre-round context",
    "sindarov_esipenko": "Open: Javokhir Sindarov vs Andrey Esipenko. Hints: Sindarov, Javokhir, Esipenko, Andrey.",
    "bluebaum_wei": "Open: Matthias Bluebaum vs Wei Yi. Hints: Bluebaum, Matthias, Wei, Wei Yi.",
    "praggnanandhaa_giri": "Open: Praggnanandhaa R. vs Anish Giri. Hints: Pragg, Praggnanandhaa, Giri, Anish.",
    "caruana_nakamura": "Open: Fabiano Caruana vs Hikaru Nakamura. Hints: Fabi, Fabio, Fabiano, Caruana, Hikaru, Nakamura.",
    "divya_humpy": "Women: Divya Deshmukh vs Humpy Koneru. Hints: Divya, Deshmukh, Humpy, Koneru.",
    "vaishali_bibisara": "Women: Vaishali Rameshbabu vs Bibisara Assaubayeva. Hints: Vaishali, Rameshbabu, Bibisara, Assaubayeva.",
    "goryachkina_lagno": "Women: Aleksandra Goryachkina vs Kateryna Lagno. Hints: Goryachkina, Aleksandra, Lagno, Kateryna.",
    "zhu_tan": "Women: Zhu Jiner vs Tan Zhongyi. Hints: Zhu, Jiner, Tan, Zhongyi.",
}

GAME_ALIASES = {
    "pragg_giri": "praggnanandhaa_giri",
    "fabi_hikaru": "caruana_nakamura",
    "wei_blubom": "bluebaum_wei",
    "wei_bluebaum": "bluebaum_wei",
    "bluebaum_yi": "bluebaum_wei",
    "divya_koneru": "divya_humpy",
    "deshmukh_humpy": "divya_humpy",
    "vaishali_assaubayeva": "vaishali_bibisara",
    "goryachkina_kateryna": "goryachkina_lagno",
    "jiner_zhongyi": "zhu_tan",
}

DEFAULT_CURRENT_GAME = "setting_the_stage"
