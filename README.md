# Chess Broadcast Auto-Annotator

Append-only transcript processing for chess broadcast commentary.

The output is one markdown annotation file per Round 1 game/section in `games/`. These files are meant to include played moves, candidate moves, rejected ideas, plans, psychology, and commentator reasoning. Pre-game tournament context goes into `games/setting_the_stage.md`.

## Local Ollama Setup

OpenAI is not required. The default provider is Ollama.

Install or pull a local model:

```powershell
ollama pull qwen3.5:2b
```

Make sure Ollama is running:

```powershell
ollama serve
```

Process transcripts:

```powershell
python main.py
```

To use a different local model for one run:

```powershell
$env:OLLAMA_MODEL="llama3.1:8b"
python main.py
```

Useful local models for this task are usually instruction-tuned models around 7B-8B or larger. A tiny model can work for plumbing tests, but may be weak at board-switch detection.

## Optional OpenAI Setup

OpenAI is still supported if you want it later:

```powershell
$env:MODEL_PROVIDER="openai"
$env:OPENAI_API_KEY="your_key_here"
python main.py
```

## Workflow

Put `.txt` transcript files in `transcripts/raw`. The processor chunks each file, asks the model for plain-text `APPEND` blocks, writes only to the current game annotation files, updates `runtime/current_game.txt`, then moves processed transcripts into `transcripts/processed`.
