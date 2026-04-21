# AIPRO
# Support Ticket Triage Pipeline

This project is a LangGraph-based support ticket processing system.

## What it does
The system reads support tickets from a CSV file and processes them in multiple steps:
1. Load tickets
2. Categorize them by department and urgency
3. Summarize the issue and detect sentiment
4. Generate a draft reply
5. Export results to JSON

## Technologies
- LangGraph for pipeline orchestration
- OpenAI Responses API for LLM calls
- Pydantic for structured outputs
- LangSmith for tracing
- CSV/JSON for input and output files

## Files
- `main.py` — main pipeline logic
- `prompts.py` — prompts for all LLM calls
- `tickets.csv` — input sample
- `output.json` — output results

## Output
The final JSON includes:
- ticket category
- urgency
- issue summary
- root cause
- suggested action
- sentiment
- draft reply

## Notes
The project includes retry and delay logic to handle API rate limits more reliably.
