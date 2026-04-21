import os
import csv
import json
import time
from typing import List, Literal

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, RateLimitError, APIError
from langgraph.graph import StateGraph, END
from langsmith import traceable

from prompts import (
    CATEGORY_SYSTEM_PROMPT,
    CATEGORY_USER_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
    REPLY_SYSTEM_PROMPT,
    REPLY_USER_PROMPT,
)

# =========================================================
# CONFIG
# =========================================================

MODEL = "gpt-5-nano"
INPUT_CSV = "tickets.csv"
OUTPUT_JSON = "output.json"

# при 3 RPM безопасно держать ~20-21 сек
REQUEST_DELAY_SECONDS = 21
MAX_RETRIES = 6


# =========================================================
# DATA MODELS
# =========================================================

class TicketInput(BaseModel):
    ticket_id: str
    subject: str
    body: str


class TicketCategory(BaseModel):
    department: Literal["Billing", "Technical", "Account", "Other"]
    urgency: Literal["Critical", "High", "Normal", "Low"]


class TicketSummary(BaseModel):
    issue_summary: str
    root_cause: str
    suggested_action: str
    sentiment: Literal["Angry", "Neutral", "Satisfied"]


class DraftReply(BaseModel):
    reply_subject: str
    reply_body: str


class CategorizedTicket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    department: Literal["Billing", "Technical", "Account", "Other"]
    urgency: Literal["Critical", "High", "Normal", "Low"]


class SummarizedTicket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    department: Literal["Billing", "Technical", "Account", "Other"]
    urgency: Literal["Critical", "High", "Normal", "Low"]
    issue_summary: str
    root_cause: str
    suggested_action: str
    sentiment: Literal["Angry", "Neutral", "Satisfied"]


class ProcessedTicket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    department: Literal["Billing", "Technical", "Account", "Other"]
    urgency: Literal["Critical", "High", "Normal", "Low"]
    issue_summary: str
    root_cause: str
    suggested_action: str
    sentiment: Literal["Angry", "Neutral", "Satisfied"]
    reply_subject: str
    reply_body: str


class State(BaseModel):
    input_csv: str = INPUT_CSV
    output_json: str = OUTPUT_JSON

    raw: List[TicketInput] = Field(default_factory=list)
    categorized: List[CategorizedTicket] = Field(default_factory=list)
    summarized: List[SummarizedTicket] = Field(default_factory=list)
    final: List[ProcessedTicket] = Field(default_factory=list)


# =========================================================
# CLIENT
# =========================================================

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


# =========================================================
# GENERIC RETRY LLM CALL
# =========================================================

def call_llm_with_retry(schema, system_prompt: str, user_prompt: str):
    client = get_client()
    delay = REQUEST_DELAY_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=schema,
            )
            return response.output_parsed

        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Rate limit after {MAX_RETRIES} attempts: {e}") from e

            print(f"[429] attempt {attempt}/{MAX_RETRIES} | sleep {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 90)

        except APIError as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"API error after {MAX_RETRIES} attempts: {e}") from e

            print(f"[API ERROR] attempt {attempt}/{MAX_RETRIES} | sleep 10s")
            time.sleep(10)

        except ValidationError as e:
            raise RuntimeError(f"Invalid structured output: {e}") from e


# =========================================================
# LLM CALLS WITH LANGSMITH TRACING
# =========================================================

@traceable(name="categorize_one")
def categorize_one(ticket: TicketInput) -> TicketCategory:
    return call_llm_with_retry(
        TicketCategory,
        CATEGORY_SYSTEM_PROMPT,
        CATEGORY_USER_PROMPT.format(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            body=ticket.body,
        ),
    )


@traceable(name="summarize_one")
def summarize_one(ticket: CategorizedTicket) -> TicketSummary:
    return call_llm_with_retry(
        TicketSummary,
        SUMMARY_SYSTEM_PROMPT,
        SUMMARY_USER_PROMPT.format(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            body=ticket.body,
            department=ticket.department,
            urgency=ticket.urgency,
        ),
    )


@traceable(name="reply_one")
def reply_one(ticket: SummarizedTicket) -> DraftReply:
    return call_llm_with_retry(
        DraftReply,
        REPLY_SYSTEM_PROMPT,
        REPLY_USER_PROMPT.format(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            body=ticket.body,
            department=ticket.department,
            urgency=ticket.urgency,
            issue_summary=ticket.issue_summary,
            suggested_action=ticket.suggested_action,
            sentiment=ticket.sentiment,
        ),
    )


# =========================================================
# FILE IO
# =========================================================

def load_tickets_from_csv(path: str) -> List[TicketInput]:
    tickets: List[TicketInput] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"id", "subject", "body"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            tickets.append(
                TicketInput(
                    ticket_id=str(row["id"]).strip(),
                    subject=str(row["subject"]).strip(),
                    body=str(row["body"]).strip(),
                )
            )

    return tickets


def save_results_to_json(path: str, results: List[ProcessedTicket]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [item.model_dump() for item in results],
            f,
            ensure_ascii=False,
            indent=4,
        )


# =========================================================
# FALLBACKS
# =========================================================

def fallback_category(ticket: TicketInput) -> CategorizedTicket:
    return CategorizedTicket(
        ticket_id=ticket.ticket_id,
        subject=ticket.subject,
        body=ticket.body,
        department="Other",
        urgency="Normal",
    )


def fallback_summary(ticket: CategorizedTicket, error_message: str) -> SummarizedTicket:
    return SummarizedTicket(
        ticket_id=ticket.ticket_id,
        subject=ticket.subject,
        body=ticket.body,
        department=ticket.department,
        urgency=ticket.urgency,
        issue_summary="Automatic summary failed.",
        root_cause=error_message[:300],
        suggested_action="Manual review required.",
        sentiment="Neutral",
    )


def fallback_reply(ticket: SummarizedTicket) -> DraftReply:
    return DraftReply(
        reply_subject="Support update",
        reply_body="We received your request and our support team is currently reviewing it.",
    )


# =========================================================
# LANGGRAPH NODES
# =========================================================

def load_node(state: State) -> State:
    print(f"Loading tickets from {state.input_csv} ...")
    state.raw = load_tickets_from_csv(state.input_csv)
    print(f"Loaded {len(state.raw)} ticket(s)")
    return state


def categorize_node(state: State) -> State:
    results: List[CategorizedTicket] = []

    for index, ticket in enumerate(state.raw, start=1):
        print(f"[categorize {index}/{len(state.raw)}] ticket {ticket.ticket_id}")
        try:
            result = categorize_one(ticket)
            results.append(
                CategorizedTicket(
                    ticket_id=ticket.ticket_id,
                    subject=ticket.subject,
                    body=ticket.body,
                    department=result.department,
                    urgency=result.urgency,
                )
            )
        except Exception as e:
            print(f"Categorize failed for {ticket.ticket_id}: {e}")
            results.append(fallback_category(ticket))

        if index < len(state.raw):
            time.sleep(REQUEST_DELAY_SECONDS)

    state.categorized = results
    return state


def summarize_node(state: State) -> State:
    results: List[SummarizedTicket] = []

    for index, ticket in enumerate(state.categorized, start=1):
        print(f"[summarize {index}/{len(state.categorized)}] ticket {ticket.ticket_id}")
        try:
            result = summarize_one(ticket)
            results.append(
                SummarizedTicket(
                    ticket_id=ticket.ticket_id,
                    subject=ticket.subject,
                    body=ticket.body,
                    department=ticket.department,
                    urgency=ticket.urgency,
                    issue_summary=result.issue_summary,
                    root_cause=result.root_cause,
                    suggested_action=result.suggested_action,
                    sentiment=result.sentiment,
                )
            )
        except Exception as e:
            print(f"Summarize failed for {ticket.ticket_id}: {e}")
            results.append(fallback_summary(ticket, str(e)))

        if index < len(state.categorized):
            time.sleep(REQUEST_DELAY_SECONDS)

    state.summarized = results
    return state


def reply_node(state: State) -> State:
    results: List[ProcessedTicket] = []

    for index, ticket in enumerate(state.summarized, start=1):
        print(f"[reply {index}/{len(state.summarized)}] ticket {ticket.ticket_id}")

        try:
            # можно всем делать reply, чтобы показать полноценную систему
            reply = reply_one(ticket)
        except Exception as e:
            print(f"Reply failed for {ticket.ticket_id}: {e}")
            reply = fallback_reply(ticket)

        results.append(
            ProcessedTicket(
                ticket_id=ticket.ticket_id,
                subject=ticket.subject,
                body=ticket.body,
                department=ticket.department,
                urgency=ticket.urgency,
                issue_summary=ticket.issue_summary,
                root_cause=ticket.root_cause,
                suggested_action=ticket.suggested_action,
                sentiment=ticket.sentiment,
                reply_subject=reply.reply_subject,
                reply_body=reply.reply_body,
            )
        )

        if index < len(state.summarized):
            time.sleep(REQUEST_DELAY_SECONDS)

    state.final = results
    return state


def export_node(state: State) -> State:
    save_results_to_json(state.output_json, state.final)
    print(f"Saved results to {state.output_json}")
    return state


# =========================================================
# BUILD GRAPH
# =========================================================

def build_graph():
    graph = StateGraph(State)

    graph.add_node("load", load_node)
    graph.add_node("categorize", categorize_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("reply", reply_node)
    graph.add_node("export", export_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "categorize")
    graph.add_edge("categorize", "summarize")
    graph.add_edge("summarize", "reply")
    graph.add_edge("reply", "export")
    graph.add_edge("export", END)

    return graph.compile()


# =========================================================
# MAIN
# =========================================================

def main():
    app = build_graph()
    state = State(
        input_csv=INPUT_CSV,
        output_json=OUTPUT_JSON,
    )
    app.invoke(state)
    print("DONE")


if __name__ == "__main__":
    main()