"""v1 — the v0 agent as a LangGraph StateGraph.

Graph: START -> agent -(tool calls)-> tools -> agent
                      -(no report yet)-> nudge -> agent
                      -(done)-> END

Compared to v0:
- explicit control flow (nodes and a conditional edge instead of a while-loop);
- state persisted after every node by a checkpointer, keyed by thread_id
  (multi-turn follow-ups on the same assessment);
- human-in-the-loop via interrupt(): clarifying questions and report approval
  (approve / edit / reject) pause the graph instead of blocking on input().
Cost: a fast-moving dependency and an extra abstraction layer.

Same model, prompt and tool schemas as v0.

Usage:
    python -m v1_langgraph.agent "<system description>"
    python -m v1_langgraph.agent --mermaid
"""

import json
import sys
import time
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from tools import compliance_tools as ct
from v0_scratch.agent import LLM_OPTIONS, MAX_NUDGES, MAX_STEPS, MODEL, SYSTEM_PROMPT, TOOL_SCHEMAS, wrap_description

llm = ChatOllama(model=MODEL, **LLM_OPTIONS).bind_tools(TOOL_SCHEMAS)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: append, don't overwrite
    questions_asked: int
    nudges: int
    report_path: str | None


# --- Nodes ----------------------------------------------------------------------
def agent_node(state: State) -> dict:
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: State) -> dict:
    """Execute the pending tool calls.

    ask_user and save_report call interrupt(). On resume LangGraph re-executes
    this node from the start and interrupt() returns the human's input, so the
    code before an interrupt must be idempotent.
    """
    questions_asked = state["questions_asked"]
    report_path = state["report_path"]
    tool_messages = []

    for call in state["messages"][-1].tool_calls:
        name, args = call["name"], call["args"]

        if name == "ask_user":
            result = _ask_user(args, questions_asked)
            if "answer" in result:
                questions_asked += 1
        elif name == "save_report":
            result = _save_report_with_approval(args)
            report_path = result.get("path", report_path)
        elif name in ct.TOOLS:
            function, _ = ct.TOOLS[name]
            try:
                result = function(**args)
            except TypeError as e:
                result = {"error": f"Invalid arguments for {name}: {e}"}
        else:
            result = {"error": f"Unknown tool '{name}'. Available tools: {', '.join(ct.TOOLS)}"}

        tool_messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                         tool_call_id=call["id"], name=name))

    return {"messages": tool_messages, "questions_asked": questions_asked, "report_path": report_path}


def _ask_user(args: dict, questions_asked: int) -> dict:
    try:
        question = ct.AskUserArgs(**args).question
    except ValidationError as e:
        return {"error": f"Invalid arguments for ask_user: {e.errors()[0]['msg']}"}
    if questions_asked >= ct.MAX_QUESTIONS:
        return {"error": f"Question limit reached ({ct.MAX_QUESTIONS}). Continue with the "
                         "information you have and state your assumptions in the report."}
    answer = interrupt({"type": "question", "question": question})
    return {"question": question, "answer": answer}


def _save_report_with_approval(args: dict) -> dict:
    try:
        error = ct.validate_report(**args)  # automated checks before human review
    except TypeError as e:
        return {"error": f"Invalid arguments for save_report: {e}"}
    if error:
        return error

    decision = interrupt({"type": "approval", **args})
    if decision["action"] == "approve":
        return ct.save_report(**args)
    if decision["action"] == "edit":
        return ct.save_report(args["title"], args["risk_category"], decision["content"])  # re-validated
    return {"error": f"The human reviewer REJECTED the report. Feedback: {decision.get('feedback', '')}"}


def nudge_node(state: State) -> dict:
    return {"nudges": state["nudges"] + 1,
            "messages": [HumanMessage("You have not saved the report yet. "
                                      "Finish the procedure and call save_report.")]}


def route_after_agent(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    if state["report_path"] is None and state["nudges"] < MAX_NUDGES:
        return "nudge"
    return END


def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("nudge", nudge_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "nudge", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("nudge", "agent")
    # In-memory for the prototype; a Postgres/SQLite saver is needed to survive restarts.
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()

# recursion_limit counts node executions: 2 per agent step (agent + tools), plus nudges.
RECURSION_LIMIT = 2 * MAX_STEPS + 2 * MAX_NUDGES


# --- Terminal handlers for the two interrupt types ------------------------------
def answer_in_terminal(question: str) -> str:
    return input(f"\n[question] {question}\n> ")


def approve_in_terminal(draft: dict) -> dict:
    print("\n" + "=" * 70)
    print(f"DRAFT REPORT — {draft['title']} [{draft['risk_category']}]")
    print("=" * 70)
    print(draft["content_markdown"])
    print("=" * 70)
    choice = ""
    while choice not in ("a", "e", "r"):
        choice = input("Approve, edit or reject? (a/e/r): ").strip().lower()
    if choice == "a":
        return {"action": "approve"}
    if choice == "e":
        draft_file = ct.REPORTS_DIR / "_draft_to_edit.md"
        ct.REPORTS_DIR.mkdir(exist_ok=True)
        draft_file.write_text(draft["content_markdown"])
        input(f"Edit {draft_file.relative_to(ct.ROOT)}, save, then press Enter.")
        content = draft_file.read_text()
        draft_file.unlink()
        return {"action": "edit", "content": content}
    return {"action": "reject", "feedback": input("Feedback for the agent: ")}


# --- Runner -----------------------------------------------------------------------
def run_turn(user_text: str, thread_id: str, first_turn: bool,
             answer_fn=answer_in_terminal, approve_fn=approve_in_terminal, verbose: bool = True) -> dict:
    """Run one user turn on a thread, resolving interrupts until END. Returns a v0-compatible trace."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    content = wrap_description(user_text) if first_turn else user_text
    # Per-turn counters are reset; messages are appended to the thread history.
    graph_input = {"messages": [HumanMessage(content)], "questions_asked": 0,
                   "nudges": 0, "report_path": None}

    trace = {"tool_calls": [], "steps": 0, "nudges": 0, "final_answer": None,
             "report": None, "stopped_by": None, "latency_s": None}
    start = time.time()
    try:
        while True:
            interrupted = None
            for update in graph.stream(graph_input, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    interrupted = update["__interrupt__"][0].value
                    continue
                _log_update(update, trace, verbose)
            if interrupted is None:
                break
            if interrupted["type"] == "question":
                resume_value = answer_fn(interrupted["question"])
            else:
                resume_value = approve_fn(interrupted)
            graph_input = Command(resume=resume_value)
        trace["stopped_by"] = "final_answer"
    except GraphRecursionError:
        trace["stopped_by"] = "max_steps"
        if verbose:
            print(f"\nstopped: recursion limit ({RECURSION_LIMIT} node runs) reached")

    state = graph.get_state(config).values
    trace["final_answer"] = state["messages"][-1].content
    trace["latency_s"] = round(time.time() - start, 1)
    return trace


def _log_update(update: dict, trace: dict, verbose: bool) -> None:
    for node, values in update.items():
        if node == "agent":
            trace["steps"] += 1
            message = values["messages"][-1]
            if verbose:
                print(f"\n[step {trace['steps']}]")
                if message.content:
                    print(f"thought      {message.content.strip()}")
                for call in message.tool_calls:
                    print(f"action       {call['name']}({_short(json.dumps(call['args'], ensure_ascii=False), 200)})")
            trace["tool_calls"] += [{"name": c["name"], "arguments": c["args"], "result": None}
                                    for c in message.tool_calls]
        elif node == "tools":
            results = [json.loads(m.content) for m in values["messages"]]
            for call, result in zip(trace["tool_calls"][-len(results):], results):
                call["result"] = result
                if call["name"] == "save_report" and result.get("status") == "saved":
                    # Read back from disk: the human may have edited the draft.
                    trace["report"] = {**call["arguments"], "path": result["path"],
                                       "content_markdown": (ct.ROOT / result["path"]).read_text()}
                if verbose:
                    label = "error      " if "error" in result else "observation"
                    print(f"{label}  {_short(json.dumps(result, ensure_ascii=False), 300)}")
        elif node == "nudge":
            trace["nudges"] += 1
            if verbose:
                print("nudge        report not saved yet, reminder sent")


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def print_result(trace: dict) -> None:
    print(f"\nresult       {trace['final_answer'] or '(no final answer)'}")
    print(f"             steps={trace['steps']} stopped_by={trace['stopped_by']} latency={trace['latency_s']}s")
    if trace["report"]:
        print(f"             report: {trace['report']['path']}")


if __name__ == "__main__":
    if "--mermaid" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()

    thread_id = str(uuid.uuid4())
    user_text = " ".join(sys.argv[1:]) or input("System description:\n> ")
    first_turn = True
    while user_text.strip():
        print_result(run_turn(user_text, thread_id, first_turn))
        first_turn = False
        user_text = input("\nFollow-up on this thread (empty to quit): ")
