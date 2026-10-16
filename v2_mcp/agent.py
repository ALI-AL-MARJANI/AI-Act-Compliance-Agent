"""v2 — the v1 graph with tools discovered and executed over MCP (stdio).

The agent spawns v2_mcp/server.py, lists its tools and calls them through the
protocol; it has no knowledge of their implementation. Human interaction
(ask_user, report approval) stays client-side.

Usage:
    python -m v2_mcp.agent "<system description>"
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from v0_scratch.agent import LLM_OPTIONS, MODEL, SYSTEM_PROMPT, TOOL_SCHEMAS, wrap_description
from v1_langgraph.agent import (RECURSION_LIMIT, State, _ask_user, _log_update, answer_in_terminal,
                                approve_in_terminal, nudge_node, print_result, route_after_agent)

ROOT = Path(__file__).resolve().parent.parent

MCP_SERVERS = {
    "compliance": {
        "command": sys.executable,
        "args": [str(ROOT / "v2_mcp" / "server.py")],
        "transport": "stdio",
    }
}
# ask_user is client-side; its schema is reused from v0.
ASK_USER_SCHEMA = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "ask_user")


def build_graph(mcp_tools: list):
    tools_by_name = {tool.name: tool for tool in mcp_tools}
    llm = ChatOllama(model=MODEL, **LLM_OPTIONS).bind_tools(mcp_tools + [ASK_USER_SCHEMA])

    async def agent_node(state: State) -> dict:
        response = await llm.ainvoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    async def call_mcp(name: str, args: dict) -> dict:
        """Call a server tool and decode its JSON text content into a dict."""
        if name not in tools_by_name:
            return {"error": f"Unknown tool '{name}'. Available tools: "
                             f"{', '.join(list(tools_by_name) + ['ask_user'])}"}
        output = await tools_by_name[name].ainvoke(args)
        text = output[0]["text"] if isinstance(output, list) else str(output)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": text}  # FastMCP's own argument validation returns plain text

    async def tools_node(state: State) -> dict:
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
                # Approval precedes server-side validation here: the guardrails live on the
                # server, so a reviewer may approve a draft the server then rejects.
                decision = interrupt({"type": "approval", **args})
                if decision["action"] == "approve":
                    result = await call_mcp(name, args)
                elif decision["action"] == "edit":
                    result = await call_mcp(name, {**args, "content_markdown": decision["content"]})
                else:
                    result = {"error": f"The human reviewer REJECTED the report. "
                                       f"Feedback: {decision.get('feedback', '')}"}
                report_path = result.get("path", report_path)
            else:
                result = await call_mcp(name, args)
            tool_messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                             tool_call_id=call["id"], name=name))
        return {"messages": tool_messages, "questions_asked": questions_asked, "report_path": report_path}

    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("nudge", nudge_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "nudge", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("nudge", "agent")
    return builder.compile(checkpointer=InMemorySaver())


async def run_turn(graph, user_text: str, thread_id: str, first_turn: bool,
                   answer_fn=answer_in_terminal, approve_fn=approve_in_terminal, verbose: bool = True) -> dict:
    """Async counterpart of v1.run_turn."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    content = wrap_description(user_text) if first_turn else user_text
    graph_input = {"messages": [HumanMessage(content)], "questions_asked": 0,
                   "nudges": 0, "report_path": None}
    trace = {"tool_calls": [], "steps": 0, "nudges": 0, "final_answer": None,
             "report": None, "stopped_by": None, "latency_s": None}
    start = time.time()
    try:
        while True:
            interrupted = None
            async for update in graph.astream(graph_input, config, stream_mode="updates"):
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
    state = (await graph.aget_state(config)).values
    trace["final_answer"] = state["messages"][-1].content
    trace["latency_s"] = round(time.time() - start, 1)
    return trace


async def main() -> None:
    client = MultiServerMCPClient(MCP_SERVERS)
    # One session for the whole run; without it each tool call spawns a new server process.
    async with client.session("compliance") as session:
        mcp_tools = await load_mcp_tools(session)
        print(f"MCP tools: {[t.name for t in mcp_tools]}")
        graph = build_graph(mcp_tools)

        thread_id = str(uuid.uuid4())
        user_text = " ".join(sys.argv[1:]) or input("System description:\n> ")
        first_turn = True
        while user_text.strip():
            trace = await run_turn(graph, user_text, thread_id, first_turn)
            print_result(trace)
            first_turn = False
            user_text = input("\nFollow-up on this thread (empty to quit): ")


if __name__ == "__main__":
    asyncio.run(main())
