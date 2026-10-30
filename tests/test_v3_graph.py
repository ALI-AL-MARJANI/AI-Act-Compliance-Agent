"""Tests for the v3 graph with a scripted chat model (no LLM required)."""

import json
import uuid

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from tools import compliance_tools as ct  # noqa: E402
from v3_multiagent import agent as v3  # noqa: E402

REPORT = f"""{ct.DISCLAIMER}

# First-pass assessment: CV ranking tool

## Risk category: high_risk
Recruitment is listed in Annex III [AI Act Art. 6]. Automated rejection is a concern [GDPR Art. 22].
"""


class ScriptedModel:
    """Fake chat model: returns the scripted messages in order and records what it receives."""

    def __init__(self, script: list[AIMessage]):
        self.script = list(script)
        self.seen = []  # one entry per invoke: (bound tool names, messages)

    def bind_tools(self, tools: list[dict]):
        return BoundModel(self, [tool["function"]["name"] for tool in tools])


class BoundModel:
    def __init__(self, parent: ScriptedModel, tool_names: list[str]):
        self.parent, self.tool_names = parent, tool_names

    def invoke(self, messages: list) -> AIMessage:
        self.parent.seen.append((self.tool_names, messages))
        return self.parent.script.pop(0)


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(uuid.uuid4())}])


def calls(*pairs: tuple[str, dict]) -> AIMessage:
    """One assistant message with several tool calls."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(uuid.uuid4())}
                                             for name, args in pairs])


def route(next_agent: str, reason: str = "next") -> AIMessage:
    return call("route", next=next_agent, reason=reason)


def text(content: str = "I think I am done.") -> AIMessage:
    return AIMessage(content=content)


def silent_agent() -> list[AIMessage]:
    """An agent that never calls a tool. Distinct objects: add_messages deduplicates by id."""
    return [text() for _ in range(1 + v3.MAX_NUDGES)]


CLASSIFY = ("submit_classification", {"risk_category": "high_risk", "cited_articles": ["6"],
                                      "reasoning": "Recruitment is an Annex III use case."})
FINDINGS = ("submit_gdpr_findings", {"personal_data": True,
                                     "concerns": [{"article": "22", "issue": "Automated rejection."}]})
SAVE = ("save_report", {"title": "CV ranking tool", "risk_category": "high_risk", "content_markdown": REPORT})


def run(script: list[AIMessage], routing: str = "fixed", approvals: list[dict] | None = None,
        answers: list[str] | None = None) -> tuple[dict, ScriptedModel]:
    model = ScriptedModel(script)
    approvals = list(approvals or [])
    answers = list(answers or [])
    trace = v3.run_turn("A tool that ranks CVs and rejects the bottom half.", routing=routing, model=model,
                        answer_fn=lambda question: answers.pop(0),
                        approve_fn=lambda draft: approvals.pop(0) if approvals else {"action": "approve"},
                        verbose=False)
    assert model.script == [], "the run stopped before the end of the script"
    return trace, model


def errors_of(trace: dict, tool: str) -> list[str]:
    return [c["result"]["error"] for c in trace["tool_calls"] if c["name"] == tool and "error" in c["result"]]


def saved_reports() -> list:
    return sorted(ct.REPORTS_DIR.glob("*.md")) if ct.REPORTS_DIR.exists() else []


@pytest.fixture(autouse=True)
def temporary_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "ROOT", tmp_path)
    monkeypatch.setattr(ct, "REPORTS_DIR", tmp_path / "reports")


# --- Full runs ---------------------------------------------------------------
def test_fixed_pipeline_produces_a_report():
    trace, _ = run([
        call("check_prohibited_practices", description="A tool that ranks CVs."),
        calls(CLASSIFY),
        call("get_article", regulation="gdpr", article="22"),
        calls(FINDINGS),
        call("get_obligations", risk_category="high_risk"),
        calls(SAVE),
    ])
    assert trace["report"]["risk_category"] == "high_risk"
    assert saved_reports() and "[GDPR Art. 22]" in trace["report"]["content_markdown"]
    assert trace["stopped_by"] == "final_answer"
    assert trace["final_answer"] == "The risk category is high_risk."
    # No supervisor call in fixed mode.
    assert trace["llm_calls"] == trace["steps"] == 6
    assert trace["handoffs"] == 3
    assert [c["agent"] for c in trace["tool_calls"]] == ["ai_act"] * 2 + ["gdpr"] * 2 + ["writer"] * 2


def test_trace_has_the_keys_read_by_the_eval():
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(SAVE)])
    assert {"tool_calls", "steps", "nudges", "final_answer", "report", "stopped_by", "latency_s",
            "llm_calls", "handoffs"} <= set(trace)
    assert {"name", "arguments", "result", "agent"} <= set(trace["tool_calls"][0])


def test_llm_routing_counts_supervisor_calls():
    trace, _ = run([route("ai_act"), calls(CLASSIFY), route("gdpr"), calls(FINDINGS),
                    route("writer"), calls(SAVE)], routing="llm")
    assert trace["report"] is not None
    assert trace["llm_calls"] == 6 and trace["steps"] == 3 and trace["handoffs"] == 3
    assert trace["refusals"] == []


def test_llm_routing_can_skip_gdpr():
    trace, _ = run([route("ai_act"), calls(CLASSIFY), route("writer"), calls(SAVE)], routing="llm")
    assert trace["report"] is not None and trace["handoffs"] == 2


# --- Context isolation -------------------------------------------------------
def test_each_agent_only_sees_its_own_history_and_tools():
    _, model = run([call("get_article", regulation="ai_act", article="6"), calls(CLASSIFY),
                    calls(FINDINGS), calls(SAVE)])
    gdpr_tools, gdpr_messages = model.seen[2]
    assert "check_prohibited_practices" not in gdpr_tools and "save_report" not in gdpr_tools
    assert "submit_classification" not in str([m.content for m in gdpr_messages])
    assert not [m for m in gdpr_messages if m.type in ("ai", "tool")]

    writer_tools, writer_messages = model.seen[3]
    assert set(writer_tools) == {"get_obligations", "get_article", "save_report"}
    assert "Recruitment is an Annex III use case." in writer_messages[1].content  # structured findings only
    assert "[GDPR Art. 22]" in writer_messages[1].content


def test_supervisor_sees_a_summary_not_the_histories():
    _, model = run([route("ai_act"), call("get_article", regulation="ai_act", article="6"), calls(CLASSIFY),
                    route("writer"), calls(SAVE)], routing="llm")
    _, messages = model.seen[3]
    assert "ai_act: done (risk_category=high_risk)" in messages[1].content
    assert not [m for m in messages if m.type in ("ai", "tool")]


# --- Supervisor guards -------------------------------------------------------
def test_writer_before_classification_is_redirected_to_ai_act():
    trace, _ = run([route("writer"), calls(CLASSIFY), route("gdpr"), calls(FINDINGS),
                    route("writer"), calls(SAVE)], routing="llm")
    assert trace["refusals"] == ["writer refused: no classification yet, using ai_act"]
    assert trace["tool_calls"][0]["agent"] == "ai_act"
    assert trace["report"] is not None


def test_finish_before_report_is_refused():
    trace, _ = run([route("FINISH"), calls(CLASSIFY), route("gdpr"), calls(FINDINGS),
                    route("writer"), calls(SAVE)], routing="llm")
    assert trace["refusals"] == ["FINISH refused: report not saved, using ai_act"]
    assert trace["report"] is not None


def test_agent_that_already_delivered_is_not_called_again():
    trace, _ = run([route("ai_act"), calls(CLASSIFY), route("ai_act"), calls(FINDINGS),
                    route("writer"), calls(SAVE)], routing="llm")
    assert trace["refusals"] == ["ai_act refused: it has already delivered, using gdpr"]


def test_invalid_route_falls_back_to_the_pipeline():
    trace, _ = run([text("The AI Act agent should go first."), calls(CLASSIFY),
                    call("route", next="legal_team"), calls(FINDINGS),
                    route("writer"), calls(SAVE)], routing="llm")
    assert trace["refusals"] == ["no valid route, using ai_act", "no valid route, using gdpr"]
    assert trace["report"] is not None


def test_supervisor_makes_no_call_once_the_report_is_saved():
    trace, _ = run([route("ai_act"), calls(CLASSIFY), route("writer"), calls(SAVE)], routing="llm")
    assert trace["llm_calls"] == 4


def test_handoff_budget_stops_the_run():
    script = [message for _ in range(v3.MAX_HANDOFFS) for message in [route("ai_act"), *silent_agent()]]
    trace, _ = run(script, routing="llm")
    assert trace["handoffs"] == v3.MAX_HANDOFFS
    assert trace["report"] is None and trace["stopped_by"] == "max_steps"


# --- Sub-agent budgets -------------------------------------------------------
def test_ai_act_never_submits_ends_without_report():
    trace, _ = run(silent_agent())
    assert trace["report"] is None
    assert trace["stopped_by"] == "no_report"
    assert trace["nudges"] == v3.MAX_NUDGES
    assert trace["handoffs"] == 1


def test_nudge_makes_the_agent_submit():
    trace, _ = run([text(), calls(CLASSIFY), calls(FINDINGS), calls(SAVE)])
    assert trace["nudges"] == 1 and trace["report"] is not None


def test_step_budget_per_agent():
    looping = [call("get_article", regulation="ai_act", article="6") for _ in range(v3.MAX_WORKER_STEPS)]
    trace, _ = run(looping)
    assert trace["steps"] == v3.MAX_WORKER_STEPS
    assert trace["stopped_by"] == "no_report"


def test_gdpr_failure_does_not_block_the_report():
    trace, model = run([calls(CLASSIFY), *silent_agent(), calls(SAVE)])
    assert trace["report"] is not None
    assert "not available" in model.seen[-1][1][1].content


# --- Writer guards -----------------------------------------------------------
def test_save_report_with_another_category_is_rejected():
    wrong = ("save_report", {**SAVE[1], "risk_category": "minimal_risk"})
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(wrong), calls(SAVE)])
    assert "must be 'high_risk'" in errors_of(trace, "save_report")[0]
    assert trace["report"]["risk_category"] == "high_risk"
    assert len(saved_reports()) == 1


def test_validate_report_runs_before_human_approval():
    invalid = ("save_report", {**SAVE[1], "content_markdown": REPORT.replace(ct.DISCLAIMER, "x" * 400)})
    reviewed = []
    model = ScriptedModel([calls(CLASSIFY), calls(FINDINGS), calls(invalid), calls(SAVE)])
    trace = v3.run_turn("A tool that ranks CVs.", routing="fixed", model=model,
                        approve_fn=lambda draft: reviewed.append(draft) or {"action": "approve"}, verbose=False)
    assert "disclaimer" in errors_of(trace, "save_report")[0]
    assert len(reviewed) == 1


def test_rejected_report_goes_back_to_the_writer():
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(SAVE), calls(SAVE)],
                   approvals=[{"action": "reject", "feedback": "Cite Art. 35."}, {"action": "approve"}])
    assert "Cite Art. 35." in errors_of(trace, "save_report")[0]
    assert trace["report"] is not None and len(saved_reports()) == 1


def test_edited_report_is_read_back_from_disk():
    edited = REPORT + "\nEdited by the reviewer.\n"
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(SAVE)],
                   approvals=[{"action": "edit", "content": edited}])
    assert trace["report"]["content_markdown"] == edited


def test_two_saves_in_one_message_write_one_file():
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(SAVE, SAVE)])
    assert len(saved_reports()) == 1
    assert "already saved" in errors_of(trace, "save_report")[0]


# --- Exit tools --------------------------------------------------------------
def test_gdpr_rejects_article_outside_the_knowledge_base():
    invented = ("submit_gdpr_findings", {"personal_data": True,
                                         "concerns": [{"article": "99", "issue": "Invented article."}]})
    trace, _ = run([calls(CLASSIFY), calls(invented), calls(FINDINGS), calls(SAVE)])
    assert "not in the knowledge base" in errors_of(trace, "submit_gdpr_findings")[0]
    assert trace["report"] is not None


def test_classification_rejects_article_outside_the_knowledge_base():
    invented = ("submit_classification", {**CLASSIFY[1], "cited_articles": ["123"]})
    trace, _ = run([calls(invented), calls(CLASSIFY), calls(FINDINGS), calls(SAVE)])
    assert "not in the knowledge base" in errors_of(trace, "submit_classification")[0]


def test_gdpr_agent_cannot_read_ai_act_articles():
    trace, _ = run([calls(CLASSIFY), call("get_article", regulation="ai_act", article="6"),
                    calls(FINDINGS), calls(SAVE)])
    assert "only read GDPR" in errors_of(trace, "get_article")[0]


def test_tools_of_another_agent_are_refused():
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(SAVE, ("ask_user", {"question": "Is it public?"}))])
    assert "Unknown tool" in errors_of(trace, "ask_user")[0]


def test_nested_schema_has_no_reference():
    schema = next(s for s in map(lambda t: v3.tool_schema(t, *v3.GDPR_TOOLS[t]), v3.GDPR_TOOLS)
                  if s["function"]["name"] == "submit_gdpr_findings")
    assert "$ref" not in json.dumps(schema) and "$defs" not in json.dumps(schema)


# --- Interrupts --------------------------------------------------------------
def test_resume_after_question_then_approval():
    questions, drafts = [], []
    model = ScriptedModel([
        call("ask_user", question="Does a human review each rejection?"),
        calls(CLASSIFY), calls(FINDINGS), calls(SAVE),
    ])
    trace = v3.run_turn("A tool that ranks CVs.", routing="fixed", model=model,
                        answer_fn=lambda question: questions.append(question) or "No human review.",
                        approve_fn=lambda draft: drafts.append(draft) or {"action": "approve"}, verbose=False)
    assert questions == ["Does a human review each rejection?"]
    assert len(drafts) == 1
    assert trace["tool_calls"][0]["result"]["answer"] == "No human review."
    assert len(saved_reports()) == 1
    # Resuming re-executes the tools node, never the model.
    assert model.script == [] and trace["llm_calls"] == 4


def test_answers_reach_the_other_agents():
    _, model = run([call("ask_user", question="Does a human review each rejection?"),
                    calls(CLASSIFY), calls(FINDINGS), calls(SAVE)], answers=["No human review."])
    assert "No human review." in model.seen[2][1][1].content  # gdpr briefing
    assert "No human review." in model.seen[3][1][1].content  # writer briefing


def test_question_budget_is_shared_between_agents():
    question = ("ask_user", {"question": "Is it used on employees?"})
    trace, _ = run([calls(question, question), calls(CLASSIFY),
                    calls(question, question), calls(FINDINGS), calls(SAVE)], answers=["yes"] * 3)
    assert sum("answer" in c["result"] for c in trace["tool_calls"] if c["name"] == "ask_user") == ct.MAX_QUESTIONS
    assert "Question limit reached" in errors_of(trace, "ask_user")[0]


def test_two_questions_in_one_message_are_resumed_in_order():
    question_1, question_2 = ("ask_user", {"question": "First question?"}), ("ask_user", {"question": "Second one?"})
    trace, _ = run([calls(question_1, question_2), calls(CLASSIFY), calls(FINDINGS), calls(SAVE)],
                   answers=["one", "two"])
    assert [c["result"]["answer"] for c in trace["tool_calls"][:2]] == ["one", "two"]


# --- Graph -------------------------------------------------------------------
def test_mermaid_shows_one_pair_of_nodes_per_agent():
    mermaid = v3.build_graph(ScriptedModel([])).get_graph().draw_mermaid()
    for node in ("supervisor", "ai_act_agent", "ai_act_tools", "gdpr_agent", "gdpr_tools",
                 "writer_agent", "writer_tools"):
        assert node in mermaid


def test_human_nudge_is_stored_in_the_agent_history():
    _, model = run([text(), calls(CLASSIFY), calls(FINDINGS), calls(SAVE)])
    assert isinstance(model.seen[1][1][-1], HumanMessage)
    assert "submit_classification" in model.seen[1][1][-1].content


# --- Eval --------------------------------------------------------------------
def test_eval_scores_a_v3_trace_and_prints_dash_for_v0():
    run_eval = pytest.importorskip("eval.run_eval")
    trace, _ = run([calls(CLASSIFY), calls(FINDINGS), calls(SAVE)])
    case = {"id": "T1", "difficulty": "easy", "incomplete": False, "risk_category": "high_risk",
            "gold_articles": ["AI Act Art. 6"], "gdpr_flags": ["automated_decision"], "hidden_facts": []}
    row = run_eval.score_trace(case, trace)
    v1_trace = {k: trace[k] for k in trace if k not in ("llm_calls", "handoffs")}
    summaries = {"v1": run_eval.summarize([run_eval.score_trace(case, v1_trace)]),
                 "v3_fixed": run_eval.summarize([row])}
    assert row["pred"] == "high_risk" and row["llm_calls"] == 3
    assert summaries["v3_fixed"]["avg_handoffs"] == 3
    assert "| avg_llm_calls | - | 3.00 |" in run_eval.to_markdown(summaries)
