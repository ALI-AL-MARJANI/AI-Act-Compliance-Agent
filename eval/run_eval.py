"""Benchmark v0 and v1 on eval/cases.json.

Non-interactive: ask_user is answered with the case's hidden_facts (joined), or
"No additional information" for complete cases; report approval is auto-accepted.

Metrics:
- classification: accuracy, macro-F1, confusion matrix; prediction = risk_category
  of the saved report ("no_report" if none);
- citations: micro precision / recall on classification articles (AI Act Art. 5, 6, 50);
- GDPR flags: micro precision / recall, derived from GDPR citations
  (Art. 5/6 -> personal_data, 9 -> special_categories, 22 -> automated_decision, 35 -> dpia);
- clarification: ask_user before the first save_report on incomplete cases,
  no question on complete cases;
- hallucinated citations in saved reports, and invented articles caught by the guardrails;
- steps, tool error rate, self-corrections, nudges, latency.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --versions v0 --limit 3
"""

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from tools import compliance_tools as ct
from v0_scratch import agent as v0
from v1_langgraph import agent as v1

EVAL_DIR = Path(__file__).resolve().parent
LABELS = ct.RISK_CATEGORIES
CLASSIFICATION_ARTICLES = {"AI Act Art. 5", "AI Act Art. 6", "AI Act Art. 50"}
GDPR_FLAG_OF_ARTICLE = {"GDPR Art. 5": "personal_data", "GDPR Art. 6": "personal_data",
                        "GDPR Art. 9": "special_categories", "GDPR Art. 22": "automated_decision",
                        "GDPR Art. 35": "dpia"}


# ---------------------------------------------------------------------------
# Run one case
# ---------------------------------------------------------------------------
def answer_from_case(case: dict):
    def answer(question: str) -> str:
        return " ".join(case["hidden_facts"]) or "No additional information: rely on the description."
    return answer


def run_case(case: dict, version: str) -> dict:
    answer = answer_from_case(case)
    if version == "v0":
        ct.set_answer_provider(answer)
        trace = v0.run_agent(case["description"], approve_fn=lambda args: (True, ""), verbose=False)
        ct.set_answer_provider(None)
    else:
        trace = v1.run_turn(case["description"], thread_id=str(uuid.uuid4()), first_turn=True,
                            answer_fn=answer, approve_fn=lambda draft: {"action": "approve"}, verbose=False)
    return score_trace(case, trace)


def score_trace(case: dict, trace: dict) -> dict:
    calls = trace["tool_calls"]
    report = trace["report"]
    citations = ct.extract_citations(report["content_markdown"]) if report else []
    names = [c["name"] for c in calls]

    # clarification: was ask_user called before the first save_report?
    first_save = names.index("save_report") if "save_report" in names else len(names)
    asked_before_conclusion = "ask_user" in names[:first_save]

    # errors and self-corrections: an error on a tool, later followed by a success on the same tool
    errors = [i for i, c in enumerate(calls) if isinstance(c["result"], dict) and "error" in c["result"]]
    self_corrections = sum(
        1 for i in errors
        if any(c["name"] == calls[i]["name"] and isinstance(c["result"], dict) and "error" not in c["result"]
               for c in calls[i + 1:])
    )
    # invented articles caught by the guardrails (get_article lookup or save_report check)
    caught = sum(1 for i in errors if "not in the knowledge base" in calls[i]["result"]["error"])
    known = {f"AI Act Art. {n}" for n in ct.ARTICLES["ai_act"]} | {f"GDPR Art. {n}" for n in ct.ARTICLES["gdpr"]}

    return {
        "id": case["id"], "difficulty": case["difficulty"], "incomplete": case["incomplete"],
        "gold": case["risk_category"], "pred": report["risk_category"] if report else "no_report",
        "citations": citations,
        "gold_articles": case["gold_articles"],
        "pred_articles": sorted(set(citations) & CLASSIFICATION_ARTICLES),
        "gold_flags": case["gdpr_flags"],
        "pred_flags": sorted({GDPR_FLAG_OF_ARTICLE[c] for c in citations if c in GDPR_FLAG_OF_ARTICLE}),
        "questions": names.count("ask_user"),
        "asked_before_conclusion": asked_before_conclusion,
        "hallucinated_in_report": [c for c in citations if c not in known],
        "hallucinations_caught": caught,
        "steps": trace["steps"], "tool_calls": len(calls), "tool_errors": len(errors),
        "self_corrections": self_corrections, "nudges": trace["nudges"],
        "stopped_by": trace["stopped_by"], "latency_s": trace["latency_s"],
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def precision_recall(rows: list[dict], gold_key: str, pred_key: str) -> tuple[float, float]:
    """Micro-averaged precision / recall over all cases."""
    tp = sum(len(set(r[gold_key]) & set(r[pred_key])) for r in rows)
    n_pred = sum(len(r[pred_key]) for r in rows)
    n_gold = sum(len(r[gold_key]) for r in rows)
    return (tp / n_pred if n_pred else 0.0), (tp / n_gold if n_gold else 0.0)


def summarize(rows: list[dict]) -> dict:
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    art_p, art_r = precision_recall(rows, "gold_articles", "pred_articles")
    flag_p, flag_r = precision_recall(rows, "gold_flags", "pred_flags")
    incomplete = [r for r in rows if r["incomplete"]]
    complete = [r for r in rows if not r["incomplete"]]
    total_calls = sum(r["tool_calls"] for r in rows)
    return {
        "n_cases": len(rows),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
        "confusion_matrix": {"labels": LABELS + ["no_report"],
                             "matrix": confusion_matrix(gold, pred, labels=LABELS + ["no_report"]).tolist()},
        "reports_saved": sum(r["pred"] != "no_report" for r in rows),
        "article_precision": art_p, "article_recall": art_r,
        "gdpr_flag_precision": flag_p, "gdpr_flag_recall": flag_r,
        "asked_on_incomplete": f"{sum(r['asked_before_conclusion'] for r in incomplete)}/{len(incomplete)}",
        "no_question_on_complete": f"{sum(r['questions'] == 0 for r in complete)}/{len(complete)}",
        "hallucinated_citations_in_reports": sum(len(r["hallucinated_in_report"]) for r in rows),
        "hallucinations_caught_by_guardrails": sum(r["hallucinations_caught"] for r in rows),
        "avg_steps": sum(r["steps"] for r in rows) / len(rows),
        "tool_error_rate": sum(r["tool_errors"] for r in rows) / total_calls if total_calls else 0.0,
        "self_corrections": sum(r["self_corrections"] for r in rows),
        "nudges": sum(r["nudges"] for r in rows),
        "max_steps_reached": sum(r["stopped_by"] == "max_steps" for r in rows),
        "avg_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
    }


def to_markdown(summaries: dict) -> str:
    versions = list(summaries)
    lines = ["| Metric | " + " | ".join(versions) + " |", "|---" * (len(versions) + 1) + "|"]
    for key in summaries[versions[0]]:
        if key == "confusion_matrix":
            continue
        values = [summaries[v][key] for v in versions]
        values = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in values]
        lines.append(f"| {key} | " + " | ".join(values) + " |")
    for v in versions:
        cm = summaries[v]["confusion_matrix"]
        lines += ["", f"**Confusion matrix — {v}** (rows = gold, columns = predicted)", "",
                  "| gold \\ pred | " + " | ".join(cm["labels"]) + " |",
                  "|---" * (len(cm["labels"]) + 1) + "|"]
        for label, row in zip(cm["labels"], cm["matrix"]):
            if label != "no_report":  # gold is never "no_report"
                lines.append(f"| {label} | " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v0", "v1"], choices=["v0", "v1"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases")
    parser.add_argument("--out", default=str(EVAL_DIR / "results.json"))
    args = parser.parse_args()

    cases = json.loads((EVAL_DIR / "cases.json").read_text())["cases"][: args.limit]
    results = {"model": v0.MODEL, "date": datetime.now().isoformat(timespec="seconds"),
               "n_cases": len(cases), "rows": {}, "summary": {}}

    for version in args.versions:
        rows = []
        for case in cases:
            start = time.time()
            try:
                row = run_case(case, version)
            except Exception as e:  # e.g. Ollama crash: record it, keep going
                row = {"id": case["id"], "error": repr(e)}
            rows.append(row)
            print(f"[{version}] {case['id']:3} gold={case['risk_category']:13} "
                  f"pred={row.get('pred', 'ERROR'):13} q={row.get('questions')} steps={row.get('steps')} "
                  f"errors={row.get('tool_errors')} {time.time() - start:.0f}s", flush=True)
            results["rows"][version] = rows
            Path(args.out).write_text(json.dumps(results, indent=2))  # save progress after each case

        ok_rows = [r for r in rows if "error" not in r]
        results["summary"][version] = summarize(ok_rows)
        results["summary"][version]["crashed_runs"] = len(rows) - len(ok_rows)
        Path(args.out).write_text(json.dumps(results, indent=2))

    results["markdown"] = to_markdown(results["summary"])
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\n" + results["markdown"])


if __name__ == "__main__":
    main()
