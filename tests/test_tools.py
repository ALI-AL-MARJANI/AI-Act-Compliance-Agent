"""Unit tests for tools/compliance_tools.py (no LLM required)."""

import pytest

from tools import compliance_tools as ct

GOOD_REPORT = f"""{ct.DISCLAIMER}

# Assessment: CV ranking tool

Risk category: high_risk. Recruitment is listed in Annex III point 4(a) [AI Act Art. 6].
Fully automated rejection raises a GDPR concern [GDPR Art. 22].
"""


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Each test gets a fresh question counter and a temporary reports/ folder."""
    ct.reset_assessment()
    ct.set_answer_provider(None)
    monkeypatch.setattr(ct, "ROOT", tmp_path)
    monkeypatch.setattr(ct, "REPORTS_DIR", tmp_path / "reports")
    yield
    ct.set_answer_provider(None)


# --- check_prohibited_practices -------------------------------------------
def test_prohibited_returns_all_practices():
    result = ct.check_prohibited_practices("A chatbot that answers customer questions.")
    ids = [p["id"] for p in result["practices"]]
    assert "5(1)(c)" in ids and "5(1)(f)" in ids


def test_prohibited_empty_description_is_an_error():
    assert "error" in ct.check_prohibited_practices("")


# --- check_high_risk_areas -------------------------------------------------
@pytest.mark.parametrize("domain, expected", [
    ("employment", "employment"),
    ("credit", "essential_services"),
    ("recruitment of job candidates", "employment"),
    ("biometrics", "biometrics"),
    ("critical_infrastructure", "critical_infrastructure"),
])
def test_high_risk_known_domains(domain, expected):
    result = ct.check_high_risk_areas(domain)
    assert expected in [area["domain"] for area in result["matches"]]


def test_high_risk_unknown_domain_returns_valid_domains():
    result = ct.check_high_risk_areas("video games")
    assert result["matches"] == []
    assert "employment" in result["valid_domains"]


def test_high_risk_no_partial_word_match():
    # "hr" is an employment keyword, it must not match inside "three"
    assert ct.check_high_risk_areas("three")["matches"] == []


# --- get_article -----------------------------------------------------------
def test_get_article_ok():
    result = ct.get_article("gdpr", "22")
    assert result["citation"] == "[GDPR Art. 22]"
    assert "automated" in result["summary"].lower()


def test_get_article_accepts_messy_number():
    assert ct.get_article("ai_act", "Article 6(3)")["citation"] == "[AI Act Art. 6]"


def test_get_article_refuses_several_articles():
    assert "ONE article per call" in ct.get_article("gdpr", "5,6,9,22")["error"]


def test_get_article_unknown_article():
    result = ct.get_article("ai_act", "999")
    assert "not in the knowledge base" in result["error"]


def test_get_article_unknown_regulation():
    assert "Invalid arguments" in ct.get_article("hipaa", "5")["error"]


# --- get_obligations -------------------------------------------------------
def test_obligations_high_risk_cite_articles():
    result = ct.get_obligations("high_risk")
    citations = [o["citation"] for o in result["obligations"]]
    assert "[AI Act Art. 14]" in citations and "[AI Act Art. 9]" in citations


def test_obligations_invalid_category():
    result = ct.get_obligations("very_risky")
    assert "Invalid arguments" in result["error"]


# --- ask_user --------------------------------------------------------------
def test_ask_user_uses_provider():
    ct.set_answer_provider(lambda q: "A human reviews every decision.")
    result = ct.ask_user("Does a human review the ranking?")
    assert result["answer"] == "A human reviews every decision."


def test_ask_user_limit():
    ct.set_answer_provider(lambda q: "yes")
    for _ in range(ct.MAX_QUESTIONS):
        assert "answer" in ct.ask_user("Is it used on employees?")
    result = ct.ask_user("One more question?")
    assert "Question limit reached" in result["error"]


def test_ask_user_reset_between_assessments():
    ct.set_answer_provider(lambda q: "yes")
    for _ in range(ct.MAX_QUESTIONS):
        ct.ask_user("Is it used on employees?")
    ct.reset_assessment()
    assert "answer" in ct.ask_user("Is it used on employees?")


# --- save_report -----------------------------------------------------------
def test_save_report_ok():
    result = ct.save_report("CV ranking tool", "high_risk", GOOD_REPORT)
    assert result["status"] == "saved"
    assert (ct.ROOT / result["path"]).read_text() == GOOD_REPORT


def test_save_report_without_citation_refused():
    content = ct.DISCLAIMER + "\n\nThis system is high risk because it is used in recruitment."
    assert "cite at least one article" in ct.save_report("CV tool", "high_risk", content)["error"]


def test_save_report_without_disclaimer_refused():
    content = GOOD_REPORT.replace(ct.DISCLAIMER, "")
    assert "disclaimer" in ct.save_report("CV tool", "high_risk", content)["error"]


def test_save_report_disclaimer_at_the_end_refused():
    content = GOOD_REPORT.replace(ct.DISCLAIMER, "") + "\n" + "x" * 400 + "\n" + ct.DISCLAIMER
    assert "disclaimer" in ct.save_report("CV tool", "high_risk", content)["error"]


def test_save_report_invented_article_refused():
    content = GOOD_REPORT + "\nSee also [AI Act Art. 123]."
    result = ct.save_report("CV tool", "high_risk", content)
    assert "AI Act Art. 123" in result["error"]


def test_save_report_invented_loose_article_refused():
    content = GOOD_REPORT + "\nArticle 30 also applies."
    assert "Art. 30" in ct.save_report("CV tool", "high_risk", content)["error"]


def test_save_report_invalid_category():
    assert "Invalid arguments" in ct.save_report("CV tool", "super_risky", GOOD_REPORT)["error"]


def test_save_report_path_traversal_stays_in_reports():
    result = ct.save_report("../../etc/passwd", "high_risk", GOOD_REPORT)
    saved = (ct.ROOT / result["path"]).resolve()
    assert saved.parent == ct.REPORTS_DIR.resolve()
    assert ".." not in result["path"]


# --- errors never crash ----------------------------------------------------
@pytest.mark.parametrize("name, bad_args", [
    ("get_article", {"regulation": None, "article": None}),
    ("check_high_risk_areas", {"domain": 42}),
    ("save_report", {"title": "", "risk_category": "", "content_markdown": ""}),
])
def test_bad_arguments_return_error_instead_of_raising(name, bad_args):
    function, _ = ct.TOOLS[name]
    result = function(**bad_args)
    assert "error" in result


def test_extract_citations():
    assert ct.extract_citations(GOOD_REPORT) == ["AI Act Art. 6", "GDPR Art. 22"]
