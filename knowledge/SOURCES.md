# Sources of the knowledge base

> **Disclaimer:** the files in `knowledge/` are **simplified summaries written for an educational prototype**. They are not legal advice and do not replace the official text. Always read the article on EUR-Lex before relying on it.

## Official texts (EUR-Lex)

| Text | Reference | Link |
|---|---|---|
| AI Act | Regulation (EU) 2024/1689 | https://eur-lex.europa.eu/eli/reg/2024/1689/oj |
| Digital Omnibus on AI (amends the AI Act) | Regulation (EU) 2026/1744 of 8 July 2026, OJ L 24.7.2026 | https://eur-lex.europa.eu/eli/reg/2026/1744/oj |
| GDPR | Regulation (EU) 2016/679 | https://eur-lex.europa.eu/eli/reg/2016/679/oj |

## Where each summary comes from

| File | Content | Source |
|---|---|---|
| `articles.json` → `ai_act` | Art. 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 26, 27, 43, 49, 50, 86, 95, 99 | Regulation (EU) 2024/1689; Art. 4, 5 and 10 as amended by Regulation (EU) 2026/1744 |
| `articles.json` → `gdpr` | Art. 5, 6, 9, 22, 35 | Regulation (EU) 2016/679 |
| `prohibited_practices.json` | Art. 5(1)(a)–(h), plus (ba) and (bb) | Art. 5 of Regulation (EU) 2024/1689; points (ba) and (bb) added by Regulation (EU) 2026/1744 |
| `annex_iii.json` | Annex III points 1–8 | Annex III of Regulation (EU) 2024/1689 |

The summaries keep the structure of the official text (article numbers, points, exceptions) but use shorter wording. Some details are left out on purpose, for example the full conditions of Art. 5(2)–(7) on real-time biometric identification, the derogations of Art. 9(2) GDPR, and the SME-specific rules.

## AI Act timeline (checked on 16 October 2026)

The timeline was changed by the Digital Omnibus on AI (Regulation (EU) 2026/1744, in force since 27 July 2026). Dates as of the check date:

| Date | What applies |
|---|---|
| 1 Aug 2024 | AI Act enters into force |
| 2 Feb 2025 | Prohibitions (Art. 5(1)(a)–(h)) and AI literacy (Art. 4) |
| 2 Aug 2025 | Governance rules and obligations for general-purpose AI models |
| 2 Aug 2026 | Transparency obligations (Art. 50) and most remaining provisions. Grace period until 2 Dec 2026 for the machine-readable marking of Art. 50(2) for systems placed on the market before 2 Aug 2026 |
| 2 Dec 2026 | New prohibitions Art. 5(1)(ba) and (bb) |
| 2 Dec 2027 | High-risk obligations for Annex III systems (Art. 6(2)) — postponed from 2 Aug 2026 |
| 2 Aug 2028 | High-risk obligations for Annex I systems (Art. 6(1)) — postponed from 2 Aug 2027 |

**How this was verified:** the EUR-Lex record of Regulation (EU) 2026/1744 confirms its date and publication, but its full text could not be retrieved programmatically at check time. The amended Art. 113 dates were read from the consolidated text on [artificialintelligenceact.eu](https://artificialintelligenceact.eu/ai-act-explorer/digital-omnibus/) and cross-checked with law firm notes ([Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/), [Jones Walker](https://www.joneswalker.com/en/insights/blogs/ai-law-blog/yes-august-2-still-matters-the-eu-approved-a-high-risk-ai-delay-but-most-trans.html)). **Check the current timeline on EUR-Lex before relying on these dates.**

The agent itself does not use dates: it classifies the risk category and lists obligations, which does not depend on when they apply.
