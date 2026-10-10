# Evaluation question set — phase 0

80–120 real installation and commissioning questions with known answers,
drawn from the manuals in `Documents/`.

Each entry needs:

| Field | Why |
|---|---|
| `question` | Phrased the way an integrator on site would actually ask it, including the vague ones |
| `expected_answer` | The correct answer, as a human verified it against the manual |
| `expected_source` | Document, section and page that supports it — this is what makes phase 1 measurable before any generation exists |
| `in_scope` | `false` for questions the assistant should refuse (sales, warranty, third-party parts) — abstention is measured, not just accuracy |
| `category` | wiring / fault code / step sequence / parameter / compatibility |

Include deliberately unanswerable questions. A set containing only
answerable questions measures coverage and cannot measure the thing that
matters most here, which is whether the system refuses when it should.

This file is the deliverable of phase 0 and it is not optional. Writing
it takes about a day and it is the highest-value day in the project.
