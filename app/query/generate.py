"""Stage 8 — answer generation from the top 5 passages.

Guardrail tier 2 lives in the prompt: answer only from the supplied
passages, "not in these documents" is a correct answer, never paraphrase
a number, tolerance or part code. Calls the generator through
app.models.generator — never a vendor SDK directly.
"""
