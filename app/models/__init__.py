"""Model layer — four roles, each behind one interface.

Generator, Embedder, Reranker, Transcriber. Every role defines a protocol
in base.py and one implementation per backend. Nothing outside this
package imports a vendor SDK or touches a model endpoint directly, and
nothing outside this package knows whether a role is currently served
from an API or from local weights.

Selecting a backend is a config change (app.core.config), not a branch in
calling code. This is the boundary that makes the offline/on-premise
profile a deployment variant rather than a rewrite.

Each role has a cloud implementation used now and a local implementation
for the offline profile (phase 7). Cloud first because the hard part of
this project is retrieval quality, not model hosting — an API key takes
hosting off the critical path, where provisioning GPUs would not.
"""
