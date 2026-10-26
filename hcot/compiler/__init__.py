"""The Gated DeltaNet compiler used for the constructive bound (hcot.p3) and the planted tracker
(hcot.plant).

Vendored from chipformer (github.com/divagr18/chipformer, commit 55331f4): dsl/graph.py and
q35/{ir,prims,emit,tiny}.py, plus the three helpers from q35/merge.py that hcot.plant uses.
Only the imports were changed, and the CHIP-8 entry points were dropped from tiny.py.
"""
