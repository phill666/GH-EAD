# Attribution and modification notices

Foundation: https://github.com/nelson1425/EfficientAD (public unofficial implementation). Its Apache-2.0 LICENSE is retained verbatim. Upstream LICENSE source was checked on2026-10-26: https://raw.githubusercontent.com/nelson1425/EfficientAD/main/LICENSE . Existing source attribution is retained.

GH-EAD modifications relative to that foundation are the GSHConv/HICGate v2 factory/modules, local dataset/config adapters and the shared evaluation interface. efficientad.py, common.py and tools/evaluate_model.py retain current audited project behavior. Dataset-root defaults in the prior clean package are portable; provenance hashes identify those copies. New release changes affect documentation, config recipes and thin CLI wrappers only; no existing experiment/model/checkpoint is edited. Source lineage is recorded in SOURCE_PROVENANCE.json and RELEASE_MANIFEST.json. No new copyright owner identity or blanket alternative license is invented.

The seed-aware training wrapper is a release addition. modules/gsh_switches.py is the exact existing sensitivity extension and is separate from the canonical default. Data, pretrained-teacher and dependency licensing remains with providers. No data or weights are distributed. Historical author/run attribution is not asserted by this release.
