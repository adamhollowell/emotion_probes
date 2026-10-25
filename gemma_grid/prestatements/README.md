# Pre-stated analysis filings

Eight decision rules and analyses in the paper were not part of the registration. Each was fixed in a written filing before its data were read. For six of them the sha256 of the file at the time it was frozen is the evidence of that order. One dated amendment to the disjoint-topic filing sits beside that filing in the first table, with its own hash. The colon-token design and the layer-45 replication rule were fixed by dated filings rather than hashes, and their dates are the evidence. They are in the second table.

The files here are working documents reproduced exactly as filed, byte for byte, so each hash below can be checked with `shasum -a 256`. They name the people and working sessions that drafted, froze, and ran them, and refer to other files by the names they had at the time. One of them, the filing for analyses A, B, and C, also names an analysis from a companion study, which is not part of this release.

## Fixed by hash

| File | Filed as | sha256 | Frozen | Analysis | Paper |
|---|---|---|---|---|---|
| `P_MAP_TRANSPORT_SPEC_FROZEN.md` | `P_MAP_TRANSPORT_SPEC_FROZEN_100T.md` | `09b4dacdafa6e3b0e9aa4b0c4d75b4a44d2aaa74b2c041303e7919ae5833ad38` | 2026-08-27 | Cross-model linear transport of the emotion vectors and its pre-set bar | §4.3, Appendix D |
| `PMAP_ROBUSTNESS_BATTERY_PRESTATEMENT.md` | `PMAP_STRESSTEST_BATTERY_PRESTATEMENT_100T.md` | `089fec1e4222bc91582fdc64c40430c5fef7a9e90894d44faeebd35bd2d3ef76` | 2026-09-02 | The eight-item robustness battery for the transport result | Appendix D |
| `APPENDED_ANALYSES_A_B_C_PRESTATEMENT.md` | `PODJOB_PMAP_APPENDED_ANALYSES_PRESTATEMENT_100T_CORRECTED.md` | `09eaa4fb42f9f1a1333aa03d16880fe5c7e0c2f73bcc9a8315047cd282fd2eef` | 2026-08-27 | A, the cleanup-basis stability of the displacement ranking. B, the seam and onset bootstrap. C, the dependence-aware test of the arousal depth gap | §7.4, §6.3, §5.3 |
| `B1_B2_EXPLORATORY_PRESTATEMENT.md` | `B1_B2_EXPLORATORY_PRESTATEMENT_100T.md` | `ae05e8bdcd5611558f0999217f2555c1037a52cc95ef2e4e033fa589cf06569a` | 2026-09-02 | B1, the threshold-free onset knee. B2, the covariate-transfer test | §6.3, §7.4 |
| `DISJOINT_TOPIC_TRANSPORT.md` | `DISJOINT_TOPIC_TRANSPORT.md` | `aab3eef2dcb6d4ef1af38d4e49b2cc81cf502e23e56804f0a3c62dfdfa6c2d6b` | 2026-10-19 | Transport between disjoint topic sets, its matched same-topic control, the within-model topic-split reference, and a bag-of-words baseline | §4.3, Appendix D |
| `DISJOINT_TOPIC_TRANSPORT_AMENDMENT_1.md` | `DISJOINT_TOPIC_TRANSPORT_AMENDMENT_1.md` | `91d5f365ee68ee15837d4e48048da00923106f49d5afda128fe93f78cc7a641e` | 2026-10-19 | Amendment 1 to the disjoint-topic filing: withdraws the fitted retrieval form of the bag-of-words baseline before it was run; the direct form is reported alone. The original filing is unchanged | §4.3, Appendix D |
| `LAYER_BAND_SWEEP_SPEC.md` | `OP3_BAND_SWEEP_SPEC.md` | `87d76939646def49146bb4d719fcb368d91438759424e7701883d59bfcfb1a05` | 2026-08-11 | The sweep of layers 42 to 45 in the base 27B model | §8.3, Appendix H |

## Fixed by dated filing

The date is the record for these two filings. No hash was recorded when either was filed. The hash in the last column was computed from the released copy when this release was staged, so it identifies the file shipped here and is not evidence of when the filing was made.

| File | Filed as | Filed | Analysis | Paper | sha256 of the released copy, computed at staging |
|---|---|---|---|---|---|
| `IT_COLON_ANALYSIS_DESIGN.md` | `IT_COLON_ANALYSIS_DESIGN.md` | 2026-08-10, amended 2026-08-11 (the amendment sits inside the file) | The colon-token prediction on the tuned models | §7.2, Appendix G | `bd5cea362090e20db116067c9df57beda585f0cf16cb7cf34078494ed8c52d91` |
| `LAYER45_REPLICATION_RULE.md` | `PROBE_L45_CLAUDE_CODE_HANDOFF.md` | 2026-07-27 | The decision rule for replicating the base 27B model's layer-45 divergence, in its §5 | §8.3, Appendix H | `57a9d9885a0d702f22678450199e8a0d3cd3b049056a1c51dbda146c65593386` |

The amendment's body abbreviates the original filing's hash as `aab3eef2…6d6b`. The full hash ends `…2d6b` and is correct in the amendment's header line and in the first table. The amendment is reproduced as filed, so the abbreviation is left as it stands.

The transport result file `geometry_out/pmap_transport.json` carries the specification's hash in its `spec_sha256` field.
