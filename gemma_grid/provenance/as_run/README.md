# Scripts as they ran

These files are byte-for-byte copies of scripts whose hashes the study's records cite: a result file, a manifest, or the findings ledger. They carry the working names and paths they ran under, so they do not run from this release's layout. They are here so that each cited hash can be checked with `shasum -a 256`. The runnable copy of each is in `scripts/` under its release name. `RELEASE_MANIFEST.md` lists both hashes and every changed line.

| As run | sha256 | Release copy | Cited for |
|---|---|---|---|
| `pmap_transport_100T.py` | `ee736f5d36c9a04c8d6e4eaf043d932cbb979a5e4fba22a0ea0e8613b91e5526` | `scripts/pmap_transport.py` | the frozen transport estimator (§4.3), cited in pmap_transport_floor.json and the battery and disjoint-topic manifests |
| `analysisB_seam_onset_bootstrap_100T.py` | `11a127903a49ce353b0ae6a14072c0d450f9c29fb8a64b91c51b41d4159f2621` | `scripts/analysisB_seam_onset_bootstrap.py` | the frozen seam and onset estimator, the registered definition (§6.3) |
| `analysisB_seam_onset_bootstrap_100T_gesvdfix.py` | `a9e5e3767a2c72645929d63948a20a3733566b2058d5121435d6fbf97f364c30` | `scripts/analysisB_seam_onset_bootstrap_gesvdfix.py` | its gesvd successor, which produced analysisB_seam_onset_bootstrap.json |
| `g54a_displacement_model_100T.py` | `e76e40266e40fe682b709d7c25f11089eaf4cdd881c0283c41a76cc2ee5c4e4a` | `scripts/g54a_displacement_model.py` | g54a_displacement_model.json (§7.4) |
| `pmap_embedding_table_transport_100T.py` | `d79ca223a74330aca139c609378dcf2fff227708a39620a57f08b18a36bad7aa` | `scripts/pmap_embedding_table_transport.py` | pmap_embedding_table_transport.json (§4.3) |
| `item3_corrected_100T.py` | `6a99571a5543059dbb8f9d12d7c022e07c20dd73adaefb92cdb640ede04ae92d` | `scripts/item3_corrected.py` | item3_splithalf_corrected.json, before the release guard was added |
| `pmap_battery_serial_100T.py` | `5413c04f5ec64aaa2c0db29d7200c3a6c9cc74548e6325b936e4a907c97faef5` | `scripts/pmap_battery_serial.py` | pmap_battery.json (Appendix D), the transport battery's driver |
