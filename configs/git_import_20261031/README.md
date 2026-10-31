# Initial private-repository import preparation

This directory supplies a strict source allowlist, an auditable size/SHA256 manifest, and a non-mutating secret scanner. It does not initialize Git, stage files, read credentials, create a repository, or change a running deployment.

Run from the repository root with the existing numerical Python environment:

```bash
/Users/slowlight/miniconda3/envs/nqs/bin/python configs/git_import_20261031/prepare_source_import.py
```

Review `generated/source_manifest.json`, `generated/candidate_secret_scan.json`, `generated/excluded_files.json`, and `generated/external_fixture_risks.json`. `generated/source_paths.nul` is a **literal NUL-separated** list of approved candidates, not a command to stage them. The generated directory is ignored so the manifest does not recursively hash itself. Any source edit after generation requires regeneration and review.

After the root agent initializes Git and stages only approved paths, scan the actual index bytes:

```bash
/Users/slowlight/miniconda3/envs/nqs/bin/python configs/git_import_20261031/prepare_source_import.py --scan-staged
```

This second command uses read-only `git ls-files`/`git show :path`; it never adds or changes index entries. Its report is `generated/staged_secret_scan.json`. It fails on findings, forbidden paths, missing candidates, or staged content differing from the approved manifest. Candidate scanning is not a substitute for this gate. The scanner reports only file path, line and rule category—never matched values. It is a deterministic high-confidence pattern check, not a claim that no credential can exist; a separate established secret scanner is still desirable before publication.

Seven numeric NPZ exceptions are individually listed and must contain only numeric/boolean arrays with `allow_pickle=False`; their companion JSON manifests remain in source. Ten current root-publication PNG/PDF figures are explicit small binary exceptions. Generated project notebooks, status PDFs, all result/log/history/environment/private-key paths, and non-exempt binaries are excluded. Tests referring to excluded scientific assets are retained and listed as integration-fixture risks, without weakening them or inventing replacement data.
