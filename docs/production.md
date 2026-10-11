# Training job readiness

Run `python scripts/preflight.py` after downloading data and before starting a GPU job. Override `--data` and `--tokenizer` for a different local layout. The dependency-free preflight requires both training and validation splits and checks every shard's magic, version, token count, and exact byte size. It validates file integrity, not tokenizer semantics or model quality.

The downloader now rejects escaping manifest paths and atomically publishes independent copies from the Hugging Face cache. A failed copy leaves no completed-looking shard, and editing a local artifact cannot corrupt that cache. Tests run entirely offline with tiny synthetic shards and interrupted copies.

Pin the selected environment and dataset/tokenizer versions when provisioning GPU workers; record those versions, the Git commit, hyperparameters, and final artifact checksums with run logs. Use the existing documented MLX or PyTorch commands only after preflight passes. This is a training repository with no serving API; training scores and GPU performance still require the intended hardware.
