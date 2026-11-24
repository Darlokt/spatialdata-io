# Optional test-data downloader

Run the downloader from the repository root. List the registered keys and reader groups with either command:

```bash
uv run --group dev python scripts/test_data_downloader --list
# or
hatch run python scripts/test_data_downloader --list
```

Replace `--list` with `--dataset KEY` or `--group GROUP` to install the
selection under `data/`. The [dataset registry](datasets.toml) records the
producer, version, source terms and files for each entry.

The [developer guide](../../docs/development/datasets.md) covers shared download
locations, checksum verification, registering a ZIP, DOI or selected files,
and running the integration tests against a fresh download.
