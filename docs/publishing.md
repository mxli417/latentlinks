# Publishing to PyPI

Releases are built with `uv` and published by GitHub Actions through PyPI Trusted
Publishing. No long-lived PyPI token is stored in the repository.

## One-time setup

1. Confirm that `latentlinks` is available on PyPI.
2. Create the PyPI project, or configure a pending Trusted Publisher for:
   - GitHub owner: `mxli417`
   - repository: `latentlinks`
   - workflow: `release.yml`
   - environment: `pypi`
3. Create a protected GitHub environment named `pypi`.

## Release checklist

1. Move the pending changelog entries into a dated release section.
2. Set the version and refresh the lockfile:

   ```console
   uv version 0.1.0
   uv lock
   ```

3. Run the complete local verification:

   ```console
   uv sync --frozen
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest --cov
   uv build --no-sources
   uv run twine check dist/*
   ```

4. Commit the version and changelog, then create and push the matching tag:

   ```console
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin main v0.1.0
   ```

The tag starts the release workflow. Its build job tests both the wheel and the
source distribution before the publish job receives an ephemeral PyPI identity.
Published filenames cannot be replaced on PyPI; corrections require a new
version.

For a manual TestPyPI rehearsal, configure a separate index and run
`uv publish --index testpypi`. Never commit a token or `.pypirc` file.
