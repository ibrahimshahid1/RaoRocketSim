# PyPI Release Checklist

LREKit is packaged with `pyproject.toml` and publishes through GitHub
Actions plus PyPI trusted publishing. Trusted publishing avoids storing a PyPI
API token in GitHub secrets.

## One-Time Setup

1. Choose and add a top-level `LICENSE` before the first public PyPI release.
   The vendored NASA reference tree already carries its own NASA Open Source
   Agreement text in `Three-Dimensional-Nozzle-Design-Code-master/license.txt`,
   but the package as a whole still needs an explicit project license decision.
2. Create or sign in to accounts on TestPyPI and PyPI.
3. On TestPyPI, create a trusted publisher for:
   - PyPI project name: `lrekit`
   - Owner: `ibrahimshahid1`
   - Repository: `RaoRocketSim`
   - Workflow: `publish.yml`
   - Environment: `testpypi`
4. On PyPI, create the matching trusted publisher with environment `pypi`.
5. In GitHub repository settings, create environments named `testpypi` and
   `pypi`. Consider requiring manual approval on the `pypi` environment.

## Local Preflight

Use Python 3.12 for release checks because the pinned JAX stack is not meant
for Python 3.13+.

```bash
python3.12 -m venv .venv-release
source .venv-release/bin/activate
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall --no-deps dist/*.whl
lrekit --help
```

## TestPyPI

Run the `Publish Python package` workflow manually from GitHub Actions. A manual
run publishes only to TestPyPI.

Then test from a clean environment:

```bash
python3.12 -m venv .venv-testpypi
source .venv-testpypi/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ lrekit
lrekit --help
```

## Real PyPI

After TestPyPI and a Windows install smoke test are green:

```bash
git tag v0.1.0
git push origin v0.1.0
```

If an earlier failed pre-release `v0.1.0` tag already exists, replace it before
publishing the renamed `lrekit` package:

```bash
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0
git tag v0.1.0
git push origin v0.1.0
```

The tag must match the version in `pyproject.toml`. The publish workflow checks
that before uploading to PyPI.

For each later release:

1. Update `pyproject.toml` and `raosim/__init__.py`.
2. Update any release notes.
3. Run the local preflight.
4. Publish to TestPyPI.
5. Tag and push the matching version.
