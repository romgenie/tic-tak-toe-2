# Release process

This project aims for research-grade reproducibility. Follow these steps when cutting a release.

1) Version and tag
	- Update version in `pyproject.toml` if needed.
	- Create an annotated tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push --tags`.

2) PyPI publish and small dataset
	- GitHub Actions workflow `release.yml` will build sdist/wheel and publish to PyPI via trusted publishing.
	- It also exports a minimal reference dataset (`data_raw/release_small`) and attaches it to the GitHub Release.

3) Environment locks
	- Pip (hash-locked): ensure `requirements-lock.txt` is up-to-date. Regenerate from the repo root using pip-tools:
		- `pip-compile pyproject.toml --extra=dev --extra=parquet --extra=tracking --generate-hashes -o requirements-lock.txt`
	- Conda (platform locks): use `environment.yml` as the source and generate per-platform locks:
		- `conda-lock lock -f environment.yml -p linux-64 -p osx-arm64 -p win-64 -p osx-64`
		- Commit the resulting `conda-*.lock.txt` files.

4) SBOM and provenance
	- CI generates a CycloneDX SBOM and attaches `sbom.json` to the release.
	- Dataset manifests contain checksums and environment lock hashes for provenance.

5) DOI (Zenodo)
	- Ensure Zenodo is enabled for the repository.
	- After the GitHub Release is created, Zenodo will archive it and mint a DOI.
	- Update the README DOI badge to point to the latest concept DOI or versioned DOI.

6) Post-release checks
	- Verify docs build and Pages deployment.
	- Verify CI green on main post-merge.

