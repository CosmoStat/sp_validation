---
id: 01KX8D3CDT73EYYTXCF72XH9CV
name: 'cosmo-numba: use @main; a git dep with dependencies=[] resolves to zero deps in uv'
tags:
    - finding
    - gotcha
    - deps
created-at: 2026-07-11T12:55:47.643001731+02:00
updated-at: 2026-07-11T12:56:25.583499035+02:00
---

Two durable facts about depending on `cosmo-numba` (`aguinot/cosmo-numba`), learned the slow way while locking sp_validation.

**Depend on `@main`, not the fork.** cosmo-numba is **not on PyPI**. Its refs diverge in a way that matters:
- `aguinot/cosmo-numba@main` — declares its deps dynamically from `requirements.txt` (uv reads this correctly) and gets numpy-2 FFT support from its **`rocket-fft`** dependency (rocket-fft teaches numba's nopython mode to compile `np.fft`). This is the ref you want.
- `cailmdaley/cosmo-numba@fix/numpy2-fft-compat` (tip `db452c48`) — a dead end. It carries an earlier `nb.objmode()` FFT workaround but its `pyproject.toml` declares `dependencies = []` (empty). Cail's PR of that workaround (aguinot#17) was **closed** in favor of the rocket-fft approach (aguinot#18, merged) — his own closing comment: *"real solution is rocket-fft dependency."*

**The uv gotcha that cost the time:** when a git/source dependency's `pyproject.toml` declares `dependencies = []` (or otherwise advertises no requirements), **uv silently resolves it to itself with zero transitive deps — no error, no warning** ("Resolved 1 package"). The lock entry simply has no `dependencies` block. So if a package you depend on isn't pulling in what you expect, check the *ref you're actually resolving* declares them — don't assume a uv or `requirements.txt` bug (uv reads `[tool.setuptools.dynamic] dependencies = {file = "requirements.txt"}` fine, verified against a minimal repro four ways). Corollary defensive move: pin the load-bearing transitive (here `numba`, for its `numpy<2.5` ceiling) directly in your own `pyproject.toml`, so the constraint survives the dependency's metadata going empty between refs.

**Process lesson:** the fast path was one command — `uv pip compile 'cosmo-numba @ git+…@main'` — which would have shown deps propagating and rocket-fft appearing immediately. Try the cheap alternative ref *before* dissecting why the first ref failed, especially when the premise that a ref is special came from an unverified source (a subagent audit that also hallucinated a PyPI release).
