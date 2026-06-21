# external/

Editable-installed, **unmodified** upstream checkouts used as independent cross-checks.
Everything here except this file and `versions.lock` is gitignored (see root `.gitignore`):
these are separate repos, not vendored code, and plutus never patches their internals.

## zipline-reloaded — independent friction cross-check (optional)

The plutus backtest engine is hand-rolled. To guard against a bug in our own friction model,
a candidate strategy can be re-run through `zipline-reloaded`, which independently models US
commissions/slippage and a US trading calendar. This is the role RQAlpha plays for hermes.
It is OPTIONAL — research and tests do not depend on it.

```
git clone https://github.com/stefan-jansen/zipline-reloaded.git external/zipline-reloaded
conda activate plutus
pip install -e external/zipline-reloaded
```

Record the exact commit you pinned in `versions.lock` so a cross-check is reproducible.
