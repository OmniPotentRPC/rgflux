# Contributor guide

## Repository layout

`src/` contains the Rust library. The public method modules are `pafi`,
`ohtst`, `hyperplane`, `ffs`, `network`, `kappa`, `stats`, and `engine`.
Optional engine adapters live below `src/engine/`. Integration tests are in
`tests/`. Canonical documentation sources are Org files in `docs/orgmode/`.
Files in `docs/source/` are Sphinx support files and exported documentation
is generated during the documentation build. CI helpers are in `ci/`.

## Build and test

Use the toolchain selected by `rust-toolchain.toml`.

```sh
cargo build --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
```

Feature gated engines need their external libraries. The default, `rgpot`,
and `rgpot-session` feature combinations can be checked with:

```sh
cargo check --locked --features rgpot
cargo check --locked --features rgpot-session
cargo check --locked --features rgpot,rgpot-session
```

## Documentation

Edit the existing page in `docs/orgmode/`. Do not edit exported RST output.
Every public Rust item, field, variant, and method needs a reference entry.
Method documentation also needs a runnable example, the estimator equations,
and verified literature citations where applicable.

Generate and validate the language model indexes with:

```sh
python3 scripts/generate_llms.py
python3 scripts/generate_llms.py --check
```

Run the documentation checks exactly as CI does:

```sh
python3 docs/check_paths.py
python3 ci/check_public_api_docs.py
python3 ci/check_links.py
emacs --batch -l docs/export.el
python3 -m pip install -r docs/requirements.txt
sphinx-build -b linkcheck -n -W --keep-going docs/source target/docs-linkcheck
sphinx-build -b html -n -W --keep-going docs/source docs/build
ci/check_rustdoc.sh
```

## Public configuration convention

Public configuration structs are marked `#[non_exhaustive]`. Construct them
with `Default::default()` or a named constructor, then use chainable
`#[must_use]` `with_*` methods. Do not require downstream users to write a
struct literal. A setter for `Option<T>` accepts `impl Into<Option<T>>`, so a
bare value sets `Some(value)` and `None` clears it. New configuration fields
must have a documented default, validation rule, setter, reference entry, and
test coverage.

## Commit messages

Use Conventional Commit subjects such as `docs: complete the API reference`
or `ci: verify generated documentation`. Keep subjects at 72 characters or
fewer and wrap body text at 72 columns. Describe one logical change per
commit. Do not add generated attribution or signoff trailers.
