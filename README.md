# polars-tdms

Read NI TDMS files into [polars](https://pola.rs/) DataFrames and LazyFrames.
Built on the [tdms-rs](https://crates.io/crates/tdms-rs) Rust crate via a
PyO3 extension, with the Python layer adding a polars-native API.

## Install

### From GitHub (prebuilt wheels, no Rust toolchain)

On tagged releases, wheels are built for all platforms and attached to [GitHub Releases](https://github.com/szst11/polars_tdms/releases). These wheels use the CPython 3.14 stable ABI (abi3) and work on Python 3.14+.

1. Find the release tag you want (e.g., `v0.1.0`).
2. Select the wheel that matches your platform and architecture (e.g., `polars_tdms-0.1.0-cp314-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`).
3. Install it directly with uv:

```bash
uv add "polars-tdms @ https://github.com/szst11/polars_tdms/releases/download/v0.1.0/polars_tdms-0.1.0-cp314-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
```

Typical wheel suffixes:
- Linux manylinux: `manylinux_2_17_x86_64.manylinux2014_x86_64`, `manylinux_2_17_aarch64.manylinux2014_aarch64`, `manylinux_2_17_i686.manylinux2014_i686`, `manylinux_2_17_armv7l.manylinux2014_armv7l`, `manylinux_2_17_ppc64le.manylinux2014_ppc64le`, `manylinux_2_17_s390x.manylinux2014_s390x`
- Linux musllinux: `musllinux_1_2_x86_64`, `musllinux_1_2_aarch64`, `musllinux_1_2_i686`, `musllinux_1_2_armv7l`
- Windows: `win_amd64`, `win32`, `win_arm64`
- macOS: `macosx_10_12_x86_64`, `macosx_11_0_arm64`

### From source (requires Rust)

To build from a Git checkout with uv:

```bash
uv sync --all-extras          # development + benchmark deps
uv sync                       # library only (requires Rust toolchain)
```

Rust ≥ 1.75 required at build time.
