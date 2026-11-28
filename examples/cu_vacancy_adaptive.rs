//! Cu-vacancy adaptive PAFI campaign entry point.
//!
//! `RgpotSurface::from_eam_table_file` and `from_eam_table_memory` construct
//! rgpot's `EAMTablePot` when `RGPOT_DIR` points at the rgpot repository root
//! or its `CppCore` directory, or after `ci/fetch_rgpot_cpp.sh` fetches the pin.
//! When `RGPOT_DIR` is unset, the build uses that fetched tree when it is
//! already at the pin, and otherwise reuses an existing cargo git checkout of
//! the pin. The published `rgpot-core` crate still does not export
//! `eam_table_from_file` or `eam_table_from_data`, so
//! `cu_reference::run_npt_lattice` refuses. This example remains the
//! diagnostic for that refusal and does not substitute a stored lattice.

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    match rgflux::cu_reference::cu_vacancy_comparison(1000.0, 0.0) {
        Ok(_) => Ok(()),
        Err(rgflux::FluxError::InvalidConfig(message)) => Err(message.into()),
        Err(other) => Err(other.into()),
    }
}
