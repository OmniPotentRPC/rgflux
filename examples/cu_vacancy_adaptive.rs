//! Cu-vacancy adaptive PAFI campaign entry point.
//!
//! The published `rgpot-core` 3.7.0 crate exposes the generic
//! `PotentialImpl` callback contract, but no constructor for rgpot's
//! `EAMTablePot`.  Consequently rgflux cannot construct the Mishin Cu01
//! potential from the crates.io API without rebuilding or vendoring rgpot
//! C++, both of which would violate the dependency boundary.  This example
//! remains an explicit, executable diagnostic until an rgpot-core release
//! exports `eam_table_from_file` / `eam_table_from_data`.

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    Err(
        "Cu vacancy campaign unavailable: requires an rgpot-core release exporting eam_table_from_file/eam_table_from_data"
            .into(),
    )
}
