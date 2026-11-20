//! Copper vacancy mobility compared with Daw and Chandross.
//!
//! The published column is Table II of Daw and Chandross, Physical Review
//! Materials 5, 043603 (2021),
//! <https://doi.org/10.1103/PhysRevMaterials.5.043603>, read from
//! `data/daw_chandross_prm_5_043603.csv`. Equation (5) of that paper defines the mobility from
//! the mean-square displacement. A nearest-neighbour fcc hop then relates a
//! total rate `Γ` to a mobility by `M = Γ a² / z`. The lattice `a` in that
//! conversion is the lattice of the run that produced `Γ`.
//!
//! [`run_npt_lattice`] is the in-tree NPT driver that would measure `a(T)`.
//! The published `rgpot-core` 3.7.0 crate has no constructor for the
//! embedded-atom table, so the driver refuses. [`cu_vacancy_comparison`]
//! calls that driver and refuses with it. Neither path substitutes a stored
//! lattice.

use crate::error::FluxError;

/// Cited Table II record packaged with the crate.
const PUBLISHED_RECORD: &str = include_str!("data/daw_chandross_prm_5_043603.csv");

/// Boltzmann constant in eV/K (CODATA 2018).
const BOLTZMANN_EV_PER_K: f64 = 8.617333262145e-5;

/// One row of the Daw and Chandross table, used only as the comparison column.
#[derive(Clone, Debug, PartialEq)]
pub struct PublishedMobility {
    /// DOI of the paper, without a resolver prefix.
    pub doi: &'static str,
    /// Table identifier in that paper.
    pub table: &'static str,
    /// Equation that defines the mobility.
    pub equation: &'static str,
    /// Prefactor `M0`.
    pub m0_a2_per_ps: f64,
    /// Unit of [`Self::m0_a2_per_ps`].
    pub m0_unit: &'static str,
    /// Migration energy `Qm`.
    pub qm_ev: f64,
    /// Unit of [`Self::qm_ev`].
    pub qm_unit: &'static str,
    /// Quoted uncertainty of `Qm`, in eV.
    pub qm_sigma_ev: f64,
    /// Quoted sampling uncertainty of `ln M0`.
    pub ln_m0_sigma: f64,
    /// Number of hop directions summed into `M`.
    pub coordination: f64,
}

/// Lattice measured by [`run_npt_lattice`].
///
/// Only that driver constructs a record. There is no stored `a(T)` to pass
/// in its place.
#[derive(Clone, Debug, PartialEq)]
pub struct NptLatticeRecord {
    temperature_k: f64,
    lattice_angstrom: f64,
}

impl NptLatticeRecord {
    /// Temperature of the NPT run, in kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Cubic lattice constant measured by that run, in Ångström.
    #[must_use]
    pub fn lattice_angstrom(&self) -> f64 {
        self.lattice_angstrom
    }
}

/// Our mobility beside the published column, at one temperature.
#[derive(Clone, Debug, PartialEq)]
pub struct MobilityComparison {
    /// Temperature of the NPT record, in kelvin.
    pub temperature_k: f64,
    /// Lattice from that record, in Ångström.
    pub lattice_angstrom: f64,
    /// `M = Γ a² / z` from the supplied total rate and the NPT lattice, in Å²/ps.
    pub ours_a2_per_ps: f64,
    /// Table II mobility at the same temperature, in Å²/ps.
    pub reference_a2_per_ps: f64,
    /// [`Self::ours_a2_per_ps`] divided by [`Self::reference_a2_per_ps`].
    pub ratio: f64,
}

/// The sentence returned while the embedded-atom table cannot be constructed.
pub const CAMPAIGN_UNAVAILABLE: &str = "Cu vacancy campaign unavailable: requires an rgpot-core release exporting eam_table_from_file/eam_table_from_data";

/// Table II copper row.
///
/// # Errors
///
/// Returns [`FluxError::InvalidConfig`] when the packaged record is missing
/// the DOI, Table II, equation (5), or the Å²/ps and eV units.
pub fn published_copper_mobility() -> Result<PublishedMobility, FluxError> {
    parse_published_record(PUBLISHED_RECORD)
}

/// Table II mobility `M0 exp(−Qm / kT)` in Å²/ps.
///
/// This is the comparison column. It is not a rate, and it is not converted
/// to one.
///
/// # Errors
///
/// Returns [`FluxError::InvalidConfig`] when `temperature_k` is not finite
/// and positive, or when the packaged record cannot be read.
pub fn published_mobility_a2_per_ps(temperature_k: f64) -> Result<f64, FluxError> {
    let published = published_copper_mobility()?;
    mobility_at(&published, temperature_k)
}

/// Run NPT and return the measured lattice.
///
/// The driver has nowhere to obtain the Foiles-Baskes-Daw copper table, so
/// it refuses instead of returning a stored lattice.
///
/// # Errors
///
/// Returns [`FluxError::InvalidConfig`] when `temperature_k` is not finite
/// and positive, and the campaign refusal when the potential cannot be built.
pub fn run_npt_lattice(temperature_k: f64) -> Result<NptLatticeRecord, FluxError> {
    if !(temperature_k.is_finite() && temperature_k > 0.0) {
        return Err(FluxError::InvalidConfig(
            "Cu NPT temperature must be finite and positive",
        ));
    }
    let _measured_at = temperature_k;
    Err(FluxError::InvalidConfig(CAMPAIGN_UNAVAILABLE))
}

/// Compare `total_rate_hz` with Table II at the lattice [`run_npt_lattice`] measures.
///
/// `total_rate_hz` is a rate from this crate. Table II enters only as the
/// denominator. The call refuses with the NPT driver while that driver
/// cannot build the potential.
///
/// # Errors
///
/// Returns the error from [`run_npt_lattice`].
pub fn cu_vacancy_comparison(
    temperature_k: f64,
    total_rate_hz: f64,
) -> Result<MobilityComparison, FluxError> {
    let record = run_npt_lattice(temperature_k)?;
    compare_mobility(total_rate_hz, &record)
}

/// `M = Γ a² / z / 10^12` in Å²/ps. `Γ` is in Hz and `a` is in Ångström.
///
/// `None` when the rate is negative or any argument is non-finite, or when
/// the lattice or the coordination is not positive.
#[must_use]
pub fn mobility_a2_per_ps(
    total_rate_hz: f64,
    lattice_angstrom: f64,
    coordination: f64,
) -> Option<f64> {
    if !(total_rate_hz.is_finite()
        && total_rate_hz >= 0.0
        && lattice_angstrom.is_finite()
        && lattice_angstrom > 0.0
        && coordination.is_finite()
        && coordination > 0.0)
    {
        return None;
    }
    let mobility = total_rate_hz * lattice_angstrom * lattice_angstrom / coordination / 1.0e12;
    mobility.is_finite().then_some(mobility)
}

fn compare_mobility(
    total_rate_hz: f64,
    record: &NptLatticeRecord,
) -> Result<MobilityComparison, FluxError> {
    let published = published_copper_mobility()?;
    let reference = mobility_at(&published, record.temperature_k)?;
    let ours = mobility_a2_per_ps(
        total_rate_hz,
        record.lattice_angstrom,
        published.coordination,
    )
    .ok_or(FluxError::InvalidConfig(
        "Cu mobility comparison needs a finite non-negative rate and the NPT lattice",
    ))?;
    let ratio = ours / reference;
    if !ratio.is_finite() {
        return Err(FluxError::NonFinite("Cu mobility ratio"));
    }
    Ok(MobilityComparison {
        temperature_k: record.temperature_k,
        lattice_angstrom: record.lattice_angstrom,
        ours_a2_per_ps: ours,
        reference_a2_per_ps: reference,
        ratio,
    })
}

fn mobility_at(published: &PublishedMobility, temperature_k: f64) -> Result<f64, FluxError> {
    if !(temperature_k.is_finite() && temperature_k > 0.0) {
        return Err(FluxError::InvalidConfig(
            "Cu comparison temperature must be finite and positive",
        ));
    }
    let mobility =
        published.m0_a2_per_ps * (-published.qm_ev / (BOLTZMANN_EV_PER_K * temperature_k)).exp();
    if mobility.is_finite() {
        Ok(mobility)
    } else {
        Err(FluxError::NonFinite("Cu published mobility"))
    }
}

fn parse_published_record(text: &str) -> Result<PublishedMobility, FluxError> {
    let mut header: Option<Vec<&str>> = None;
    let mut row: Option<Vec<&str>> = None;
    let mut saw_doi = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if line.contains("10.1103/PhysRevMaterials.5.043603") {
                saw_doi = true;
            }
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if header.is_none() {
            header = Some(fields);
        } else if row.is_none() {
            row = Some(fields);
        }
    }
    let header = header.ok_or(FluxError::InvalidConfig(
        "Cu published mobility record has no header",
    ))?;
    let row = row.ok_or(FluxError::InvalidConfig(
        "Cu published mobility record has no row",
    ))?;
    if !saw_doi {
        return Err(FluxError::InvalidConfig(
            "Cu published mobility record does not cite 10.1103/PhysRevMaterials.5.043603",
        ));
    }
    let value = |name: &str| -> Result<&str, FluxError> {
        header
            .iter()
            .position(|field| *field == name)
            .and_then(|index| row.get(index).copied())
            .ok_or(FluxError::InvalidConfig(
                "Cu published mobility record is missing a cited field",
            ))
    };
    let doi = value("doi")?;
    let table = value("table")?;
    let equation = value("equation")?;
    let m0_unit = value("M0_unit")?;
    let qm_unit = value("Qm_unit")?;
    if doi != "10.1103/PhysRevMaterials.5.043603"
        || table != "II"
        || equation != "(5)"
        || m0_unit != "Angstrom^2/ps"
        || qm_unit != "eV"
    {
        return Err(FluxError::InvalidConfig(
            "Cu published mobility record is not Table II equation (5) in Angstrom^2/ps and eV",
        ));
    }
    let number = |name: &str| -> Result<f64, FluxError> {
        value(name)?.parse().map_err(|_| {
            FluxError::InvalidConfig("Cu published mobility record has a non-numeric field")
        })
    };
    Ok(PublishedMobility {
        doi: "10.1103/PhysRevMaterials.5.043603",
        table: "II",
        equation: "(5)",
        m0_a2_per_ps: number("M0_A2_per_ps")?,
        m0_unit: "Angstrom^2/ps",
        qm_ev: number("Qm_eV")?,
        qm_unit: "eV",
        qm_sigma_ev: number("Qm_sigma_eV")?,
        ln_m0_sigma: number("ln_M0_sigma")?,
        coordination: number("coordination")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_packaged_row_is_table_ii_with_its_units() {
        let row = published_copper_mobility().expect("row");
        assert_eq!(row.doi, "10.1103/PhysRevMaterials.5.043603");
        assert_eq!(row.table, "II");
        assert_eq!(row.equation, "(5)");
        assert_eq!(row.m0_unit, "Angstrom^2/ps");
        assert_eq!(row.qm_unit, "eV");
        assert_eq!(row.coordination, 12.0);
        let thermal = BOLTZMANN_EV_PER_K * 1000.0;
        let expected = row.m0_a2_per_ps * (-row.qm_ev / thermal).exp();
        let got = published_mobility_a2_per_ps(1000.0).expect("column");
        assert!((got - expected).abs() <= 1.0e-12 * expected);
        assert!(published_mobility_a2_per_ps(0.0).is_err());
    }

    #[test]
    fn the_definition_inverts_a_supplied_lattice_and_is_not_a_run() {
        let row = published_copper_mobility().expect("row");
        let lattice = 4.0;
        let mobility = published_mobility_a2_per_ps(1000.0).expect("column");
        let rate = row.coordination * mobility / (lattice * lattice) * 1.0e12;
        let back = mobility_a2_per_ps(rate, lattice, row.coordination).expect("mobility");
        assert!((back - mobility).abs() <= 1.0e-9 * mobility);
        assert!(mobility_a2_per_ps(-1.0, lattice, row.coordination).is_none());
        assert!(mobility_a2_per_ps(rate, 0.0, row.coordination).is_none());
    }

    #[test]
    fn the_driver_and_the_comparison_refuse_without_a_stored_lattice() {
        let driver = run_npt_lattice(1000.0).expect_err("driver");
        assert_eq!(driver, FluxError::InvalidConfig(CAMPAIGN_UNAVAILABLE));
        let comparison = cu_vacancy_comparison(1000.0, 1.0e6).expect_err("comparison");
        assert_eq!(comparison, driver);
        assert!(run_npt_lattice(0.0).is_err());
        assert!(run_npt_lattice(f64::NAN).is_err());
        let source = include_str!("cu_reference.rs");
        let zero_kelvin_lattice = ["3.", "615"].concat();
        assert!(
            !source.contains(&zero_kelvin_lattice),
            "the copper path has no zero-kelvin lattice"
        );
        let stored_table = ["cu_vacancy", "_lattice"].concat();
        assert!(!source.contains(&stored_table));
    }
}
