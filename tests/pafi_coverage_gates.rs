use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;

const REACTANT_PARTITION_ERROR: &str =
    "invalid configuration: PAFI reactant partition must be finite and positive";

fn run_python(arguments: &[&str]) -> Output {
    Command::new("python3")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()
        .expect("python3 must run the PAFI coverage gate")
}

fn assert_success(output: Output, name: &str) -> (String, String) {
    assert!(
        output.status.success(),
        "{name} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    (
        String::from_utf8(output.stdout).expect("stdout must be UTF-8"),
        String::from_utf8(output.stderr).expect("stderr must be UTF-8"),
    )
}

fn committed_results() -> Value {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/pafi/results.json"),
    )
    .expect("committed PAFI results must be readable");
    serde_json::from_str(&text).expect("committed PAFI results must be JSON")
}

#[test]
fn zero_width_reactant_partition_repro_passes() {
    let native = env!("CARGO_BIN_EXE_pafi-dimer-coverage");
    assert!(Path::new(native).is_file());
    let (stdout, _) = assert_success(
        run_python(&[
            "bench/pafi/repro_zero_width_reactant_partition.py",
            "--native-bin",
            native,
        ]),
        "zero-width reactant partition repro",
    );
    let record: Value = serde_json::from_str(&stdout).expect("repro must print the record");
    assert_eq!(record["status"], "rejected", "{record}");
    assert_eq!(record["error"], REACTANT_PARTITION_ERROR, "{record}");
}

#[test]
fn analyzer_negative_tests_pass() {
    let (_, stderr) = assert_success(
        run_python(&["bench/pafi/test_analyze.py"]),
        "analyzer negative tests",
    );
    assert!(
        stderr.trim_end().ends_with("OK"),
        "unittest did not report OK:\n{stderr}"
    );
}

#[test]
fn committed_results_pass_coverage_gates() {
    let (stdout, _) = assert_success(
        run_python(&[
            "bench/pafi/analyze.py",
            "--validate-results",
            "bench/pafi/results.json",
        ]),
        "committed-result coverage gates",
    );
    let validation: Value =
        serde_json::from_str(&stdout).expect("validator must print its verdict");
    assert_eq!(validation["status"], "passed", "{validation}");
    assert_eq!(validation["nonfinite_samples"], 0, "{validation}");
    assert_eq!(
        validation,
        committed_results()["execution_summary"]["analysis_validation"],
        "validator verdict differs from the committed verdict"
    );

    let markdown = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/pafi/results.md"),
    )
    .expect("committed PAFI prose must be readable");
    let claim = format!(
        "Coverage gates passed ({} checks)",
        validation["gates_checked"]
    );
    assert!(
        markdown.contains(&claim),
        "results.md does not state `{claim}`"
    );
}
