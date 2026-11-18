use std::path::Path;
use std::process::{Command, Output};

fn run_python(arguments: &[&str]) -> Output {
    Command::new("python3")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()
        .expect("python3 must run the PAFI coverage gate")
}

fn assert_success(output: Output, name: &str) {
    assert!(
        output.status.success(),
        "{name} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn zero_width_reactant_partition_repro_passes() {
    let native = env!("CARGO_BIN_EXE_pafi-dimer-coverage");
    assert!(Path::new(native).is_file());
    assert_success(
        run_python(&[
            "bench/pafi/repro_zero_width_reactant_partition.py",
            "--native-bin",
            native,
        ]),
        "zero-width reactant partition repro",
    );
}

#[test]
fn analyzer_negative_tests_pass() {
    assert_success(
        run_python(&["bench/pafi/test_analyze.py"]),
        "analyzer negative tests",
    );
}

#[test]
fn committed_results_pass_coverage_gates() {
    assert_success(
        run_python(&[
            "bench/pafi/analyze.py",
            "--validate-results",
            "bench/pafi/results.json",
        ]),
        "committed-result coverage gates",
    );
}
