use std::path::Path;
use std::process::{Command, Output};
use std::sync::OnceLock;

use serde_json::Value;

const REACTANT_PARTITION_ERROR: &str =
    "invalid configuration: PAFI reactant partition must be finite and positive";

const PYTHON_VARIABLE: &str = "PAFI_PYTHON";
const PYTHON_ON_PATH: [&str; 2] = ["python3", "python"];
const MINIMUM_PYTHON: (u32, u32) = (3, 10);

fn probe_python_version(interpreter: &str) -> Result<(u32, u32), String> {
    let output = Command::new(interpreter)
        .args(["-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
        .output()
        .map_err(|error| format!("`{interpreter}` could not be run: {error}"))?;
    if !output.status.success() {
        return Err(format!("`{interpreter}` exited with {}", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_python_version(&text)
        .ok_or_else(|| format!("`{interpreter}` printed an unreadable version: {text:?}"))
}

fn parse_python_version(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn resolve_python(
    configured: Option<&str>,
    probe: impl Fn(&str) -> Result<(u32, u32), String>,
) -> Result<String, String> {
    let requirement = format!(
        "Python {}.{} or newer is required by bench/pafi/analyze.py; \
         set {PYTHON_VARIABLE} to such an interpreter",
        MINIMUM_PYTHON.0, MINIMUM_PYTHON.1
    );
    let (interpreter, version) = match configured.filter(|value| !value.is_empty()) {
        Some(interpreter) => {
            let version = probe(interpreter)
                .map_err(|error| format!("{PYTHON_VARIABLE} is set but {error}. {requirement}"))?;
            (interpreter.to_string(), version)
        }
        None => {
            let mut failures = Vec::new();
            let mut found = None;
            for candidate in PYTHON_ON_PATH {
                match probe(candidate) {
                    Ok(version) => {
                        found = Some((candidate.to_string(), version));
                        break;
                    }
                    Err(error) => failures.push(error),
                }
            }
            found.ok_or_else(|| {
                format!(
                    "{PYTHON_VARIABLE} is unset and no interpreter on PATH ran ({}). {requirement}",
                    failures.join("; ")
                )
            })?
        }
    };
    if version < MINIMUM_PYTHON {
        return Err(format!(
            "`{interpreter}` is Python {}.{}. {requirement}",
            version.0, version.1
        ));
    }
    Ok(interpreter)
}

fn python() -> &'static str {
    static INTERPRETER: OnceLock<Result<String, String>> = OnceLock::new();
    INTERPRETER
        .get_or_init(|| {
            resolve_python(
                std::env::var(PYTHON_VARIABLE).ok().as_deref(),
                probe_python_version,
            )
        })
        .as_deref()
        .unwrap_or_else(|error| panic!("{error}"))
}

fn run_python(arguments: &[&str]) -> Output {
    Command::new(python())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()
        .expect("the Python interpreter must run the PAFI coverage gate")
}

fn unittest_summary(stderr: &str) -> Result<usize, String> {
    let lines: Vec<&str> = stderr.lines().collect();
    let ran = lines
        .iter()
        .rposition(|line| line.starts_with("Ran "))
        .ok_or_else(|| format!("no unittest `Ran` summary line in:\n{stderr}"))?;
    let tests = lines[ran]
        .split_whitespace()
        .nth(1)
        .and_then(|count| count.parse::<usize>().ok())
        .ok_or_else(|| format!("unreadable unittest summary line: {:?}", lines[ran]))?;
    let status = lines[ran + 1..]
        .iter()
        .map(|line| line.trim())
        .find(|line| !line.is_empty())
        .ok_or_else(|| format!("no unittest result after {:?}", lines[ran]))?;
    if status == "OK" || status.starts_with("OK (") {
        Ok(tests)
    } else {
        Err(format!("unittest reported {status:?} after {tests} tests"))
    }
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
        "reactant partition rejection guard",
    );
    let record: Value = serde_json::from_str(&stdout).expect("repro must print the record");
    assert_eq!(record["status"], "rejected", "{record}");
    assert_eq!(record["error"], REACTANT_PARTITION_ERROR, "{record}");
}

#[cfg(feature = "pafi-python")]
const ANALYZER_TEST_ARGS: &[&str] = &["bench/pafi/test_analyze.py"];

#[cfg(not(feature = "pafi-python"))]
const ANALYZER_TEST_ARGS: &[&str] = &[
    "bench/pafi/test_analyze.py",
    "AnalysisFailureTests",
    "VerdictProseTests",
    "RawArchiveTests",
    "RejectionClassificationTests",
    "StrictPairingTests",
];

#[test]
fn analyzer_negative_tests_pass() {
    let (_, stderr) = assert_success(run_python(ANALYZER_TEST_ARGS), "analyzer negative tests");
    let tests = unittest_summary(&stderr).unwrap_or_else(|error| panic!("{error}"));
    assert!(tests > 0, "unittest ran no tests:\n{stderr}");
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

#[test]
fn unittest_summary_ignores_warnings_after_the_result() {
    let stderr = "....\n---\nRan 4 tests in 0.1s\n\nOK\nDeprecationWarning: late\n";
    assert_eq!(unittest_summary(stderr), Ok(4));
    let skipped = "Ran 3 tests in 0.1s\n\nOK (skipped=1)\nwarning\n";
    assert_eq!(unittest_summary(skipped), Ok(3));
}

#[test]
fn unittest_summary_rejects_failures_and_missing_summaries() {
    let failed = "Ran 2 tests in 0.1s\n\nFAILED (failures=1)\nOK\n";
    assert!(unittest_summary(failed).is_err());
    assert!(unittest_summary("OK\n").is_err());
    assert!(unittest_summary("Ran 1 tests in 0.1s\n").is_err());
}

#[test]
fn python_version_text_is_parsed() {
    assert_eq!(parse_python_version("3.12\n"), Some((3, 12)));
    assert_eq!(parse_python_version("three"), None);
}

#[test]
fn python_comes_from_the_variable_before_path() {
    let probe = |name: &str| match name {
        "custom" => Ok((3, 11)),
        _ => Ok((3, 9)),
    };
    assert_eq!(
        resolve_python(Some("custom"), probe),
        Ok("custom".to_string())
    );
    let path_only = |name: &str| match name {
        "python3" => Err("`python3` could not be run".to_string()),
        _ => Ok((3, 10)),
    };
    assert_eq!(resolve_python(None, path_only), Ok("python".to_string()));
}

#[test]
fn python_older_than_the_analyzer_needs_is_reported() {
    let message = resolve_python(None, |_| Ok((3, 9))).unwrap_err();
    assert!(message.contains("`python3` is Python 3.9"), "{message}");
    assert!(message.contains("3.10 or newer"), "{message}");
    assert!(message.contains(PYTHON_VARIABLE), "{message}");
}

#[test]
fn missing_python_is_reported() {
    let message =
        resolve_python(None, |name| Err(format!("`{name}` could not be run"))).unwrap_err();
    assert!(message.contains("`python3` could not be run"), "{message}");
    assert!(message.contains("`python` could not be run"), "{message}");
    let configured =
        resolve_python(Some("absent"), |name| Err(format!("`{name}` missing"))).unwrap_err();
    assert!(
        configured.contains("PAFI_PYTHON is set but"),
        "{configured}"
    );
}
