fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/engine/eam_table_bridge.cc");
    println!("cargo:rerun-if-env-changed=RGPOT_DIR");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=LAMMPC_DIR");
    println!("cargo:rerun-if-env-changed=LAMMPC_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR");
    if std::env::var_os("CARGO_FEATURE_RGPOT").is_some() {
        compile_rgpot_eam_table_bridge()?;
    }
    if std::env::var_os("CARGO_FEATURE_LAMMPC").is_none() {
        return Ok(());
    }

    if let Some(lib_dir) = std::env::var_os("LAMMPC_LIB_DIR") {
        let lib_dir = lib_dir.to_string_lossy();
        if lib_dir.trim().is_empty() {
            return Err("LAMMPC_LIB_DIR is set but empty; set it to a liblammpc directory".into());
        }
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=lammpc");
        return Ok(());
    }

    let pkg_probe = pkg_config::Config::new()
        .cargo_metadata(true)
        .probe("lammpc");
    if pkg_probe.is_ok() {
        return Ok(());
    }

    let lammpc_dir = match std::env::var("LAMMPC_DIR") {
        Ok(value) if !value.trim().is_empty() => value,
        Ok(_) => {
            return Err("LAMMPC_DIR is set but empty; set it to a lammpc install prefix".into());
        }
        Err(_) => {
            return Err(
                "lammpc feature requires pkg-config package 'lammpc', LAMMPC_LIB_DIR, or LAMMPC_DIR"
                    .into(),
            );
        }
    };

    let prefix = std::path::Path::new(&lammpc_dir);
    for lib_dir in [prefix.join("lib"), prefix.join("lib64")] {
        emit_link_search(&lib_dir);
    }
    let multiarch_parent = prefix.join("lib");
    if let Ok(entries) = std::fs::read_dir(multiarch_parent) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().ends_with("-linux-gnu"))
            {
                emit_link_search(&path);
            }
        }
    }
    println!("cargo:rustc-link-lib=lammpc");
    Ok(())
}

fn emit_link_search(path: &std::path::Path) {
    if path.is_dir() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn eam_marker() -> std::path::PathBuf {
    std::path::PathBuf::from("rgpot")
        .join("EAMTable")
        .join("EAMTablePot.cc")
}

fn cpp_core_at(path: &std::path::Path) -> Option<std::path::PathBuf> {
    let marker = eam_marker();
    if path.join(&marker).is_file() {
        return Some(path.to_path_buf());
    }
    let nested = path.join("CppCore");
    if nested.join(&marker).is_file() {
        return Some(nested);
    }
    None
}

/// CppCore revision ci/fetch_rgpot_cpp.sh checks out. A cargo git checkout
/// is used only when its HEAD is this revision.
const RGPOT_CPP_PIN: &str = "424629e5ba9c0948b8d6224f5b49c014ffb4ac7c";

fn checkout_is_pin(path: &std::path::Path) -> bool {
    let Ok(output) = std::process::Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let rev = String::from_utf8_lossy(&output.stdout);
    let rev = rev.trim();
    rev == RGPOT_CPP_PIN
}

/// Checkout written by ci/fetch_rgpot_cpp.sh. Used only when it is already
/// at RGPOT_CPP_PIN. The build does not fetch one itself.
fn script_checkout() -> Option<std::path::PathBuf> {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")?;
    let root = std::path::PathBuf::from(manifest)
        .join("target")
        .join("rgpot-cpp");
    println!("cargo:rerun-if-changed={}", root.display());
    if !checkout_is_pin(&root) {
        return None;
    }
    cpp_core_at(&root)
}

// Three sentences, the same wording as the contributor notes.
fn missing_rgpot_core() -> String {
    "The rgpot feature compiles rgpot's EAMTablePot for RgpotSurface::from_eam_table_file and from_eam_table_memory. Set RGPOT_DIR to the rgpot repository root or its CppCore directory, or run ci/fetch_rgpot_cpp.sh, which fetches the pin. When RGPOT_DIR is unset, the build uses that fetched tree when it is already at the pin, and otherwise reuses an existing cargo git checkout of the pin.".into()
}

fn find_rgpot_cpp_core() -> Result<std::path::PathBuf, String> {
    if let Some(dir) = std::env::var_os("RGPOT_DIR") {
        let path = std::path::PathBuf::from(&dir);
        if path.as_os_str().is_empty() {
            return Err(
                "RGPOT_DIR is set but empty; set it to an rgpot tree or its CppCore directory"
                    .into(),
            );
        }
        return cpp_core_at(&path).ok_or_else(|| {
            format!(
                "RGPOT_DIR {} does not contain rgpot/EAMTable/EAMTablePot.cc",
                path.display()
            )
        });
    }
    if let Some(cpp_core) = script_checkout() {
        return Ok(cpp_core);
    }
    let Some(cargo_home) = std::env::var_os("CARGO_HOME") else {
        return Err(missing_rgpot_core());
    };
    let checkouts = std::path::PathBuf::from(cargo_home)
        .join("git")
        .join("checkouts");
    if let Ok(entries) = std::fs::read_dir(&checkouts) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if !name.to_string_lossy().starts_with("rgpot-") {
                continue;
            }
            let Ok(revs) = std::fs::read_dir(entry.path()) else {
                continue;
            };
            for rev_entry in revs.flatten() {
                if !checkout_is_pin(&rev_entry.path()) {
                    continue;
                }
                if let Some(cpp_core) = cpp_core_at(&rev_entry.path()) {
                    return Ok(cpp_core);
                }
            }
        }
    }
    Err(missing_rgpot_core())
}

fn contraction_off(src: &std::path::Path) -> bool {
    matches!(
        src.file_name().and_then(|name| name.to_str()),
        Some("MinimumImage.cpp" | "exp.cc" | "exact_fma_hw.cc" | "PotHelpers.cc" | "Slices.cc")
    )
}

fn nm_lines(obj: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = std::process::Command::new("nm")
        .arg("-g")
        .arg(obj)
        .output()?;
    if !output.status.success() {
        return Err(format!("nm failed on {}", obj.display()).into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn symbol_name(line: &str) -> Option<&str> {
    line.split_whitespace().last()
}

/// exp_fixed_n is an ifunc. When no other object references it, keep the
/// symbol inside exp.cc.
fn localize_exp_fixed_n(
    sources: &[std::path::PathBuf],
    objects: &[std::path::PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(exp_index) = sources
        .iter()
        .position(|src| src.file_name().and_then(|name| name.to_str()) == Some("exp.cc"))
    else {
        return Ok(());
    };
    for (idx, obj) in objects.iter().enumerate() {
        if idx == exp_index {
            continue;
        }
        for line in nm_lines(obj)?.lines() {
            let mut parts = line.split_whitespace();
            let kind = parts.next();
            let name = parts.next();
            let (kind, name) = match (kind, name, parts.next()) {
                (Some(kind), Some(name), None) if kind.len() == 1 => (kind, name),
                (Some(_addr), Some(kind), Some(name)) if kind.len() == 1 => (kind, name),
                _ => continue,
            };
            if kind == "U" && name.contains("exp_fixed_n") {
                return Ok(());
            }
        }
    }
    let exp_obj = &objects[exp_index];
    let mut names = Vec::new();
    for line in nm_lines(exp_obj)?.lines() {
        if let Some(name) = symbol_name(line)
            && name.contains("exp_fixed_n")
        {
            names.push(name.to_string());
        }
    }
    if names.is_empty() {
        return Err(format!(
            "exp.cc object {} has no exp_fixed_n symbol",
            exp_obj.display()
        )
        .into());
    }
    let mut cmd = std::process::Command::new("objcopy");
    for name in &names {
        cmd.arg(format!("--localize-symbol={name}"));
    }
    let status = cmd.arg(exp_obj).status()?;
    if !status.success() {
        return Err(format!("objcopy failed on {}", exp_obj.display()).into());
    }
    Ok(())
}

fn cxx_compiler() -> String {
    match std::env::var("CXX") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => "g++".to_string(),
    }
}

/// Repository include tree that holds rgpot/plugin.h, next to CppCore.
fn plugin_include(cpp_core: &std::path::Path) -> Option<std::path::PathBuf> {
    let include = cpp_core.parent()?.join("include");
    if include.join("rgpot").join("plugin.h").is_file() {
        Some(include)
    } else {
        None
    }
}

fn compile_rgpot_eam_table_bridge() -> Result<(), Box<dyn std::error::Error>> {
    let cpp_core = find_rgpot_cpp_core()?;
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let cxx = cxx_compiler();
    let mut sources = vec![
        std::path::PathBuf::from("src/engine/eam_table_bridge.cc"),
        cpp_core.join("rgpot/EAMTable/EAMTablePot.cc"),
        cpp_core.join("rgpot/nlist/Neighbours.cc"),
        cpp_core.join("rgpot/nlist/MinimumImage.cpp"),
        cpp_core.join("rgpot/simd/exact_fma_hw.cc"),
        cpp_core.join("rgpot/simd/exp.cc"),
        cpp_core.join("rgpot/Slices.cc"),
        cpp_core.join("rgpot/PotHelpers.cc"),
    ];
    let images = cpp_core.join("rgpot/EAMTable/EAMTableImages.cc");
    println!("cargo:rustc-check-cfg=cfg(rgpot_eam_images)");
    if images.is_file() {
        sources.push(images);
        println!("cargo:rustc-cfg=rgpot_eam_images");
    }
    let plugin = plugin_include(&cpp_core);
    if let Some(include) = &plugin {
        println!("cargo:rerun-if-changed={}", include.display());
    }
    let mut objects = Vec::with_capacity(sources.len());
    for (idx, src) in sources.iter().enumerate() {
        let obj = out_dir.join(format!("rgpot_eam_{idx}.o"));
        println!("cargo:rerun-if-changed={}", src.display());
        let mut command = std::process::Command::new(&cxx);
        command
            .arg("-std=c++20")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-fvisibility=hidden");
        if contraction_off(src) {
            command.arg("-ffp-contract=off");
        }
        command.arg("-c").arg(src).arg("-I").arg(&cpp_core);
        if let Some(include) = &plugin {
            command.arg("-I").arg(include);
        }
        let status = command.arg("-o").arg(&obj).status()?;
        if !status.success() {
            return Err(format!("{cxx} failed on {}", src.display()).into());
        }
        objects.push(obj);
    }
    localize_exp_fixed_n(&sources, &objects)?;
    let archive = out_dir.join("librgflux_eam_table.a");
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.arg("crus").arg(&archive);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status()?;
    if !status.success() {
        return Err("ar failed building librgflux_eam_table.a".into());
    }
    if let Ok(output) = std::process::Command::new(&cxx)
        .arg("-print-file-name=libstdc++.so")
        .output()
        && output.status.success()
    {
        let path_str = String::from_utf8_lossy(&output.stdout);
        let path = std::path::Path::new(path_str.trim());
        if let Some(parent) = path.parent() {
            emit_link_search(parent);
        }
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=rgflux_eam_table");
    println!("cargo:rustc-link-lib=stdc++");
    Ok(())
}
