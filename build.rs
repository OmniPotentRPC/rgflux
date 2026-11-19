fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/engine/eam_table_bridge.cc");
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

fn find_rgpot_cpp_core() -> Option<std::path::PathBuf> {
    let cargo_home = std::env::var_os("CARGO_HOME").map(std::path::PathBuf::from)?;
    let checkouts = cargo_home.join("git").join("checkouts");
    for entry in std::fs::read_dir(checkouts).ok()?.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("rgpot-") {
            continue;
        }
        for rev_entry in std::fs::read_dir(entry.path()).ok()?.flatten() {
            let cpp_core = rev_entry.path().join("CppCore");
            if cpp_core
                .join("rgpot")
                .join("EAMTable")
                .join("EAMTablePot.cc")
                .is_file()
            {
                return Some(cpp_core);
            }
        }
    }
    None
}

fn compile_rgpot_eam_table_bridge() -> Result<(), Box<dyn std::error::Error>> {
    let Some(cpp_core) = find_rgpot_cpp_core() else {
        return Err("could not locate rgpot CppCore in CARGO_HOME/git/checkouts".into());
    };
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let sources = [
        std::path::PathBuf::from("src/engine/eam_table_bridge.cc"),
        cpp_core.join("rgpot/EAMTable/EAMTablePot.cc"),
        cpp_core.join("rgpot/nlist/Neighbours.cc"),
        cpp_core.join("rgpot/nlist/MinimumImage.cpp"),
        cpp_core.join("rgpot/simd/exact_fma_hw.cc"),
        cpp_core.join("rgpot/simd/exp.cc"),
        cpp_core.join("rgpot/Slices.cc"),
        cpp_core.join("rgpot/PotHelpers.cc"),
    ];
    let mut objects = Vec::with_capacity(sources.len());
    for (idx, src) in sources.iter().enumerate() {
        let obj = out_dir.join(format!("rgpot_eam_{idx}.o"));
        let status = std::process::Command::new("g++")
            .arg("-std=c++20")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-c")
            .arg(src)
            .arg("-I")
            .arg(&cpp_core)
            .arg("-o")
            .arg(&obj)
            .status()?;
        if !status.success() {
            return Err(format!("g++ failed on {}", src.display()).into());
        }
        objects.push(obj);
    }
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
    if let Ok(output) = std::process::Command::new("g++")
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
