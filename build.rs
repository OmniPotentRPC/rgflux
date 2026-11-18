fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LAMMPC_DIR");
    println!("cargo:rerun-if-env-changed=LAMMPC_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR");
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
