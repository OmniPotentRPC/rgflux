fn main() {
    println!("cargo:rerun-if-env-changed=LAMMPC_DIR");
    if std::env::var_os("CARGO_FEATURE_LAMMPC").is_none() {
        return;
    }

    let pkg_probe = pkg_config::Config::new()
        .cargo_metadata(true)
        .probe("lammpc");
    if pkg_probe.is_ok() {
        return;
    }

    let lammpc_dir = match std::env::var("LAMMPC_DIR") {
        Ok(value) if !value.trim().is_empty() => value,
        Ok(_) => {
            panic!("LAMMPC_DIR is set but empty; set it to a lammpc install prefix");
        }
        Err(_) => {
            panic!("lammpc feature requires either pkg-config package 'lammpc' or LAMMPC_DIR");
        }
    };

    println!("cargo:rustc-link-search=native={lammpc_dir}/lib");
    println!("cargo:rustc-link-search=native={lammpc_dir}/lib64");
    println!("cargo:rustc-link-lib=lammpc");
}
