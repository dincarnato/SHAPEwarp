use std::{env, path::PathBuf};

fn main() {
    let vrna = pkg_config::Config::new()
        .atleast_version("2.4.18")
        .probe("RNAlib2")
        .unwrap();

    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(
            vrna.include_paths
                .into_iter()
                .map(|path| format!("-I{}", path.display())),
        )
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("vrna_mfe")
        .allowlist_function("vrna_mfe_dimer")
        .allowlist_function("vrna_fold")
        .allowlist_function("vrna_circfold")
        .allowlist_function("vrna_alifold")
        .allowlist_function("vrna_circalifold")
        .allowlist_function("vrna_cofold")
        .allowlist_function("vrna_fold_compound_.*")
        .allowlist_function("vrna_md_set_default")
        .allowlist_function("vrna_sc_init")
        .allowlist_function("vrna_sc_set_stack_comparative")
        .allowlist_function("vrna_sc_add_SHAPE_deigan_ali")
        .allowlist_function("vrna_mfe_window_cb")
        .allowlist_var("VRNA_OPTION.*")
        .allowlist_type("vrna_sc_s")
        .allowlist_type("vrna_sc_bp_storage_t")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
