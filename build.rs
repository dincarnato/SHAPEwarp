use semver::Version;

fn main() {
    let vrna = pkg_config::Config::new()
        .range_version("2.4.18".."2.8")
        .cargo_metadata(false)
        .env_metadata(false)
        .print_system_libs(false)
        .print_system_cflags(false)
        .probe("RNAlib2")
        .unwrap();

    println!("cargo:rerun-if-changed=build.rs");

    let version: Version = vrna
        .version
        .parse()
        .expect("unable to parse ViennaRNA version");

    let version_cfg = format!("vrna{}{}", version.major, version.minor);
    println!("cargo:rustc-cfg={version_cfg}");

    if version.major == 2 && version.minor == 5 {
        let version_cfg = format!("vrna{}{}{}", version.major, version.minor, version.patch);
        println!("cargo:rustc-cfg={version_cfg}");
    }
}
