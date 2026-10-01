use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let source = root.join("../frontend/dist");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory")).join("web-dist");
    println!("cargo:rerun-if-changed={}", source.display());
    if out.exists() {
        fs::remove_dir_all(&out).expect("clean embedded assets");
    }
    fs::create_dir_all(&out).expect("create embedded asset directory");

    if source.is_dir() {
        copy_tree(&source, &out).expect("copy frontend build");
    } else if env::var("PROFILE").as_deref() == Ok("debug") {
        fs::write(out.join("index.html"), "<!doctype html><html lang=\"fr\"><body>Build React frontend before packaging AUTO-COC.</body></html>")
            .expect("write frontend build notice");
    } else {
        panic!("frontend/dist is required for a release build");
    }
}

fn copy_tree(source: &Path, target: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let destination = target.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            fs::create_dir_all(&destination)?;
            copy_tree(&entry.path(), &destination)?;
        } else {
            fs::copy(entry.path(), destination)?;
        }
    }
    Ok(())
}
