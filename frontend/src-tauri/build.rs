const DESKTOP_COMMANDS: &[&str] = &[
    "get_snapshot",
    "get_macro",
    "create_macro",
    "rename_macro",
    "delete_macro",
    "select_macro",
    "start_recording",
    "stop_recording",
    "start_playback",
    "stop_playback",
    "patch_settings",
    "put_shortcuts",
    "launch_coc",
    "screenshot",
    "open_macros_folder",
    "quit_application",
    "telegram_config",
    "telegram_config_clear",
    "pairing_start",
    "pairing_cancel",
    "telegram_test",
    "get_migration",
    "select_migration_source",
    "import_migration",
    "shutdown_prepare",
    "shutdown_confirm",
    "get_help",
    "get_diagnostics",
    "complete_onboarding",
];

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(DESKTOP_COMMANDS)),
    )
    .expect("failed to configure Tauri command permissions");
}
