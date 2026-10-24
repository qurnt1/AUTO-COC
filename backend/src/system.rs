use std::{io, path::Path};

const COC_LAUNCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
const COC_LAUNCH_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

const GOOGLE_PLAY_GAMES_COC_URI: &str =
    "googleplaygames://launch/?id=com.supercell.clashofclans&lid=1&pid=1";

pub fn launch_configured_path(path: &str) -> io::Result<()> {
    launch_target(validate_configured_target(path)?)
}

pub fn reload_configured_path(path: &str) -> io::Result<()> {
    let target = validate_configured_target(path)?;
    reload_coc_with(
        crate::native_input::coc_windows,
        |window| crate::native_input::close_coc_window(*window),
        crate::native_input::wait_for_coc_windows,
        || launch_target(target),
    )
}

fn validate_configured_target(target: &str) -> io::Result<&str> {
    if target.is_empty() || target.len() > 2048 || target.contains('\0') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid configured target",
        ));
    }
    if target == GOOGLE_PLAY_GAMES_COC_URI {
        return Ok(target);
    }
    let path = Path::new(target);
    if !path.is_absolute() || !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "configured path not found",
        ));
    }
    Ok(target)
}

fn launch_target(target: &str) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::{ffi::OsStr, os::windows::ffi::OsStrExt};
        use windows_sys::Win32::UI::Shell::ShellExecuteW;
        let verb: Vec<u16> = "open".encode_utf16().chain(Some(0)).collect();
        let target: Vec<u16> = OsStr::new(target).encode_wide().chain(Some(0)).collect();
        let result = unsafe {
            ShellExecuteW(
                std::ptr::null_mut(),
                verb.as_ptr(),
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                1,
            )
        };
        if result as isize <= 32 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = target;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Windows launch API unavailable",
        ))
    }
}

fn reload_coc_with<T>(
    find_windows: impl FnOnce() -> io::Result<Vec<T>>,
    mut close_window: impl FnMut(&T) -> io::Result<()>,
    wait_until_closed: impl FnOnce() -> io::Result<()>,
    launch: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    let windows = find_windows()?;
    if !windows.is_empty() {
        for window in &windows {
            close_window(window)?;
        }
        wait_until_closed()?;
    }
    launch()
}

pub fn open_known_folder(path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::UI::Shell::ShellExecuteW;
        let verb: Vec<u16> = "open".encode_utf16().chain(Some(0)).collect();
        let target: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let result = unsafe {
            ShellExecuteW(
                std::ptr::null_mut(),
                verb.as_ptr(),
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                1,
            )
        };
        if result as isize <= 32 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Windows shell API unavailable",
        ))
    }
}

pub fn coc_screenshot_png() -> io::Result<Vec<u8>> {
    #[cfg(windows)]
    {
        let windows = crate::native_input::coc_windows()?;
        let window = unique_coc_window(&windows)?;
        crate::native_input::focus_coc_window(window)?;
        capture_coc_window_png(window)
    }
    #[cfg(not(windows))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Clash of Clans window capture is only available on Windows",
        ))
    }
}

fn unique_coc_window(windows: &[isize]) -> io::Result<isize> {
    match windows {
        [window] => Ok(*window),
        [] => Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Clash of Clans window was not found",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::NotFound,
            "multiple Clash of Clans windows were found",
        )),
    }
}

pub fn wait_for_coc_window() -> io::Result<()> {
    #[cfg(windows)]
    {
        wait_for_coc_window_with(
            crate::native_input::coc_windows,
            || std::thread::sleep(COC_LAUNCH_POLL_INTERVAL),
            std::time::Instant::now() + COC_LAUNCH_TIMEOUT,
        )
    }
    #[cfg(not(windows))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Clash of Clans window detection is only available on Windows",
        ))
    }
}

fn wait_for_coc_window_with<T>(
    mut find_windows: impl FnMut() -> io::Result<Vec<T>>,
    mut wait: impl FnMut(),
    deadline: std::time::Instant,
) -> io::Result<()> {
    loop {
        if !find_windows()?.is_empty() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Clash of Clans window did not appear after launch",
            ));
        }
        wait();
    }
}

#[cfg(windows)]
fn capture_coc_window_png(window: isize) -> io::Result<Vec<u8>> {
    use std::mem::size_of;
    use windows_sys::Win32::{
        Foundation::{HWND, POINT, RECT},
        Graphics::Gdi::{
            BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, ClientToScreen, CreateCompatibleBitmap,
            CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits,
            ReleaseDC, SRCCOPY, SelectObject,
        },
        UI::WindowsAndMessaging::{GetClientRect, GetForegroundWindow},
    };

    let window = window as HWND;
    if unsafe { GetForegroundWindow() } != window {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Clash of Clans lost foreground focus before capture",
        ));
    }
    let mut client: RECT = unsafe { std::mem::zeroed() };
    if unsafe { GetClientRect(window, &mut client) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut origin = POINT { x: 0, y: 0 };
    if unsafe { ClientToScreen(window, &mut origin) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let width = client.right - client.left;
    let height = client.bottom - client.top;
    if width <= 0
        || height <= 0
        || width > 32_768
        || height > 32_768
        || (width as u64).saturating_mul(height as u64) > 100_000_000
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Clash of Clans window dimensions",
        ));
    }

    let screen_dc = unsafe { GetDC(std::ptr::null_mut()) };
    if screen_dc.is_null() {
        return Err(io::Error::last_os_error());
    }
    let memory_dc = unsafe { CreateCompatibleDC(screen_dc) };
    if memory_dc.is_null() {
        unsafe { ReleaseDC(std::ptr::null_mut(), screen_dc) };
        return Err(io::Error::last_os_error());
    }
    let bitmap = unsafe { CreateCompatibleBitmap(screen_dc, width, height) };
    if bitmap.is_null() {
        unsafe {
            DeleteDC(memory_dc);
            ReleaseDC(std::ptr::null_mut(), screen_dc);
        }
        return Err(io::Error::last_os_error());
    }
    let previous = unsafe { SelectObject(memory_dc, bitmap.cast()) };
    if previous.is_null() || previous as isize == -1 {
        unsafe {
            DeleteObject(bitmap.cast());
            DeleteDC(memory_dc);
            ReleaseDC(std::ptr::null_mut(), screen_dc);
        }
        return Err(io::Error::last_os_error());
    }
    let copied = unsafe {
        BitBlt(
            memory_dc, 0, 0, width, height, screen_dc, origin.x, origin.y, SRCCOPY,
        )
    };
    let foreground_is_still_coc = unsafe { GetForegroundWindow() } == window;
    if copied == 0 || !foreground_is_still_coc {
        unsafe {
            SelectObject(memory_dc, previous);
            DeleteObject(bitmap.cast());
            DeleteDC(memory_dc);
            ReleaseDC(std::ptr::null_mut(), screen_dc);
        }
        return Err(if !foreground_is_still_coc {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Clash of Clans lost foreground focus during capture",
            )
        } else {
            io::Error::last_os_error()
        });
    }

    let mut info: BITMAPINFO = unsafe { std::mem::zeroed() };
    info.bmiHeader = BITMAPINFOHEADER {
        biSize: size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: width,
        biHeight: -height,
        biPlanes: 1,
        biBitCount: 32,
        biCompression: BI_RGB,
        biSizeImage: 0,
        biXPelsPerMeter: 0,
        biYPelsPerMeter: 0,
        biClrUsed: 0,
        biClrImportant: 0,
    };
    let mut bgra = vec![0u8; width as usize * height as usize * 4];
    unsafe { SelectObject(memory_dc, previous) };
    let lines = unsafe {
        GetDIBits(
            memory_dc,
            bitmap,
            0,
            height as u32,
            bgra.as_mut_ptr().cast(),
            &mut info,
            DIB_RGB_COLORS,
        )
    };
    unsafe {
        DeleteObject(bitmap.cast());
        DeleteDC(memory_dc);
        ReleaseDC(std::ptr::null_mut(), screen_dc);
    }
    if lines != height {
        return Err(io::Error::last_os_error());
    }
    if !crate::native_input::coc_windows()?.contains(&(window as isize)) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Clash of Clans window changed during capture",
        ));
    }
    encode_rgb_png(width as u32, height as u32, &bgrx_to_rgb(&bgra))
}

fn bgrx_to_rgb(pixels: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(pixels.len() / 4 * 3);
    for pixel in pixels.chunks_exact(4) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    rgb
}

fn encode_rgb_png(width: u32, height: u32, pixels: &[u8]) -> io::Result<Vec<u8>> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|size| size.checked_mul(3))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "PNG dimensions overflow"))?;
    if width == 0 || height == 0 || pixels.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "RGB pixel buffer has the wrong size",
        ));
    }
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(io::Error::other)?;
    writer.write_image_data(pixels).map_err(io::Error::other)?;
    drop(writer);
    Ok(output)
}

#[cfg(windows)]
pub fn request_shutdown() -> io::Result<()> {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn ExitWindowsEx(flags: u32, reason: u32) -> i32;
    }
    const EWX_SHUTDOWN: u32 = 0x0000_0001;
    const EWX_FORCEIFHUNG: u32 = 0x0000_0010;
    const SHTDN_REASON_MAJOR_APPLICATION: u32 = 0x0004_0000;
    const SHTDN_REASON_FLAG_PLANNED: u32 = 0x8000_0000;
    let ok = unsafe {
        ExitWindowsEx(
            EWX_SHUTDOWN | EWX_FORCEIFHUNG,
            SHTDN_REASON_MAJOR_APPLICATION | SHTDN_REASON_FLAG_PLANNED,
        )
    };
    if ok == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, io::Cursor};

    #[test]
    fn accepts_only_the_exact_google_play_games_coc_uri() {
        assert_eq!(
            validate_configured_target(GOOGLE_PLAY_GAMES_COC_URI).unwrap(),
            GOOGLE_PLAY_GAMES_COC_URI
        );

        for variant in [
            "GooglePlayGames://launch/?id=com.supercell.clashofclans&lid=1&pid=1",
            "googleplaygames://launch/?id=com.supercell.clashofclans&lid=1&pid=2",
            "googleplaygames://launch/?id=com.supercell.clashofclans&lid=1&pid=1/",
            "googleplaygames://launch/?id=com.supercell.clashofclans&pid=1&lid=1",
            "googleplaygames://launch/?id=com.supercell.clashofclans&lid=1&pid=1&extra=1",
            "https://googleplaygames/launch/?id=com.supercell.clashofclans&lid=1&pid=1",
            "file:///googleplaygames://launch/?id=com.supercell.clashofclans&lid=1&pid=1",
            "steam://launch/?id=com.supercell.clashofclans&lid=1&pid=1",
        ] {
            assert!(
                validate_configured_target(variant).is_err(),
                "accepted {variant}"
            );
        }
    }

    #[test]
    fn configured_file_target_must_be_absolute_and_exist() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("Clash of Clans.lnk");
        std::fs::write(&target, b"shortcut").unwrap();
        let target = target.to_str().unwrap();

        assert_eq!(validate_configured_target(target).unwrap(), target);
        assert!(validate_configured_target("Clash of Clans.lnk").is_err());
    }

    #[test]
    fn reload_closes_all_detected_windows_waits_then_launches() {
        let calls = RefCell::new(Vec::new());
        reload_coc_with(
            || {
                calls.borrow_mut().push("find");
                Ok(vec!["first", "second"])
            },
            |window| {
                calls.borrow_mut().push(*window);
                Ok(())
            },
            || {
                calls.borrow_mut().push("wait");
                Ok(())
            },
            || {
                calls.borrow_mut().push("launch");
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            *calls.borrow(),
            ["find", "first", "second", "wait", "launch"]
        );
    }

    #[test]
    fn reload_launches_directly_when_no_coc_window_is_found() {
        let calls = RefCell::new(Vec::new());
        reload_coc_with(
            || {
                calls.borrow_mut().push("find");
                Ok::<_, io::Error>(Vec::<()>::new())
            },
            |_| {
                calls.borrow_mut().push("close");
                Ok(())
            },
            || {
                calls.borrow_mut().push("wait");
                Ok(())
            },
            || {
                calls.borrow_mut().push("launch");
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(*calls.borrow(), ["find", "launch"]);
    }

    #[test]
    fn reload_does_not_launch_when_window_detection_fails() {
        let calls = RefCell::new(Vec::new());
        let result = reload_coc_with(
            || {
                calls.borrow_mut().push("find");
                Err::<Vec<()>, _>(io::Error::other("window identity unavailable"))
            },
            |_| {
                calls.borrow_mut().push("close");
                Ok(())
            },
            || {
                calls.borrow_mut().push("wait");
                Ok(())
            },
            || {
                calls.borrow_mut().push("launch");
                Ok(())
            },
        );

        assert!(result.is_err());
        assert_eq!(*calls.borrow(), ["find"]);
    }

    #[test]
    fn reload_does_not_launch_when_close_or_wait_fails() {
        let calls = RefCell::new(Vec::new());
        let close_result = reload_coc_with(
            || Ok(vec![()]),
            |_| {
                calls.borrow_mut().push("close");
                Err(io::Error::other("close failed"))
            },
            || {
                calls.borrow_mut().push("wait");
                Ok(())
            },
            || {
                calls.borrow_mut().push("launch");
                Ok(())
            },
        );
        assert!(close_result.is_err());
        assert_eq!(*calls.borrow(), ["close"]);

        calls.borrow_mut().clear();
        let wait_result = reload_coc_with(
            || Ok(vec![()]),
            |_| {
                calls.borrow_mut().push("close");
                Ok(())
            },
            || {
                calls.borrow_mut().push("wait");
                Err(io::Error::new(io::ErrorKind::TimedOut, "still open"))
            },
            || {
                calls.borrow_mut().push("launch");
                Ok(())
            },
        );
        assert!(wait_result.is_err());
        assert_eq!(*calls.borrow(), ["close", "wait"]);
    }

    #[test]
    fn coc_launch_wait_returns_when_a_window_appears() {
        let mut checks = 0;
        wait_for_coc_window_with(
            || {
                checks += 1;
                Ok(if checks == 1 {
                    Vec::<u8>::new()
                } else {
                    vec![1]
                })
            },
            || {},
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(checks, 2);
    }

    #[test]
    fn coc_launch_wait_times_out_and_propagates_detection_errors() {
        let deadline = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let timeout =
            wait_for_coc_window_with(|| Ok(Vec::<u8>::new()), || {}, deadline).unwrap_err();
        assert_eq!(timeout.kind(), io::ErrorKind::TimedOut);

        let detection = wait_for_coc_window_with::<u8>(
            || Err(io::Error::other("window detection failed")),
            || {},
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .unwrap_err();
        assert_eq!(detection.to_string(), "window detection failed");
    }

    #[test]
    fn coc_capture_requires_one_identified_game_window() {
        assert_eq!(unique_coc_window(&[42]).unwrap(), 42);
        assert_eq!(
            unique_coc_window(&[]).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            unique_coc_window(&[42, 84]).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    fn encodes_rgb_pixels_as_a_valid_png() {
        let pixels = [255, 0, 0, 0, 255, 0];
        let encoded = encode_rgb_png(2, 1, &pixels).unwrap();
        let decoder = png::Decoder::new(Cursor::new(encoded));
        let mut reader = decoder.read_info().unwrap();
        let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut decoded).unwrap();
        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(&decoded[..frame.buffer_size()], &pixels);
    }

    #[test]
    fn rejects_mismatched_rgb_buffer_length() {
        assert!(encode_rgb_png(1, 1, &[0, 0]).is_err());
    }
}
