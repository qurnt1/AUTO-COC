use std::{io, path::Path};

pub fn launch_configured_path(path: &str) -> io::Result<()> {
    if path.is_empty() || path.len() > 2048 || path.contains('\0') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid configured path",
        ));
    }
    let path = Path::new(path);
    if !path.is_absolute() || !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "configured path not found",
        ));
    }
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
            "Windows launch API unavailable",
        ))
    }
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

pub fn screenshot_png() -> io::Result<Vec<u8>> {
    #[cfg(windows)]
    {
        use std::mem::size_of;
        use windows_sys::Win32::{
            Graphics::Gdi::{
                BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleBitmap,
                CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits,
                ReleaseDC, SRCCOPY, SelectObject,
            },
            UI::WindowsAndMessaging::{
                GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN,
                SM_YVIRTUALSCREEN,
            },
        };

        let left = unsafe { GetSystemMetrics(SM_XVIRTUALSCREEN) };
        let top = unsafe { GetSystemMetrics(SM_YVIRTUALSCREEN) };
        let width = unsafe { GetSystemMetrics(SM_CXVIRTUALSCREEN) };
        let height = unsafe { GetSystemMetrics(SM_CYVIRTUALSCREEN) };
        if width <= 0
            || height <= 0
            || width > 32_768
            || height > 32_768
            || (width as u64).saturating_mul(height as u64) > 100_000_000
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid desktop dimensions",
            ));
        }

        let screen_dc = unsafe { GetDC(std::ptr::null_mut()) };
        if screen_dc.is_null() {
            return Err(io::Error::last_os_error());
        }
        let memory_dc = unsafe { CreateCompatibleDC(screen_dc) };
        if memory_dc.is_null() {
            unsafe {
                ReleaseDC(std::ptr::null_mut(), screen_dc);
            }
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
                memory_dc,
                0,
                0,
                width,
                height,
                screen_dc,
                left,
                top,
                SRCCOPY | CAPTUREBLT,
            )
        };
        if copied == 0 {
            unsafe {
                SelectObject(memory_dc, previous);
                DeleteObject(bitmap.cast());
                DeleteDC(memory_dc);
                ReleaseDC(std::ptr::null_mut(), screen_dc);
            }
            return Err(io::Error::last_os_error());
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
        let byte_count = width as usize * height as usize * 4;
        let mut bgra = vec![0u8; byte_count];
        unsafe {
            SelectObject(memory_dc, previous);
        }
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

        let rgb = bgrx_to_rgb(&bgra);
        encode_rgb_png(width as u32, height as u32, &rgb)
    }
    #[cfg(not(windows))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "screen capture is only available on Windows",
        ))
    }
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
    use std::io::Cursor;

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
