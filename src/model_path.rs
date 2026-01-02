//! Model file discovery with multiple fallback locations.
//!
//! Search order:
//! 1. ANR_MODEL_DIR environment variable
//! 2. Next to the plugin/executable binary
//! 3. Platform-specific well-known locations

use std::path::PathBuf;

const PMLN_MODEL: &str = "PMLN_model.onnx";
const STLN_MODEL: &str = "STLN_model.onnx";

/// Returns the directory containing the plugin binary itself.
fn get_binary_dir() -> Option<PathBuf> {
    // Define a dummy static so we have a symbol inside THIS binary
    static DUMMY: usize = 0;

    #[cfg(target_os = "linux")]
    {
        use std::ffi::CStr;
        use std::os::unix::ffi::OsStrExt;

        let mut info = unsafe { std::mem::zeroed::<libc::Dl_info>() };
        // Ask Linux: "Who owns the memory address of DUMMY?"
        if unsafe { libc::dladdr(&DUMMY as *const _ as *mut _, &mut info) } != 0 {
            if !info.dli_fname.is_null() {
                let name = unsafe { CStr::from_ptr(info.dli_fname) };
                let path = PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes()));
                return path.parent().map(|p| p.to_path_buf());
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::ffi::CStr;
        use std::os::unix::ffi::OsStrExt;

        // macOS uses the same dladdr API as Linux
        #[repr(C)]
        struct DlInfo {
            dli_fname: *const std::ffi::c_char,
            dli_fbase: *mut std::ffi::c_void,
            dli_sname: *const std::ffi::c_char,
            dli_saddr: *mut std::ffi::c_void,
        }

        extern "C" {
            fn dladdr(addr: *const std::ffi::c_void, info: *mut DlInfo) -> i32;
        }

        let mut info = DlInfo {
            dli_fname: std::ptr::null(),
            dli_fbase: std::ptr::null_mut(),
            dli_sname: std::ptr::null(),
            dli_saddr: std::ptr::null_mut(),
        };

        if unsafe { dladdr(&DUMMY as *const _ as *const _, &mut info) } != 0 {
            if !info.dli_fname.is_null() {
                let name = unsafe { CStr::from_ptr(info.dli_fname) };
                let path = PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes()));
                return path.parent().map(|p| p.to_path_buf());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;
        use windows_sys::Win32::System::LibraryLoader::{
            GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        };

        let mut module = std::ptr::null_mut();
        let address = &DUMMY as *const _ as *const std::ffi::c_void;

        unsafe {
            if GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                address as *const _,
                &mut module,
            ) != 0
            {
                let mut buffer = [0u16; 1024];
                let len = GetModuleFileNameW(module, buffer.as_mut_ptr(), 1024);
                if len > 0 {
                    let path = PathBuf::from(OsString::from_wide(&buffer[0..len as usize]));
                    return path.parent().map(|p| p.to_path_buf());
                }
            }
        }
    }

    None
}

/// Returns platform-specific well-known model directories.
fn well_known_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    #[cfg(target_os = "linux")]
    {
        // XDG data dirs
        if let Ok(data_home) = std::env::var("XDG_DATA_HOME") {
            dirs.push(PathBuf::from(data_home).join("anr-plugin"));
        } else if let Ok(home) = std::env::var("HOME") {
            dirs.push(PathBuf::from(&home).join(".local/share/anr-plugin"));
        }

        // System-wide
        dirs.push(PathBuf::from("/usr/share/anr-plugin"));
        dirs.push(PathBuf::from("/usr/local/share/anr-plugin"));
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            // User application support
            dirs.push(PathBuf::from(&home).join("Library/Application Support/ANR Plugin"));
        }
        // System-wide
        dirs.push(PathBuf::from("/Library/Application Support/ANR Plugin"));
    }

    #[cfg(target_os = "windows")]
    {
        // User local data
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            dirs.push(PathBuf::from(local_app_data).join("ANR Plugin"));
        }
        // User roaming data
        if let Ok(app_data) = std::env::var("APPDATA") {
            dirs.push(PathBuf::from(app_data).join("ANR Plugin"));
        }
        // Program Files (common location for shared data)
        if let Ok(program_data) = std::env::var("PROGRAMDATA") {
            dirs.push(PathBuf::from(program_data).join("ANR Plugin"));
        }
    }

    dirs
}

/// Check if a directory contains both required model files.
fn has_models(dir: &PathBuf) -> bool {
    dir.join(PMLN_MODEL).is_file() && dir.join(STLN_MODEL).is_file()
}

/// Discover the model directory, checking multiple locations in order.
/// Returns the first directory containing both PMLN_model.onnx and STLN_model.onnx.
pub fn find_model_dir() -> Option<PathBuf> {
    // 1. Environment variable (highest priority)
    if let Ok(env_dir) = std::env::var("ANR_MODEL_DIR") {
        let path = PathBuf::from(&env_dir);
        if has_models(&path) {
            return Some(path);
        }
    }

    // 2. Next to the plugin/executable binary
    if let Some(binary_dir) = get_binary_dir() {
        if has_models(&binary_dir) {
            return Some(binary_dir);
        }
        // Also check a "models" subdirectory next to binary
        let models_subdir = binary_dir.join("models");
        if has_models(&models_subdir) {
            return Some(models_subdir);
        }
    }

    // 3. Well-known platform-specific locations
    for dir in well_known_dirs() {
        if has_models(&dir) {
            return Some(dir);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_dir() {
        // Should return Some path in tests
        let dir = get_binary_dir();
        println!("Binary dir: {:?}", dir);
        assert!(dir.is_some());
    }

    #[test]
    fn test_well_known_dirs() {
        let dirs = well_known_dirs();
        println!("Well-known dirs: {:?}", dirs);
        assert!(!dirs.is_empty());
    }
}
