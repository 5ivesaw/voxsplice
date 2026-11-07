use serde::{Deserialize, Serialize};
use std::net::UdpSocket;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Manager, State, WebviewWindow};

#[derive(Default)]
struct DaemonState {
    started: Mutex<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct UdpSayResponse {
    pub ok: bool,
    pub assembled_text: String,
    pub total_ms: f64,
    pub chunks: usize,
    pub fx_mode: String,
}

#[tauri::command]
fn invoke_udp_say(
    text: String,
    mode: Option<String>,
    fx: Option<String>,
    speaker: Option<String>,
) -> Result<UdpSayResponse, String> {
    let socket = UdpSocket::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(Duration::from_millis(650)))
        .map_err(|e| e.to_string())?;

    let payload = serde_json::json!({
        "text": text,
        "mode": mode,
        "fx": fx,
        "speaker": speaker,
    })
    .to_string();

    socket
        .send_to(payload.as_bytes(), "127.0.0.1:9999")
        .map_err(|e| e.to_string())?;

    let mut buf = [0u8; 8192];
    let (len, _) = socket.recv_from(&mut buf).map_err(|e| e.to_string())?;
    let parsed: UdpSayResponse =
        serde_json::from_slice(&buf[..len]).map_err(|e| e.to_string())?;
    Ok(parsed)
}

#[tauri::command]
fn toggle_always_on_top(window: WebviewWindow, pinned: bool) -> Result<bool, String> {
    window
        .set_always_on_top(pinned)
        .map_err(|e| e.to_string())?;
    Ok(pinned)
}

#[tauri::command]
fn ensure_backend_daemon(state: State<'_, DaemonState>) -> Result<String, String> {
    let mut started = state.started.lock().map_err(|e| e.to_string())?;
    if *started {
        return Ok("Daemon already managed by Tauri".to_string());
    }

    for candidate in ["voxsplice", "5mic", "python3"] {
        let res = if candidate == "python3" {
            Command::new("python3")
                .args(["-m", "voxsplice.cli", "daemon"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
        } else {
            Command::new(candidate)
                .arg("daemon")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
        };

        if res.is_ok() {
            *started = true;
            return Ok(format!("Spawned background engine via {}", candidate));
        }
    }

    Ok("Using standalone UI engine".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(DaemonState::default())
        .setup(|app| {
            let state = app.state::<DaemonState>();
            let _ = ensure_backend_daemon(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            invoke_udp_say,
            toggle_always_on_top,
            ensure_backend_daemon
        ])
        .run(tauri::generate_context!())
        .expect("error while running VoxSplice Tauri application");
}
