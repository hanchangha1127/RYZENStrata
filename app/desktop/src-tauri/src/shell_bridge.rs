//! Token-protected loopback bridge for opening helper windows and sending desktop notifications.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::Duration;
use tauri::AppHandle;

/// What the host needs to find the bridge.
pub struct Bridge {
    pub url: String,
    pub token: String,
}

fn respond(stream: &mut TcpStream, status: &str, body: &serde_json::Value) {
    let text = body.to_string();
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
            text.len()
        )
        .as_bytes(),
    );
}

fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The loopback bridge is served on its own thread. Window construction is an
/// OS UI operation, so wait for Tauri's main thread to finish it before
/// answering the host. In particular, WebView2 and macOS reject window work
/// from this listener thread.
fn open_helper_window_on_main_thread(app: &AppHandle, host_port: u16, session: &str, helper: &str, name: &str) -> Result<(), String> {
    let (sent, received) = mpsc::sync_channel(1);
    let app_for_main = app.clone();
    let session = session.to_owned();
    let helper = helper.to_owned();
    let name = name.to_owned();
    app.run_on_main_thread(move || {
        let result = crate::open_helper_window(&app_for_main, host_port, &session, &helper, &name);
        let _ = sent.send(result);
    }).map_err(|e| format!("could not schedule the helper window: {e}"))?;
    received.recv_timeout(Duration::from_secs(5))
        .map_err(|_| "the helper window did not open on the UI thread in time".to_string())?
}

fn handle(mut stream: TcpStream, app: &AppHandle, token: &str, host_port: u16) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") && buf.len() < 8192 {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let head = String::from_utf8_lossy(&buf).to_string();
    let mut lines = head.lines();
    let request = lines.next().unwrap_or_default().to_string();
    let supplied = lines
        .find_map(|l| l.strip_prefix("authorization: ").or_else(|| l.strip_prefix("Authorization: ")))
        .map(|v| v.trim().trim_start_matches("Bearer ").to_string())
        .unwrap_or_default();
    if !same(&supplied, token) {
        return respond(&mut stream, "401 Unauthorized", &serde_json::json!({ "error": "not allowed" }));
    }
    let mut parts = request.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    match (method, path) {
        ("POST", p) if p.starts_with("/shell/helper?") => {
            let session = crate::shell::query_value(p, "sessionId").unwrap_or_default();
            let helper = crate::shell::query_value(p, "helperId").unwrap_or_default();
            let name = crate::shell::clean_text(&crate::shell::query_value(p, "name").unwrap_or_default(), 80);
            match open_helper_window_on_main_thread(app, host_port, &session, &helper, &name) {
                Ok(()) => respond(&mut stream, "200 OK", &serde_json::json!({ "opened": true })),
                Err(e) => respond(&mut stream, "400 Bad Request", &serde_json::json!({ "error": e })),
            }
        }
        // The host asks for a desktop notification (an approval is waiting): title and body are in the query, bounded and cleaned here.
        ("POST", p) if p.starts_with("/shell/notify") => {
            let title = crate::shell::clean_text(&crate::shell::query_value(p, "title").unwrap_or_default(), 120);
            let body = crate::shell::clean_text(&crate::shell::query_value(p, "body").unwrap_or_default(), 300);
            #[cfg(not(target_os = "linux"))]
            let shown = crate::shell::notify(app, &title, &body);
            #[cfg(target_os = "linux")]
            let shown = false;
            respond(&mut stream, "200 OK", &serde_json::json!({ "shown": shown }))
        }
        _ => respond(&mut stream, "404 Not Found", &serde_json::json!({ "error": "unknown route" })),
    }
}

/// Open the bridge on a free loopback port.
pub fn start(app: &AppHandle, token: String, host_port: u16) -> Option<Bridge> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).ok()?;
    let port = listener.local_addr().ok()?.port();
    let app = app.clone();
    let secret = token.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            handle(stream, &app, &secret, host_port);
        }
    });
    Some(Bridge { url: format!("http://127.0.0.1:{port}"), token })
}

/// Environment for the App Host: where the bridge is, which app version this is.
#[allow(dead_code)]
pub fn apply_env(cmd: &mut std::process::Command, app: &AppHandle, bridge: Option<&Bridge>) {
    cmd.env("STRATA_APP_VERSION", app.package_info().version.to_string());
    if let Some(bridge) = bridge {
        cmd.env("STRATA_SHELL_BRIDGE", &bridge.url).env("STRATA_SHELL_TOKEN", &bridge.token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison_needs_every_byte() {
        assert!(same("abc123", "abc123"));
        assert!(!same("abc123", "abc124"));
        assert!(!same("abc123", "abc12"));
        assert!(!same("", "x"));
    }
}
