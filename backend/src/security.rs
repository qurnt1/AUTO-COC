use std::{net::SocketAddr, sync::Arc};

use axum::http::{HeaderMap, Method};
use rand::{RngCore, rng};

#[derive(Clone)]
pub struct SecurityState {
    host: Arc<str>,
    origin: Arc<str>,
    #[cfg(debug_assertions)]
    vite_origin: Arc<str>,
    token: Arc<str>,
}

impl SecurityState {
    pub fn new(addr: SocketAddr, origin: String) -> Self {
        let mut bytes = [0u8; 32];
        rng().fill_bytes(&mut bytes);
        let token = bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Self {
            host: Arc::from(addr.to_string()),
            origin: Arc::from(origin),
            #[cfg(debug_assertions)]
            vite_origin: Arc::from("http://127.0.0.1:5173"),
            token: Arc::from(token),
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn host_is_valid(&self, headers: &HeaderMap) -> bool {
        let values = headers.get_all(axum::http::header::HOST);
        values.iter().count() == 1
            && values.iter().next().and_then(|value| value.to_str().ok()) == Some(self.host())
    }

    pub fn origin_is_exact(&self, headers: &HeaderMap) -> bool {
        let values = headers.get_all(axum::http::header::ORIGIN);
        values.iter().count() == 1
            && values.iter().next().and_then(|value| value.to_str().ok()) == Some(self.origin())
    }

    fn origin_is_allowed(&self, origin: &str) -> bool {
        if origin == self.origin() {
            return true;
        }
        #[cfg(debug_assertions)]
        if origin == self.vite_origin.as_ref() {
            return true;
        }
        false
    }

    fn origin_header_is_allowed(&self, headers: &HeaderMap) -> bool {
        let values = headers.get_all(axum::http::header::ORIGIN);
        values.iter().count() == 1
            && values
                .iter()
                .next()
                .and_then(|value| value.to_str().ok())
                .is_some_and(|origin| self.origin_is_allowed(origin))
    }

    pub fn same_origin_fallback(&self, headers: &HeaderMap) -> bool {
        let site = headers
            .get("sec-fetch-site")
            .and_then(|value| value.to_str().ok());
        let referers = headers.get_all(axum::http::header::REFERER);
        if site != Some("same-origin") || referers.iter().count() != 1 {
            return false;
        }
        let Some(referer) = referers.iter().next().and_then(|value| value.to_str().ok()) else {
            return false;
        };
        let Ok(url) = url::Url::parse(referer) else {
            return false;
        };
        self.origin_is_allowed(&url.origin().ascii_serialization())
    }

    pub fn browser_origin_valid(&self, method: &Method, headers: &HeaderMap) -> bool {
        self.origin_header_is_allowed(headers)
            || (*method == Method::GET
                && !headers.contains_key(axum::http::header::ORIGIN)
                && self.same_origin_fallback(headers))
    }

    pub fn bootstrap_origin_valid(&self, headers: &HeaderMap) -> bool {
        self.origin_header_is_allowed(headers)
            || (!headers.contains_key(axum::http::header::ORIGIN)
                && self.same_origin_fallback(headers))
    }

    pub fn token_is_valid(&self, headers: &HeaderMap) -> bool {
        let Some(candidate) = headers
            .get("x-auto-coc-session")
            .and_then(|value| value.to_str().ok())
        else {
            return false;
        };
        constant_time_eq(candidate.as_bytes(), self.token.as_bytes())
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn state() -> SecurityState {
        SecurityState::new(
            "127.0.0.1:40123".parse().unwrap(),
            "http://127.0.0.1:40123".into(),
        )
    }

    #[test]
    fn accepts_only_exact_host_and_browser_origin() {
        let state = state();
        let mut headers = HeaderMap::new();
        headers.insert("host", HeaderValue::from_static("127.0.0.1:40123"));
        headers.insert("origin", HeaderValue::from_static("http://127.0.0.1:40123"));
        assert!(state.host_is_valid(&headers));
        assert!(state.origin_is_exact(&headers));
        headers.insert("host", HeaderValue::from_static("localhost:40123"));
        assert!(!state.host_is_valid(&headers));
        headers.append("host", HeaderValue::from_static("127.0.0.1:40123"));
        assert!(!state.host_is_valid(&headers));
    }

    #[test]
    fn originless_get_requires_fetch_metadata_and_exact_referer_origin() {
        let state = state();
        let mut headers = HeaderMap::new();
        headers.insert("sec-fetch-site", HeaderValue::from_static("same-origin"));
        headers.insert(
            "referer",
            HeaderValue::from_static("http://127.0.0.1:40123/macros"),
        );
        assert!(state.browser_origin_valid(&Method::GET, &headers));
        assert!(!state.browser_origin_valid(&Method::POST, &headers));
        headers.insert(
            "referer",
            HeaderValue::from_static("http://127.0.0.1.evil/"),
        );
        assert!(!state.browser_origin_valid(&Method::GET, &headers));
    }

    #[test]
    fn rejects_null_and_duplicate_origins() {
        let state = state();
        let mut headers = HeaderMap::new();
        headers.insert("origin", HeaderValue::from_static("null"));
        assert!(!state.origin_is_exact(&headers));
        assert!(!state.origin_header_is_allowed(&headers));
    }

    #[test]
    fn vite_origin_is_debug_only_and_exact() {
        let state = state();
        let mut headers = HeaderMap::new();
        headers.insert("origin", HeaderValue::from_static("http://127.0.0.1:5173"));
        assert_eq!(
            state.origin_header_is_allowed(&headers),
            cfg!(debug_assertions)
        );
        headers.insert("origin", HeaderValue::from_static("http://127.0.0.1:5174"));
        assert!(!state.origin_header_is_allowed(&headers));
        headers.insert("origin", HeaderValue::from_static("http://127.0.0.1:40123"));
        headers.append("origin", HeaderValue::from_static("http://127.0.0.1:40123"));
        assert!(!state.origin_is_exact(&headers));
    }

    #[test]
    fn session_token_check_does_not_accept_prefixes() {
        let state = state();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-auto-coc-session",
            HeaderValue::from_str(state.token()).unwrap(),
        );
        assert!(state.token_is_valid(&headers));
        headers.insert("x-auto-coc-session", HeaderValue::from_static("wrong"));
        assert!(!state.token_is_valid(&headers));
    }
}
