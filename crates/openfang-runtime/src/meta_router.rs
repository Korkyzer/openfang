//! Meta-router client — calls an external Python dispatcher for intelligent model routing.
//!
//! Sends a lightweight HTTP POST to the meta-router's `/v1/route` endpoint
//! with the task description and channel. Returns `(provider, model, base_url)`.
//! On timeout (3s) or error, returns `None` so the caller can fall back to
//! the local `ModelRouter` heuristics.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

/// Default timeout for the meta-router HTTP call.
const META_ROUTER_TIMEOUT: Duration = Duration::from_secs(8);

/// Request payload sent to the meta-router `/v1/route` endpoint.
#[derive(Debug, Serialize)]
struct RouteRequest<'a> {
    task: &'a str,
    channel: &'a str,
}

/// Response from the meta-router `/v1/route` endpoint.
#[derive(Debug, Deserialize)]
pub struct RouteResponse {
    pub model: String,
    pub provider: String,
    pub base_url: String,
    #[serde(default)]
    pub complexity: String,
    #[serde(default)]
    pub reason: String,
}

/// Query the external meta-router for a routing decision.
///
/// Returns `None` on timeout, connection error, or invalid response.
/// The caller should fall back to local heuristics when this returns `None`.
pub async fn query_meta_router(
    meta_router_url: &str,
    task: &str,
    channel: &str,
) -> Option<RouteResponse> {
    let url = format!("{}/v1/route", meta_router_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(META_ROUTER_TIMEOUT)
        .build()
        .ok()?;

    let payload = RouteRequest { task, channel };

    debug!(url = %url, task_len = task.len(), channel = %channel, "Querying meta-router");

    match client.post(&url).json(&payload).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<RouteResponse>().await {
                    Ok(route) => {
                        debug!(
                            provider = %route.provider,
                            model = %route.model,
                            complexity = %route.complexity,
                            "Meta-router returned routing decision"
                        );
                        Some(route)
                    }
                    Err(e) => {
                        warn!(error = %e, "Meta-router response parse failed");
                        None
                    }
                }
            } else {
                warn!(status = %resp.status(), "Meta-router returned error status");
                None
            }
        }
        Err(e) => {
            if e.is_timeout() {
                warn!(
                    "Meta-router timed out ({}ms limit)",
                    META_ROUTER_TIMEOUT.as_millis()
                );
            } else if e.is_connect() {
                debug!("Meta-router not reachable — using local routing");
            } else {
                warn!(error = %e, "Meta-router request failed");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_request_serialization() {
        let req = RouteRequest {
            task: "Create a React button",
            channel: "asa",
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["task"], "Create a React button");
        assert_eq!(json["channel"], "asa");
    }

    #[test]
    fn test_route_response_deserialization() {
        let json = r#"{"model":"claude-sonnet-4-6","provider":"anthropic","base_url":"https://api.anthropic.com","complexity":"medium","reason":"well-defined task"}"#;
        let resp: RouteResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "claude-sonnet-4-6");
        assert_eq!(resp.provider, "anthropic");
        assert_eq!(resp.base_url, "https://api.anthropic.com");
        assert_eq!(resp.complexity, "medium");
    }

    #[test]
    fn test_route_response_missing_optional_fields() {
        let json = r#"{"model":"claude-sonnet-4-6","provider":"anthropic","base_url":"https://api.anthropic.com"}"#;
        let resp: RouteResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.complexity, "");
        assert_eq!(resp.reason, "");
    }

    #[tokio::test]
    async fn test_query_unreachable_returns_none() {
        let result = query_meta_router("http://127.0.0.1:1", "test task", "general").await;
        assert!(result.is_none());
    }
}
