# Meta-Router Hook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Before OpenFang sends a request to its configured LLM provider, call the Python meta-router dispatcher to decide the optimal model/provider/endpoint, with fallback to the existing Rust ModelRouter heuristics.

**Architecture:** The kernel's `execute_llm_agent` already uses `ModelRouter` (heuristics) to override model+provider before calling `resolve_driver`. We insert an HTTP call to `http://localhost:5002/v1/route` **before** that existing logic. If the HTTP call succeeds (within 3s), we use its decision. If it fails/times out, we fall through to the existing `ModelRouter` heuristics unchanged.

**Tech Stack:** Rust (reqwest HTTP client), Python (stdlib http.server), OpenFang kernel/runtime crates

**Existing code that matters:**
- `~/meta-router/proxy_server.py` — HTTP server on port 5002 (has `/v1/chat/completions` and `/health`)
- `~/meta-router/orchestrator.py` — `analyze(task, channel)` returns `{model, complexity, type, project, reason}`
- `crates/openfang-runtime/src/routing.rs` — `ModelRouter` with `select_model()` (already tested)
- `crates/openfang-kernel/src/kernel.rs:2120-2150` — ModelRouter already wired in `execute_llm_agent`
- `crates/openfang-kernel/src/kernel.rs:3853` — `resolve_driver()` creates driver from manifest.model.provider

---

### Task 1: Add `/v1/route` endpoint to Python dispatcher

**Files:**
- Modify: `~/meta-router/proxy_server.py`

**Step 1: Add the /v1/route handler to ProxyHandler**

In `proxy_server.py`, the `do_POST` method currently only handles `/v1/chat/completions`. Add a branch for `/v1/route` that calls `orchestrator.analyze()` and returns just the routing decision (no LLM generation).

```python
def do_POST(self):
    if self.path == "/v1/route":
        return self._handle_route()
    if self.path != "/v1/chat/completions":
        self._error(404, "Not found")
        return
    # ... existing chat completions code ...

def _handle_route(self):
    try:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
    except Exception:
        self._error(400, "Invalid JSON body")
        return

    task = body.get("task", "")
    channel = body.get("channel", "general")
    if not task:
        self._error(400, "Missing 'task' field")
        return

    try:
        analysis = analyze(task, channel)
    except Exception as e:
        print(f"[route] analyze error: {e}")
        self._error(502, f"Analyze error: {e}")
        return

    # Map orchestrator model names to OpenFang provider/model/base_url
    model_name = analysis.get("model", "sonnet")
    mapping = _route_mapping(model_name)

    response = {
        "model": mapping["model"],
        "provider": mapping["provider"],
        "base_url": mapping["base_url"],
        "complexity": analysis.get("complexity", "medium"),
        "reason": analysis.get("reason", ""),
    }
    print(f"[route] {channel} -> {response['provider']}/{response['model']} ({response['complexity']})")
    self._json(200, response)
```

Also add the mapping function (before the class):

```python
def _route_mapping(model_name: str) -> dict:
    """Map orchestrator model names to OpenFang-compatible provider/model/base_url."""
    MAPPINGS = {
        "free":       {"provider": "openrouter", "model": "qwen/qwen3-coder:free",
                       "base_url": "https://openrouter.ai/api/v1"},
        "sonnet":     {"provider": "anthropic", "model": "claude-sonnet-4-6",
                       "base_url": "https://api.anthropic.com"},
        "opus":       {"provider": "anthropic", "model": "claude-opus-4-6",
                       "base_url": "https://api.anthropic.com"},
        "codex":      {"provider": "openai", "model": "codex-5.4",
                       "base_url": "https://api.openai.com/v1"},
        "gemini-pro": {"provider": "gemini", "model": "gemini-3.1-pro-preview",
                       "base_url": "https://generativelanguage.googleapis.com"},
    }
    return MAPPINGS.get(model_name, MAPPINGS["sonnet"])
```

Also add the missing import at the top:

```python
from orchestrator import analyze
```

(already imported via `from dispatcher import dispatch` — but `analyze` is in orchestrator, not dispatcher. Add explicit import.)

**Step 2: Test the endpoint manually**

```bash
# On VPS, restart the server
cd ~/meta-router && python3 proxy_server.py &

# Test /v1/route
curl -s -X POST http://localhost:5002/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task": "Crée un bouton React simple", "channel": "asa"}' | python3 -m json.tool
```

Expected: JSON with `model`, `provider`, `base_url`, `complexity`, `reason` fields.

**Step 3: Commit**

```bash
cd ~/meta-router && git init 2>/dev/null
git add proxy_server.py
git commit -m "feat: add /v1/route endpoint for model routing decisions"
```

---

### Task 2: Add meta_router HTTP client module in Rust

**Files:**
- Create: `crates/openfang-runtime/src/meta_router.rs`
- Modify: `crates/openfang-runtime/src/lib.rs` (add `pub mod meta_router;`)

**Step 1: Create the meta_router module**

```rust
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
const META_ROUTER_TIMEOUT: Duration = Duration::from_secs(3);

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
                warn!("Meta-router timed out ({}ms limit)", META_ROUTER_TIMEOUT.as_millis());
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
        // Port 1 is very unlikely to be listening
        let result = query_meta_router("http://127.0.0.1:1", "test task", "general").await;
        assert!(result.is_none());
    }
}
```

**Step 2: Register the module in lib.rs**

Add `pub mod meta_router;` to `crates/openfang-runtime/src/lib.rs` (alphabetical order, after `mcp_server`).

**Step 3: Verify it compiles**

```bash
cargo build --workspace --lib
cargo test --package openfang-runtime -- meta_router --nocapture
```

Expected: 4 tests pass (3 unit + 1 async).

**Step 4: Commit**

```bash
git add crates/openfang-runtime/src/meta_router.rs crates/openfang-runtime/src/lib.rs
git commit -m "feat: add meta_router HTTP client module for external routing"
```

---

### Task 3: Add `meta_router_url` to KernelConfig

**Files:**
- Modify: `crates/openfang-types/src/config.rs` — add field to `KernelConfig`

**Step 1: Find KernelConfig and add the field**

Add to `KernelConfig` struct:

```rust
/// URL of the external meta-router dispatcher (e.g. "http://localhost:5002").
/// When set, the kernel queries this service for model routing decisions before
/// falling back to local ModelRouter heuristics.
#[serde(default)]
pub meta_router_url: Option<String>,
```

The `#[serde(default)]` ensures backward compatibility — existing configs without this field will deserialize to `None`.

No change needed in `Default` impl since `Option<String>` defaults to `None`.

**Step 2: Verify it compiles**

```bash
cargo build --workspace --lib
```

**Step 3: Commit**

```bash
git add crates/openfang-types/src/config.rs
git commit -m "feat: add meta_router_url config field to KernelConfig"
```

---

### Task 4: Wire meta-router call into kernel's execute_llm_agent

**Files:**
- Modify: `crates/openfang-kernel/src/kernel.rs` (~line 2104-2155)

**Context:** Currently `execute_llm_agent` has this flow at lines ~2104-2155:

```
if is_stable {
    // use pinned_model
} else if let Some(ref routing_config) = manifest.routing {
    // ModelRouter heuristics → override manifest.model.model + provider
}

let driver = self.resolve_driver(&manifest)?;
```

We change this to:

```
if is_stable {
    // use pinned_model (unchanged)
} else {
    // Step 1: Try external meta-router (if configured)
    let mut routed_externally = false;
    if let Some(ref meta_url) = self.config.meta_router_url {
        // Derive channel from agent metadata or name
        let channel = manifest.metadata
            .get("channel")
            .and_then(|v| v.as_str())
            .unwrap_or(&manifest.name);

        if let Some(route) = openfang_runtime::meta_router::query_meta_router(
            meta_url, message, channel
        ).await {
            info!(
                agent = %manifest.name,
                provider = %route.provider,
                model = %route.model,
                complexity = %route.complexity,
                reason = %route.reason,
                "Meta-router selected model"
            );
            manifest.model.model = route.model;
            manifest.model.provider = route.provider;
            if !route.base_url.is_empty() {
                manifest.model.base_url = Some(route.base_url);
            }
            routed_externally = true;
        }
    }

    // Step 2: Fallback to local ModelRouter heuristics (if meta-router didn't answer)
    if !routed_externally {
        if let Some(ref routing_config) = manifest.routing {
            // ... existing ModelRouter code unchanged ...
        }
    }
}

let driver = self.resolve_driver(&manifest)?;
```

**Step 1: Apply the change**

In `kernel.rs`, replace the block from `let is_stable = ...` through the existing model routing `else if` block (roughly lines 2104-2155) with the new logic above. Keep the existing ModelRouter code as-is inside the `!routed_externally` branch.

**Important:** The `manifest.model.base_url = Some(route.base_url)` override ensures `resolve_driver()` will create a driver pointing to the correct endpoint, since `resolve_driver` respects `manifest.model.base_url` when present (`has_custom_url` check at line 3860).

**Step 2: Verify it compiles**

```bash
cargo build --workspace --lib
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 3: Run existing tests**

```bash
cargo test --workspace
```

All 1744+ tests must still pass. The meta_router call is behind `if let Some(ref meta_url) = self.config.meta_router_url` so existing configs (where it's `None`) are unaffected.

**Step 4: Commit**

```bash
git add crates/openfang-kernel/src/kernel.rs
git commit -m "feat: wire meta-router HTTP call into execute_llm_agent with ModelRouter fallback"
```

---

### Task 5: Add meta_router_url to openfang.toml config

**Files:**
- Modify: `openfang.toml.example`
- Modify: the actual `~/.openfang/config.toml` on the VPS

**Step 1: Add to openfang.toml.example**

Add this section (near other URL configs):

```toml
# External meta-router URL for intelligent model routing.
# When set, the kernel queries this service before falling back to local heuristics.
# meta_router_url = "http://localhost:5002"
```

**Step 2: Set it in the live config**

```bash
# On VPS
echo 'meta_router_url = "http://localhost:5002"' >> ~/.openfang/config.toml
```

**Step 3: Commit**

```bash
git add openfang.toml.example
git commit -m "docs: add meta_router_url to example config"
```

---

### Task 6: Build, deploy, and live test

**Step 1: Create the feature branch**

```bash
cd /root/openfang-src
git checkout -b feature/meta-router-hook
# All commits from tasks 2-5 should already be on this branch
```

(Note: create the branch FIRST before starting tasks 2-5, or rebase onto it.)

**Step 2: Build release**

```bash
cargo build --workspace --lib
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

All three must pass.

**Step 3: Start the Python meta-router**

```bash
cd ~/meta-router
# Kill existing if running
pkill -f proxy_server.py || true
nohup python3 proxy_server.py > /tmp/meta-router.log 2>&1 &
sleep 1
curl -s http://localhost:5002/health
# Expected: {"status": "ok"}
```

**Step 4: Test /v1/route directly**

```bash
curl -s -X POST http://localhost:5002/v1/route \
  -H "Content-Type: application/json" \
  -d '{"task": "Crée un bouton React", "channel": "asa"}' | python3 -m json.tool
```

Expected output like:
```json
{
    "model": "qwen/qwen3-coder:free",
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "complexity": "simple",
    "reason": "Tache atomique et isolee"
}
```

**Step 5: Build and start OpenFang daemon**

```bash
# Kill existing daemon
pkill -f openfang || true
sleep 3

cargo build --release -p openfang-cli
GROQ_API_KEY=<key> target/release/openfang start &
sleep 6
curl -s http://127.0.0.1:4200/api/health
```

**Step 6: Send a test message and check logs**

```bash
# Get an agent ID
AGENT_ID=$(curl -s http://127.0.0.1:4200/api/agents | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")

# Send a message
curl -s -X POST "http://127.0.0.1:4200/api/agents/$AGENT_ID/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Say hello in 5 words."}'
```

Check OpenFang logs for:
```
Meta-router selected model  provider=anthropic model=claude-sonnet-4-6 complexity=medium
```

Or if meta-router was down:
```
Meta-router not reachable — using local routing
Model routing applied  complexity=simple routed_model=claude-haiku-4-5-20251001
```

**Step 7: Test fallback (kill meta-router, send another message)**

```bash
pkill -f proxy_server.py

curl -s -X POST "http://127.0.0.1:4200/api/agents/$AGENT_ID/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2+2?"}'
```

Check logs for the local ModelRouter fallback kicking in.

---

## Summary of changes

| File | Action | Description |
|------|--------|-------------|
| `~/meta-router/proxy_server.py` | Modify | Add `/v1/route` POST endpoint + `_route_mapping()` |
| `crates/openfang-runtime/src/meta_router.rs` | Create | HTTP client for `/v1/route` with 3s timeout |
| `crates/openfang-runtime/src/lib.rs` | Modify | Add `pub mod meta_router;` |
| `crates/openfang-types/src/config.rs` | Modify | Add `meta_router_url: Option<String>` to KernelConfig |
| `crates/openfang-kernel/src/kernel.rs` | Modify | Wire meta-router call before ModelRouter fallback |
| `openfang.toml.example` | Modify | Document `meta_router_url` option |
| `~/.openfang/config.toml` | Modify | Set `meta_router_url = "http://localhost:5002"` |
