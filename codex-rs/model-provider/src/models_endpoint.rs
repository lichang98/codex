use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use codex_api::ModelsClient;
use codex_api::RequestTelemetry;
use codex_api::ReqwestTransport;
use codex_api::TransportError;
use codex_api::auth_header_telemetry;
use codex_api::map_api_error;
use codex_feedback::FeedbackRequestTags;
use codex_feedback::emit_feedback_request_tags_with_auth_env;
use codex_login::AuthEnvTelemetry;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_login::collect_auth_env_telemetry;
use codex_login::default_client::build_reqwest_client;
use codex_model_provider_info::ModelProviderInfo;
use codex_models_manager::manager::ModelsEndpointClient;
use codex_otel::TelemetryAuthMode;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CoreResult;
use codex_protocol::openai_models::ModelInfo;
use codex_response_debug_context::extract_response_debug_context;
use codex_response_debug_context::telemetry_transport_error_message;
use http::HeaderMap;
use tokio::time::timeout;

use crate::auth::resolve_provider_auth;

const MODELS_REFRESH_TIMEOUT: Duration = Duration::from_secs(5);
const MODELS_ENDPOINT: &str = "/models";

/// Provider-owned OpenAI-compatible `/models` endpoint.
#[derive(Debug)]
pub(crate) struct OpenAiModelsEndpoint {
    provider_info: ModelProviderInfo,
    auth_manager: Option<Arc<AuthManager>>,
}

impl OpenAiModelsEndpoint {
    pub(crate) fn new(
        provider_info: ModelProviderInfo,
        auth_manager: Option<Arc<AuthManager>>,
    ) -> Self {
        Self {
            provider_info,
            auth_manager,
        }
    }

    async fn auth(&self) -> Option<CodexAuth> {
        match self.auth_manager.as_ref() {
            Some(auth_manager) => auth_manager.auth().await,
            None => None,
        }
    }

    fn auth_env(&self) -> AuthEnvTelemetry {
        let codex_api_key_env_enabled = self
            .auth_manager
            .as_ref()
            .is_some_and(|auth_manager| auth_manager.codex_api_key_env_enabled());
        collect_auth_env_telemetry(&self.provider_info, codex_api_key_env_enabled)
    }
}

#[async_trait]
impl ModelsEndpointClient for OpenAiModelsEndpoint {
    fn provider_id(&self) -> String {
        provider_config_fingerprint(&self.provider_info)
    }

    fn has_command_auth(&self) -> bool {
        self.provider_info.has_command_auth()
    }

    fn has_self_provided_auth(&self) -> bool {
        // Explicit bearer source. `api_key()` (rather than `env_key.is_some()`)
        // so providers that declare `env_key` without the variable being set
        // don't claim to have auth.
        if self.provider_info.api_key().ok().flatten().is_some()
            || self.provider_info.experimental_bearer_token.is_some()
        {
            return true;
        }
        // Non-OpenAI providers (`requires_openai_auth = false`) handle their
        // own auth via whatever mechanism the user configured — header-based
        // auth (`http_headers` / `env_http_headers`), token-less local OSS
        // servers, etc. We can't reliably introspect those, so treat the
        // provider's declaration of "I don't use OpenAI auth" as sufficient
        // signal that `/models` is worth attempting; a 401 just falls back
        // to the bundled catalog.
        !self.provider_info.requires_openai_auth
    }

    async fn uses_codex_backend(&self) -> bool {
        self.auth()
            .await
            .as_ref()
            .is_some_and(CodexAuth::uses_codex_backend)
    }

    async fn list_models(
        &self,
        client_version: &str,
    ) -> CoreResult<(Vec<ModelInfo>, Option<String>)> {
        let _timer =
            codex_otel::start_global_timer("codex.remote_models.fetch_update.duration_ms", &[]);
        let auth = self.auth().await;
        let auth_mode = auth.as_ref().map(CodexAuth::auth_mode);
        let api_provider = self.provider_info.to_api_provider(auth_mode)?;
        let api_auth = resolve_provider_auth(auth.as_ref(), &self.provider_info)?;
        let transport = ReqwestTransport::new(build_reqwest_client());
        let auth_telemetry = auth_header_telemetry(api_auth.as_ref());
        let request_telemetry: Arc<dyn RequestTelemetry> = Arc::new(ModelsRequestTelemetry {
            auth_mode: auth_mode.map(|mode| TelemetryAuthMode::from(mode).to_string()),
            auth_header_attached: auth_telemetry.attached,
            auth_header_name: auth_telemetry.name,
            auth_env: self.auth_env(),
        });
        let client = ModelsClient::new(transport, api_provider, api_auth)
            .with_telemetry(Some(request_telemetry));

        timeout(
            MODELS_REFRESH_TIMEOUT,
            client.list_models(client_version, HeaderMap::new()),
        )
        .await
        .map_err(|_| CodexErr::Timeout)?
        .map_err(map_api_error)
    }
}

#[derive(Clone)]
struct ModelsRequestTelemetry {
    auth_mode: Option<String>,
    auth_header_attached: bool,
    auth_header_name: Option<&'static str>,
    auth_env: AuthEnvTelemetry,
}

impl RequestTelemetry for ModelsRequestTelemetry {
    fn on_request(
        &self,
        attempt: u64,
        status: Option<http::StatusCode>,
        error: Option<&TransportError>,
        duration: Duration,
    ) {
        let success = status.is_some_and(|code| code.is_success()) && error.is_none();
        let error_message = error.map(telemetry_transport_error_message);
        let response_debug = error
            .map(extract_response_debug_context)
            .unwrap_or_default();
        let status = status.map(|status| status.as_u16());
        tracing::event!(
            target: "codex_otel.log_only",
            tracing::Level::INFO,
            event.name = "codex.api_request",
            duration_ms = %duration.as_millis(),
            http.response.status_code = status,
            success = success,
            error.message = error_message.as_deref(),
            attempt = attempt,
            endpoint = MODELS_ENDPOINT,
            auth.header_attached = self.auth_header_attached,
            auth.header_name = self.auth_header_name,
            auth.env_openai_api_key_present = self.auth_env.openai_api_key_env_present,
            auth.env_codex_api_key_present = self.auth_env.codex_api_key_env_present,
            auth.env_codex_api_key_enabled = self.auth_env.codex_api_key_env_enabled,
            auth.env_provider_key_name = self.auth_env.provider_env_key_name.as_deref(),
            auth.env_provider_key_present = self.auth_env.provider_env_key_present,
            auth.env_refresh_token_url_override_present = self.auth_env.refresh_token_url_override_present,
            auth.request_id = response_debug.request_id.as_deref(),
            auth.cf_ray = response_debug.cf_ray.as_deref(),
            auth.error = response_debug.auth_error.as_deref(),
            auth.error_code = response_debug.auth_error_code.as_deref(),
            auth.mode = self.auth_mode.as_deref(),
        );
        tracing::event!(
            target: "codex_otel.trace_safe",
            tracing::Level::INFO,
            event.name = "codex.api_request",
            duration_ms = %duration.as_millis(),
            http.response.status_code = status,
            success = success,
            error.message = error_message.as_deref(),
            attempt = attempt,
            endpoint = MODELS_ENDPOINT,
            auth.header_attached = self.auth_header_attached,
            auth.header_name = self.auth_header_name,
            auth.env_openai_api_key_present = self.auth_env.openai_api_key_env_present,
            auth.env_codex_api_key_present = self.auth_env.codex_api_key_env_present,
            auth.env_codex_api_key_enabled = self.auth_env.codex_api_key_env_enabled,
            auth.env_provider_key_name = self.auth_env.provider_env_key_name.as_deref(),
            auth.env_provider_key_present = self.auth_env.provider_env_key_present,
            auth.env_refresh_token_url_override_present = self.auth_env.refresh_token_url_override_present,
            auth.request_id = response_debug.request_id.as_deref(),
            auth.cf_ray = response_debug.cf_ray.as_deref(),
            auth.error = response_debug.auth_error.as_deref(),
            auth.error_code = response_debug.auth_error_code.as_deref(),
            auth.mode = self.auth_mode.as_deref(),
        );
        emit_feedback_request_tags_with_auth_env(
            &FeedbackRequestTags {
                endpoint: MODELS_ENDPOINT,
                auth_header_attached: self.auth_header_attached,
                auth_header_name: self.auth_header_name,
                auth_mode: self.auth_mode.as_deref(),
                auth_retry_after_unauthorized: None,
                auth_recovery_mode: None,
                auth_recovery_phase: None,
                auth_connection_reused: None,
                auth_request_id: response_debug.request_id.as_deref(),
                auth_cf_ray: response_debug.cf_ray.as_deref(),
                auth_error: response_debug.auth_error.as_deref(),
                auth_error_code: response_debug.auth_error_code.as_deref(),
                auth_recovery_followup_success: None,
                auth_recovery_followup_status: None,
            },
            &self.auth_env,
        );
    }
}

/// Stable identifier for a `ModelProviderInfo` used to scope the on-disk
/// models cache. `ModelProviderInfo::name` alone is insufficient because it
/// is a friendly display label — both the built-in `ollama` and `lmstudio`
/// providers, for example, share `name = "gpt-oss"` and only differ in
/// `base_url`. The fingerprint hashes the canonicalized provider config so
/// any field that could change `/models` semantics (base_url, wire_api,
/// env_key, auth, aws, http_headers, etc.) produces a distinct cache entry.
pub(crate) fn provider_config_fingerprint(provider_info: &ModelProviderInfo) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    let canonical = canonical_json(
        serde_json::to_value(provider_info).unwrap_or(serde_json::Value::Null),
    );
    let serialized = serde_json::to_string(&canonical).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    // Prefix with the display name to keep cache files self-documenting in
    // logs while the hash suffix guarantees uniqueness across configs that
    // share a name (ollama vs lmstudio, two user-defined providers with the
    // same friendly label, etc.).
    format!("{}:{:016x}", provider_info.name, hasher.finish())
}

fn canonical_json(value: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                if let Some(v) = map.get(&key) {
                    sorted.insert(key, canonical_json(v.clone()));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical_json).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use codex_protocol::config_types::ModelProviderAuthInfo;

    fn provider_info_with_command_auth() -> ModelProviderInfo {
        ModelProviderInfo {
            auth: Some(ModelProviderAuthInfo {
                command: "print-token".to_string(),
                args: Vec::new(),
                timeout_ms: NonZeroU64::new(5_000).expect("timeout should be non-zero"),
                refresh_interval_ms: 300_000,
                cwd: std::env::current_dir()
                    .expect("current dir should be available")
                    .try_into()
                    .expect("current dir should be absolute"),
            }),
            requires_openai_auth: false,
            ..ModelProviderInfo::create_openai_provider(/*base_url*/ None)
        }
    }

    #[test]
    fn command_auth_provider_reports_command_auth_without_cached_auth() {
        let endpoint = OpenAiModelsEndpoint::new(
            provider_info_with_command_auth(),
            /*auth_manager*/ None,
        );

        assert!(endpoint.has_command_auth());
    }

    #[test]
    fn provider_without_command_auth_reports_no_command_auth() {
        let endpoint = OpenAiModelsEndpoint::new(
            ModelProviderInfo::create_openai_provider(/*base_url*/ None),
            /*auth_manager*/ None,
        );

        assert!(!endpoint.has_command_auth());
    }

    #[test]
    fn provider_with_experimental_bearer_token_reports_self_provided_auth() {
        let mut info = ModelProviderInfo::create_openai_provider(/*base_url*/ None);
        info.experimental_bearer_token = Some("provider-token".to_string());
        let endpoint = OpenAiModelsEndpoint::new(info, /*auth_manager*/ None);

        assert!(endpoint.has_self_provided_auth());
    }

    #[test]
    fn provider_without_self_provided_auth_reports_none() {
        // Stock OpenAI provider requires OpenAI auth and has no bearer source.
        let endpoint = OpenAiModelsEndpoint::new(
            ModelProviderInfo::create_openai_provider(/*base_url*/ None),
            /*auth_manager*/ None,
        );

        assert!(!endpoint.has_self_provided_auth());
    }

    #[test]
    fn non_openai_provider_reports_self_provided_auth_without_explicit_bearer() {
        // A custom provider (`requires_openai_auth = false`) authenticated via
        // env_http_headers (or no auth at all for local OSS) qualifies for
        // /models refresh even with no bearer-token field set.
        let info = ModelProviderInfo {
            name: "header-auth-provider".to_string(),
            base_url: Some("http://localhost:9999/v1".to_string()),
            env_http_headers: Some(
                [("X-API-Key".to_string(), "MY_PROVIDER_API_KEY".to_string())]
                    .into_iter()
                    .collect(),
            ),
            requires_openai_auth: false,
            ..ModelProviderInfo::default()
        };
        let endpoint = OpenAiModelsEndpoint::new(info, /*auth_manager*/ None);

        assert!(endpoint.has_self_provided_auth());
    }

    #[test]
    fn provider_id_distinguishes_oss_providers_with_shared_display_name() {
        // Built-in ollama and lmstudio providers both share name = "gpt-oss"
        // but differ in base_url. The cache fingerprint must disambiguate
        // them so switching providers does not reuse the other catalog.
        let ollama = codex_model_provider_info::create_oss_provider_with_base_url(
            &format!(
                "http://localhost:{}/v1",
                codex_model_provider_info::DEFAULT_OLLAMA_PORT
            ),
            codex_model_provider_info::WireApi::Responses,
        );
        let lmstudio = codex_model_provider_info::create_oss_provider_with_base_url(
            &format!(
                "http://localhost:{}/v1",
                codex_model_provider_info::DEFAULT_LMSTUDIO_PORT
            ),
            codex_model_provider_info::WireApi::Responses,
        );
        assert_eq!(ollama.name, lmstudio.name);

        let ollama_endpoint = OpenAiModelsEndpoint::new(ollama, /*auth_manager*/ None);
        let lmstudio_endpoint = OpenAiModelsEndpoint::new(lmstudio, /*auth_manager*/ None);

        assert_ne!(
            ollama_endpoint.provider_id(),
            lmstudio_endpoint.provider_id(),
            "providers with the same display name must produce distinct cache fingerprints"
        );
    }

    #[test]
    fn provider_id_is_stable_for_equal_configs() {
        // Two endpoints built from clones of the same provider info must
        // hash to the same fingerprint so cache reads hit on a re-launch.
        let info = codex_model_provider_info::create_oss_provider_with_base_url(
            "http://localhost:11434/v1",
            codex_model_provider_info::WireApi::Responses,
        );
        let a = OpenAiModelsEndpoint::new(info.clone(), /*auth_manager*/ None);
        let b = OpenAiModelsEndpoint::new(info, /*auth_manager*/ None);

        assert_eq!(a.provider_id(), b.provider_id());
    }
}
