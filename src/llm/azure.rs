// Azure OpenAI LLM Provider
//
// Uses the standard /chat/completions endpoint (not the new /responses API).
// This is necessary because Azure OpenAI doesn't support OpenAI's Responses API.
//
// Config format: "azure:gpt-4.1"
// Env: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::sync::LazyLock;
use std::time::Duration;

use octolib::llm::{Message, ProviderResponse};

static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
	Client::builder()
		.pool_max_idle_per_host(10)
		.pool_idle_timeout(Duration::from_secs(30))
		.timeout(Duration::from_secs(120))
		.connect_timeout(Duration::from_secs(10))
		.build()
		.expect("Failed to create HTTP client for Azure OpenAI LLM")
});

fn get_credentials() -> Result<(String, String)> {
	let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| {
		anyhow::anyhow!("AZURE_OPENAI_API_KEY not set")
	})?;
	let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
		anyhow::anyhow!("AZURE_OPENAI_ENDPOINT not set")
	})?;
	Ok((api_key, endpoint.trim_end_matches('/').to_string()))
}

/// Call Azure OpenAI chat completions with the standard format.
pub async fn chat_completion(
	messages: &[Message],
	model: &str,
	temperature: f32,
	max_tokens: u32,
) -> Result<String> {
	let (api_key, endpoint) = get_credentials()?;

	let url = format!(
		"{}/openai/deployments/{}/chat/completions?api-version=2024-12-01-preview",
		endpoint, model
	);

	// Convert octolib Messages to standard chat/completions format
	let msgs: Vec<serde_json::Value> = messages
		.iter()
		.map(|m| {
			json!({
				"role": m.role,
				"content": m.content
			})
		})
		.collect();

	let request_body = json!({
		"messages": msgs,
		"temperature": temperature,
		"max_tokens": max_tokens,
	});

	let response = HTTP_CLIENT
		.post(&url)
		.header("api-key", &api_key)
		.header("Content-Type", "application/json")
		.json(&request_body)
		.send()
		.await
		.map_err(|e| anyhow::anyhow!("Azure OpenAI LLM request failed: {}", e))?;

	let status = response.status();
	let response_text = response.text().await?;

	if !status.is_success() {
		return Err(anyhow::anyhow!(
			"Azure OpenAI LLM error ({}): {}",
			status,
			&response_text[..response_text.len().min(500)]
		));
	}

	let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

	response_json["choices"][0]["message"]["content"]
		.as_str()
		.map(|s| s.to_string())
		.ok_or_else(|| anyhow::anyhow!("Azure OpenAI LLM response missing content"))
}

/// Check if a model string is for the Azure LLM provider.
pub fn is_azure_llm(model_str: &str) -> bool {
	let provider = model_str.split(':').next().unwrap_or("");
	provider.eq_ignore_ascii_case("azure") || provider.eq_ignore_ascii_case("azure_openai")
}
