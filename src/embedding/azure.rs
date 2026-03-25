// Azure OpenAI Embedding Provider
//
// Provides embedding generation via Azure OpenAI Service endpoints.
// Supports text-embedding-3-small (1536d) and text-embedding-3-large (3072d).
//
// Required environment variables:
//   AZURE_OPENAI_API_KEY    - Azure OpenAI API key
//   AZURE_OPENAI_ENDPOINT   - Azure OpenAI endpoint URL (e.g., https://my-resource.openai.azure.com)
//
// Config format: "azure:text-embedding-3-large"

use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::sync::LazyLock;
use std::time::Duration;

use octolib::embedding::types::InputType;

static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
	Client::builder()
		.pool_max_idle_per_host(10)
		.pool_idle_timeout(Duration::from_secs(30))
		.timeout(Duration::from_secs(120))
		.connect_timeout(Duration::from_secs(10))
		.build()
		.expect("Failed to create HTTP client for Azure OpenAI")
});

const SUPPORTED_MODELS: &[(&str, usize)] = &[
	("text-embedding-3-small", 1536),
	("text-embedding-3-large", 3072),
	("text-embedding-ada-002", 1536),
];

/// Get the vector dimension for a given Azure model name.
pub fn get_dimension(model: &str) -> Result<usize> {
	SUPPORTED_MODELS
		.iter()
		.find(|(name, _)| *name == model)
		.map(|(_, dim)| *dim)
		.ok_or_else(|| {
			anyhow::anyhow!(
				"Unsupported Azure OpenAI model '{}'. Supported: {}",
				model,
				SUPPORTED_MODELS
					.iter()
					.map(|(n, _)| *n)
					.collect::<Vec<_>>()
					.join(", ")
			)
		})
}

/// Check if a model name is supported by the Azure provider.
pub fn is_supported(model: &str) -> bool {
	SUPPORTED_MODELS.iter().any(|(name, _)| *name == model)
}

fn get_credentials() -> Result<(String, String)> {
	let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| {
		anyhow::anyhow!(
			"AZURE_OPENAI_API_KEY environment variable not set. \
			 Set it to your Azure OpenAI API key."
		)
	})?;
	let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
		anyhow::anyhow!(
			"AZURE_OPENAI_ENDPOINT environment variable not set. \
			 Set it to your Azure OpenAI endpoint (e.g., https://my-resource.openai.azure.com)"
		)
	})?;
	Ok((api_key, endpoint.trim_end_matches('/').to_string()))
}

/// Generate embeddings for a single text using Azure OpenAI.
pub async fn generate_embedding(text: &str, model: &str) -> Result<Vec<f32>> {
	let results = generate_embeddings_batch(vec![text.to_string()], model, InputType::None).await?;
	results
		.into_iter()
		.next()
		.ok_or_else(|| anyhow::anyhow!("Azure OpenAI returned empty embedding result"))
}

/// Generate embeddings for a batch of texts using Azure OpenAI.
///
/// Uses the Azure OpenAI REST API with api-version 2024-02-01.
/// The deployment name is derived from the model name.
pub async fn generate_embeddings_batch(
	texts: Vec<String>,
	model: &str,
	input_type: InputType,
) -> Result<Vec<Vec<f32>>> {
	if texts.is_empty() {
		return Ok(Vec::new());
	}

	let (api_key, endpoint) = get_credentials()?;

	// Apply input_type prefix for asymmetric search (Azure doesn't have native input_type)
	let processed_texts: Vec<String> = texts
		.into_iter()
		.map(|text| input_type.apply_prefix(&text))
		.collect();

	// Azure OpenAI uses deployment names that typically match model names
	// URL format: {endpoint}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01
	let url = format!(
		"{}/openai/deployments/{}/embeddings?api-version=2024-02-01",
		endpoint, model
	);

	let request_body = json!({
		"input": processed_texts,
	});

	let response = HTTP_CLIENT
		.post(&url)
		.header("api-key", &api_key)
		.header("Content-Type", "application/json")
		.json(&request_body)
		.send()
		.await
		.map_err(|e| anyhow::anyhow!("Azure OpenAI request failed: {}", e))?;

	let status = response.status();
	let response_text = response.text().await?;

	if !status.is_success() {
		return Err(anyhow::anyhow!(
			"Azure OpenAI API error ({}): {}",
			status,
			response_text
		));
	}

	let response_json: serde_json::Value = serde_json::from_str(&response_text)
		.map_err(|e| anyhow::anyhow!("Failed to parse Azure OpenAI response: {}", e))?;

	let data = response_json["data"]
		.as_array()
		.ok_or_else(|| anyhow::anyhow!("Azure OpenAI response missing 'data' array"))?;

	let mut embeddings = Vec::with_capacity(data.len());
	for item in data {
		let embedding = item["embedding"]
			.as_array()
			.ok_or_else(|| anyhow::anyhow!("Azure OpenAI response missing 'embedding' array"))?
			.iter()
			.map(|v| v.as_f64().unwrap_or(0.0) as f32)
			.collect::<Vec<f32>>();
		embeddings.push(embedding);
	}

	// Azure returns results sorted by index, but let's ensure correct ordering
	let mut sorted_embeddings = vec![Vec::new(); embeddings.len()];
	for (i, item) in data.iter().enumerate() {
		let index = item["index"].as_u64().unwrap_or(i as u64) as usize;
		if index < sorted_embeddings.len() {
			sorted_embeddings[index] = embeddings[i].clone();
		}
	}

	Ok(sorted_embeddings)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_get_dimension() {
		assert_eq!(get_dimension("text-embedding-3-small").unwrap(), 1536);
		assert_eq!(get_dimension("text-embedding-3-large").unwrap(), 3072);
		assert_eq!(get_dimension("text-embedding-ada-002").unwrap(), 1536);
		assert!(get_dimension("nonexistent-model").is_err());
	}

	#[test]
	fn test_is_supported() {
		assert!(is_supported("text-embedding-3-large"));
		assert!(is_supported("text-embedding-3-small"));
		assert!(!is_supported("gpt-4"));
	}
}
