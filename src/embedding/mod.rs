// Copyright 2025 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Re-export embedding functionality from octolib and add octocode-specific logic
//! Extends octolib with Azure OpenAI embedding provider support.

use crate::config::Config;
use anyhow::Result;

// Azure OpenAI provider (octocode-specific extension)
pub mod azure;

// Re-export core functionality from octolib::embedding
pub use octolib::embedding::{
	count_tokens, create_embedding_provider_from_parts, split_texts_into_token_limited_batches,
	truncate_output, EmbeddingProvider, InputType,
};

// Re-export types for backward compatibility
pub use octolib::embedding::types::{parse_provider_model, EmbeddingProviderType};

// Create a types module for backward compatibility
pub mod types {
	pub use octolib::embedding::types::*;
}

// Create a provider module for backward compatibility
pub mod provider {
	pub use octolib::embedding::provider::*;
}

/// Check if a provider:model string refers to the Azure provider.
fn is_azure_provider(provider_str: &str) -> bool {
	provider_str.eq_ignore_ascii_case("azure") || provider_str.eq_ignore_ascii_case("azure_openai")
}

/// Get vector dimension for any provider, including Azure (which octolib doesn't know about).
/// This is the octocode-level extension point for dimension resolution.
pub async fn get_vector_dimension_extended(provider_model: &str) -> Result<usize> {
	let (provider_str, model) = provider_model.split_once(':').ok_or_else(|| {
		anyhow::anyhow!("Invalid model format '{}': expected 'provider:model'", provider_model)
	})?;

	if is_azure_provider(provider_str) {
		return azure::get_dimension(model);
	}

	// Delegate to octolib for all other providers
	let (provider, model) = parse_provider_model(provider_model)?;
	let provider_impl = create_embedding_provider_from_parts(&provider, &model).await?;
	Ok(provider_impl.get_dimension())
}

/// Configuration for embedding generation (octocode-specific)
#[derive(Debug, Clone)]
pub struct EmbeddingGenerationConfig {
	/// Code embedding model (format: "provider:model")
	pub code_model: String,
	/// Text embedding model (format: "provider:model")
	pub text_model: String,
	/// Batch size for embedding generation
	pub batch_size: usize,
	/// Maximum tokens per batch
	pub max_tokens_per_batch: usize,
}

impl Default for EmbeddingGenerationConfig {
	fn default() -> Self {
		Self {
			code_model: "voyage:voyage-code-3".to_string(),
			text_model: "voyage:voyage-3.5-lite".to_string(),
			batch_size: 16,
			max_tokens_per_batch: 100_000,
		}
	}
}

/// Convert octocode Config to octocode EmbeddingGenerationConfig
impl From<&Config> for EmbeddingGenerationConfig {
	fn from(config: &Config) -> Self {
		Self {
			code_model: config.embedding.code_model.clone(),
			text_model: config.embedding.text_model.clone(),
			batch_size: config.index.embeddings_batch_size,
			max_tokens_per_batch: config.index.embeddings_max_tokens_per_batch,
		}
	}
}

/// Generate embeddings based on configured provider (supports provider:model format)
/// Handles Azure provider locally, delegates all others to octolib.
pub async fn generate_embeddings(
	contents: &str,
	is_code: bool,
	config: &Config,
) -> Result<Vec<f32>> {
	let embedding_config = EmbeddingGenerationConfig::from(config);

	let model_string = if is_code {
		&embedding_config.code_model
	} else {
		&embedding_config.text_model
	};

	let (provider, model) = if let Some((p, m)) = model_string.split_once(':') {
		(p, m)
	} else {
		return Err(anyhow::anyhow!("Invalid model format: {}", model_string));
	};

	// Intercept Azure provider — octolib doesn't know about it
	if is_azure_provider(provider) {
		return azure::generate_embedding(contents, model).await;
	}

	octolib::embedding::generate_embeddings(contents, provider, model).await
}

/// Generate batch embeddings based on configured provider (supports provider:model format)
/// Handles Azure provider locally, delegates all others to octolib.
pub async fn generate_embeddings_batch(
	texts: Vec<String>,
	is_code: bool,
	config: &Config,
	input_type: InputType,
) -> Result<Vec<Vec<f32>>> {
	let embedding_config = EmbeddingGenerationConfig::from(config);

	let model_string = if is_code {
		&embedding_config.code_model
	} else {
		&embedding_config.text_model
	};

	let (provider, model) = if let Some((p, m)) = model_string.split_once(':') {
		(p, m)
	} else {
		return Err(anyhow::anyhow!("Invalid model format: {}", model_string));
	};

	// Intercept Azure provider — octolib doesn't know about it
	if is_azure_provider(provider) {
		// Azure doesn't have native token-aware batching, so we split manually
		let batches = split_texts_into_token_limited_batches(
			texts,
			embedding_config.batch_size,
			embedding_config.max_tokens_per_batch,
		);

		let mut all_embeddings = Vec::new();
		for batch in batches {
			let mut batch_embeddings =
				azure::generate_embeddings_batch(batch, model, input_type.clone()).await?;
			all_embeddings.append(&mut batch_embeddings);
		}
		return Ok(all_embeddings);
	}

	octolib::embedding::generate_embeddings_batch(
		texts,
		provider,
		model,
		input_type,
		embedding_config.batch_size,
		embedding_config.max_tokens_per_batch,
	)
	.await
}

/// Search mode embeddings result (octocode-specific)
#[derive(Debug, Clone)]
pub struct SearchModeEmbeddings {
	pub code_embeddings: Option<Vec<f32>>,
	pub text_embeddings: Option<Vec<f32>>,
}

/// Generate embeddings for search based on mode - centralized logic to avoid duplication
/// Compatibility wrapper for octocode Config (octocode-specific)
pub async fn generate_search_embeddings(
	query: &str,
	mode: &str,
	config: &Config,
) -> Result<SearchModeEmbeddings> {
	match mode {
		"code" => {
			let embeddings = generate_embeddings(query, true, config).await?;
			Ok(SearchModeEmbeddings {
				code_embeddings: Some(embeddings),
				text_embeddings: None,
			})
		}
		"docs" | "text" => {
			let embeddings = generate_embeddings(query, false, config).await?;
			Ok(SearchModeEmbeddings {
				code_embeddings: None,
				text_embeddings: Some(embeddings),
			})
		}
		"all" => {
			let embedding_config = EmbeddingGenerationConfig::from(config);
			let code_model = &embedding_config.code_model;
			let text_model = &embedding_config.text_model;

			if code_model == text_model {
				let embeddings = generate_embeddings(query, true, config).await?;
				Ok(SearchModeEmbeddings {
					code_embeddings: Some(embeddings.clone()),
					text_embeddings: Some(embeddings),
				})
			} else {
				let code_embeddings = generate_embeddings(query, true, config).await?;
				let text_embeddings = generate_embeddings(query, false, config).await?;
				Ok(SearchModeEmbeddings {
					code_embeddings: Some(code_embeddings),
					text_embeddings: Some(text_embeddings),
				})
			}
		}
		_ => Err(anyhow::anyhow!(
			"Invalid search mode '{}'. Use 'all', 'code', 'docs', or 'text'.",
			mode
		)),
	}
}

/// Calculate a unique hash for content including file path (octocode-specific)
pub fn calculate_unique_content_hash(contents: &str, file_path: &str) -> String {
	use sha2::{Digest, Sha256};
	let mut hasher = Sha256::new();
	hasher.update(contents.as_bytes());
	hasher.update(file_path.as_bytes());
	format!("{:x}", hasher.finalize())
}

/// Calculate a unique hash for content including file path and line ranges (octocode-specific)
/// This ensures blocks are reindexed when their position changes in the file
pub fn calculate_content_hash_with_lines(
	contents: &str,
	file_path: &str,
	start_line: usize,
	end_line: usize,
) -> String {
	use sha2::{Digest, Sha256};
	let mut hasher = Sha256::new();
	hasher.update(contents.as_bytes());
	hasher.update(file_path.as_bytes());
	hasher.update(start_line.to_string().as_bytes());
	hasher.update(end_line.to_string().as_bytes());
	format!("{:x}", hasher.finalize())
}

/// Calculate content hash without file path (octocode-specific)
pub fn calculate_content_hash(contents: &str) -> String {
	use sha2::{Digest, Sha256};
	let mut hasher = Sha256::new();
	hasher.update(contents.as_bytes());
	format!("{:x}", hasher.finalize())
}
