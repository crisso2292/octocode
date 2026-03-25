// MCP tool provider for external documentation source management.
//
// Tools: add_source, remove_source, list_sources, index_source

use anyhow::Result;
use serde_json::{json, Value};
use tracing::{debug, info};

use crate::config::Config;
use crate::mcp::types::{McpError, McpTool};
use crate::sources::{self, Source, SourceType, SourcesConfig};

#[derive(Clone)]
pub struct SourcesProvider {
	config: Config,
	working_directory: std::path::PathBuf,
}

impl SourcesProvider {
	pub fn new(config: Config, working_directory: std::path::PathBuf) -> Self {
		Self {
			config,
			working_directory,
		}
	}

	pub fn get_tool_definitions() -> Vec<McpTool> {
		vec![
			McpTool {
				name: "add_source".to_string(),
				description: "Add an external documentation URL to be indexed alongside code. Supports web pages and doc sites. Content is fetched, converted to markdown, and embedded for semantic search.".to_string(),
				input_schema: json!({
					"type": "object",
					"properties": {
						"name": {
							"type": "string",
							"description": "Unique name for this source (e.g., 'react-docs', 'api-reference')",
							"minLength": 1,
							"maxLength": 100
						},
						"url": {
							"type": "string",
							"description": "URL to fetch and index (e.g., 'https://docs.rs/tokio/latest/tokio/')"
						},
						"type": {
							"type": "string",
							"description": "Source type: 'url' (single page) or 'sitemap' (multiple pages from sitemap.xml)",
							"enum": ["url", "sitemap"],
							"default": "url"
						},
					},
					"required": ["name", "url"],
					"additionalProperties": false
				}),
			},
			McpTool {
				name: "remove_source".to_string(),
				description: "Remove an external documentation source and its indexed content.".to_string(),
				input_schema: json!({
					"type": "object",
					"properties": {
						"name": {
							"type": "string",
							"description": "Name of the source to remove"
						},
					},
					"required": ["name"],
					"additionalProperties": false
				}),
			},
			McpTool {
				name: "list_sources".to_string(),
				description: "List all configured external documentation sources.".to_string(),
				input_schema: json!({
					"type": "object",
					"properties": {},
					"additionalProperties": false
				}),
			},
			McpTool {
				name: "index_source".to_string(),
				description: "Fetch and index a specific external documentation source. Re-indexes if already indexed.".to_string(),
				input_schema: json!({
					"type": "object",
					"properties": {
						"name": {
							"type": "string",
							"description": "Name of the source to index"
						},
					},
					"required": ["name"],
					"additionalProperties": false
				}),
			},
		]
	}

	pub async fn execute_add_source(&self, arguments: &Value) -> Result<String, McpError> {
		let name = arguments
			.get("name")
			.and_then(|v| v.as_str())
			.ok_or_else(|| {
				McpError::invalid_params("Missing required parameter 'name'", "add_source")
			})?
			.to_string();

		let url = arguments
			.get("url")
			.and_then(|v| v.as_str())
			.ok_or_else(|| {
				McpError::invalid_params("Missing required parameter 'url'", "add_source")
			})?
			.to_string();

		// Validate URL
		if !url.starts_with("http://") && !url.starts_with("https://") {
			return Err(McpError::invalid_params(
				"URL must start with http:// or https://",
				"add_source",
			));
		}

		let source_type = match arguments
			.get("type")
			.and_then(|v| v.as_str())
			.unwrap_or("url")
		{
			"url" => SourceType::Url,
			"sitemap" => SourceType::Sitemap,
			other => {
				return Err(McpError::invalid_params(
					format!("Invalid source type '{}': must be 'url' or 'sitemap'", other),
					"add_source",
				))
			}
		};

		let source = Source {
			name: name.clone(),
			source_type,
			url: url.clone(),
			selector: None,
			max_depth: 10,
			last_indexed: 0,
		};

		let mut sources_config = SourcesConfig::load(&self.working_directory).map_err(|e| {
			McpError::internal_error(format!("Failed to load sources config: {}", e), "add_source")
		})?;

		sources_config.add_source(source);
		sources_config
			.save(&self.working_directory)
			.map_err(|e| {
				McpError::internal_error(
					format!("Failed to save sources config: {}", e),
					"add_source",
				)
			})?;

		info!(name = %name, url = %url, "Added documentation source");
		Ok(format!(
			"Source '{}' added successfully (URL: {}). Use index_source to fetch and index it.",
			name, url
		))
	}

	pub async fn execute_remove_source(&self, arguments: &Value) -> Result<String, McpError> {
		let name = arguments
			.get("name")
			.and_then(|v| v.as_str())
			.ok_or_else(|| {
				McpError::invalid_params("Missing required parameter 'name'", "remove_source")
			})?;

		let mut sources_config =
			SourcesConfig::load(&self.working_directory).map_err(|e| {
				McpError::internal_error(
					format!("Failed to load sources config: {}", e),
					"remove_source",
				)
			})?;

		if sources_config.remove_source(name).is_none() {
			return Err(McpError::invalid_params(
				format!("Source '{}' not found", name),
				"remove_source",
			));
		}

		// Remove indexed content for this source from the store
		let original_dir = std::env::current_dir().map_err(|e| {
			McpError::internal_error(format!("Failed to get current dir: {}", e), "remove_source")
		})?;
		let _ = std::env::set_current_dir(&self.working_directory);

		if let Ok(store) = crate::store::Store::new().await {
			// Remove blocks whose path starts with "source://{name}/"
			let source_path_prefix = format!("source://{}/", name);
			if let Ok(indexed_paths) = store.get_all_indexed_file_paths().await {
				for path in indexed_paths {
					if path.starts_with(&source_path_prefix) {
						let _ = store.remove_blocks_by_path(&path).await;
					}
				}
			}
		}
		let _ = std::env::set_current_dir(&original_dir);

		sources_config
			.save(&self.working_directory)
			.map_err(|e| {
				McpError::internal_error(
					format!("Failed to save sources config: {}", e),
					"remove_source",
				)
			})?;

		info!(name = %name, "Removed documentation source");
		Ok(format!(
			"Source '{}' removed successfully and indexed content cleaned up.",
			name
		))
	}

	pub async fn execute_list_sources(&self, _arguments: &Value) -> Result<String, McpError> {
		let sources_config = SourcesConfig::load(&self.working_directory).map_err(|e| {
			McpError::internal_error(
				format!("Failed to load sources config: {}", e),
				"list_sources",
			)
		})?;

		let sources = sources_config.list_sources();
		if sources.is_empty() {
			return Ok("No external documentation sources configured. Use add_source to add one."
				.to_string());
		}

		let mut output = format!("DOCUMENTATION SOURCES ({})\n\n", sources.len());
		for source in sources {
			let indexed_status = if source.last_indexed > 0 {
				let dt = chrono::DateTime::from_timestamp(source.last_indexed as i64, 0)
					.map(|d| d.format("%Y-%m-%d %H:%M UTC").to_string())
					.unwrap_or_else(|| "unknown".to_string());
				format!("Last indexed: {}", dt)
			} else {
				"Not yet indexed".to_string()
			};

			output.push_str(&format!(
				"- {} ({:?})\n  URL: {}\n  {}\n\n",
				source.name, source.source_type, source.url, indexed_status
			));
		}

		Ok(output)
	}

	pub async fn execute_index_source(&self, arguments: &Value) -> Result<String, McpError> {
		let name = arguments
			.get("name")
			.and_then(|v| v.as_str())
			.ok_or_else(|| {
				McpError::invalid_params("Missing required parameter 'name'", "index_source")
			})?;

		let mut sources_config =
			SourcesConfig::load(&self.working_directory).map_err(|e| {
				McpError::internal_error(
					format!("Failed to load sources config: {}", e),
					"index_source",
				)
			})?;

		let source = sources_config
			.sources
			.get(name)
			.ok_or_else(|| {
				McpError::invalid_params(format!("Source '{}' not found", name), "index_source")
			})?
			.clone();

		debug!(name = %name, url = %source.url, "Fetching documentation source");

		// Fetch content
		let markdown = sources::fetch_url_as_markdown(&source.url)
			.await
			.map_err(|e| {
				McpError::internal_error(
					format!("Failed to fetch source '{}': {}", name, e),
					"index_source",
				)
			})?;

		// Chunk into sections
		let sections = sources::chunk_markdown_by_sections(&markdown);

		if sections.is_empty() {
			return Ok(format!(
				"Source '{}' fetched but no content sections found.",
				name
			));
		}

		// Change to working directory for store operations
		let original_dir = std::env::current_dir().map_err(|e| {
			McpError::internal_error(format!("Failed to get current dir: {}", e), "index_source")
		})?;
		let _ = std::env::set_current_dir(&self.working_directory);

		let store = crate::store::Store::new().await.map_err(|e| {
			let _ = std::env::set_current_dir(&original_dir);
			McpError::internal_error(format!("Failed to open store: {}", e), "index_source")
		})?;

		// Remove previous content for this source
		let source_path_prefix = format!("source://{}/", name);
		if let Ok(indexed_paths) = store.get_all_indexed_file_paths().await {
			for path in indexed_paths {
				if path.starts_with(&source_path_prefix) {
					let _ = store.remove_blocks_by_path(&path).await;
				}
			}
		}

		// Create document blocks from sections
		let mut doc_blocks = Vec::new();
		let mut line_offset = 0;
		for (title, content, level) in &sections {
			let line_count = content.lines().count();
			let path = format!("source://{}/{}", name, title.replace(' ', "_"));
			let hash = crate::embedding::calculate_unique_content_hash(&content, &path);

			doc_blocks.push(crate::store::DocumentBlock {
				path,
				title: title.clone(),
				content: content.clone(),
				context: vec![source.url.clone()],
				level: *level,
				start_line: line_offset,
				end_line: line_offset + line_count,
				hash,
				distance: None,
			});
			line_offset += line_count;
		}

		// Generate embeddings
		let texts: Vec<String> = doc_blocks.iter().map(|b| b.content.clone()).collect();
		let embeddings = crate::embedding::generate_embeddings_batch(
			texts,
			false, // Use text model for docs
			&self.config,
			crate::embedding::InputType::Document,
		)
		.await
		.map_err(|e| {
			let _ = std::env::set_current_dir(&original_dir);
			McpError::internal_error(
				format!("Failed to generate embeddings: {}", e),
				"index_source",
			)
		})?;

		// Store in LanceDB
		store
			.store_document_blocks(&doc_blocks, &embeddings)
			.await
			.map_err(|e| {
				let _ = std::env::set_current_dir(&original_dir);
				McpError::internal_error(
					format!("Failed to store document blocks: {}", e),
					"index_source",
				)
			})?;

		let _ = std::env::set_current_dir(&original_dir);

		// Update last_indexed timestamp
		if let Some(src) = sources_config.sources.get_mut(name) {
			src.last_indexed = std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.unwrap_or_default()
				.as_secs();
		}
		let _ = sources_config.save(&self.working_directory);

		info!(
			name = %name,
			sections = doc_blocks.len(),
			"Indexed documentation source"
		);

		Ok(format!(
			"Source '{}' indexed successfully: {} sections from {}",
			name,
			doc_blocks.len(),
			source.url
		))
	}
}
