// External source management for multi-source indexing.
//
// Manages URL-based documentation sources that can be indexed alongside code.
// Sources are stored per-project in a sources.toml file within the project storage.

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

static HTTP_CLIENT: std::sync::LazyLock<Client> = std::sync::LazyLock::new(|| {
	Client::builder()
		.timeout(Duration::from_secs(30))
		.connect_timeout(Duration::from_secs(10))
		.user_agent("octocode/0.12.2")
		.build()
		.expect("Failed to create HTTP client for source fetching")
});

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
	Url,
	Sitemap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
	pub name: String,
	pub source_type: SourceType,
	pub url: String,
	/// Optional CSS selector to extract specific content (e.g., "article", "main", ".docs-content")
	#[serde(default)]
	pub selector: Option<String>,
	/// Maximum depth for sitemap crawling
	#[serde(default = "default_max_depth")]
	pub max_depth: usize,
	/// When this source was last indexed (Unix timestamp)
	#[serde(default)]
	pub last_indexed: u64,
}

fn default_max_depth() -> usize {
	10
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourcesConfig {
	#[serde(default)]
	pub sources: HashMap<String, Source>,
}

impl SourcesConfig {
	/// Load sources config from project storage
	pub fn load(project_path: &Path) -> Result<Self> {
		let config_path = Self::get_path(project_path)?;
		if config_path.exists() {
			let content = std::fs::read_to_string(&config_path)?;
			Ok(toml::from_str(&content)?)
		} else {
			Ok(Self::default())
		}
	}

	/// Save sources config to project storage
	pub fn save(&self, project_path: &Path) -> Result<()> {
		let config_path = Self::get_path(project_path)?;
		if let Some(parent) = config_path.parent() {
			if !parent.exists() {
				std::fs::create_dir_all(parent)?;
			}
		}
		let content = toml::to_string_pretty(self)?;
		std::fs::write(config_path, content)?;
		Ok(())
	}

	fn get_path(project_path: &Path) -> Result<std::path::PathBuf> {
		let storage_path = crate::storage::get_project_storage_path(project_path)?;
		Ok(storage_path.join("sources.toml"))
	}

	pub fn add_source(&mut self, source: Source) {
		self.sources.insert(source.name.clone(), source);
	}

	pub fn remove_source(&mut self, name: &str) -> Option<Source> {
		self.sources.remove(name)
	}

	pub fn list_sources(&self) -> Vec<&Source> {
		self.sources.values().collect()
	}
}

/// Fetch a URL and return its content as cleaned markdown text.
pub async fn fetch_url_as_markdown(url: &str) -> Result<String> {
	let response = HTTP_CLIENT
		.get(url)
		.send()
		.await
		.map_err(|e| anyhow::anyhow!("Failed to fetch URL '{}': {}", url, e))?;

	let status = response.status();
	if !status.is_success() {
		return Err(anyhow::anyhow!(
			"HTTP {} fetching URL '{}'",
			status,
			url
		));
	}

	let content_type = response
		.headers()
		.get("content-type")
		.and_then(|v| v.to_str().ok())
		.unwrap_or("")
		.to_lowercase();

	let body = response.text().await?;

	if content_type.contains("text/html") || content_type.contains("application/xhtml") {
		Ok(html_to_markdown(&body))
	} else {
		// Already text/markdown or plain text
		Ok(body)
	}
}

/// Simple HTML to markdown converter.
/// Strips tags, preserves headings, links, code blocks, and paragraph structure.
fn html_to_markdown(html: &str) -> String {
	let mut result = String::with_capacity(html.len() / 2);
	let mut in_tag = false;
	let mut tag_name = String::new();
	let mut skip_content = false;
	let mut in_pre = false;
	let mut last_was_newline = false;

	let chars: Vec<char> = html.chars().collect();
	let len = chars.len();
	let mut i = 0;

	while i < len {
		let ch = chars[i];

		if ch == '<' {
			in_tag = true;
			tag_name.clear();
			i += 1;
			continue;
		}

		if in_tag {
			if ch == '>' {
				in_tag = false;
				let tag = tag_name.trim().to_lowercase();
				let is_closing = tag.starts_with('/');
				let clean_tag = tag.trim_start_matches('/').split_whitespace().next().unwrap_or("");

				match clean_tag {
					"script" | "style" | "nav" | "footer" | "header" => {
						skip_content = !is_closing;
					}
					"pre" | "code" => {
						if !is_closing {
							in_pre = true;
							result.push_str("\n```\n");
						} else {
							in_pre = false;
							result.push_str("\n```\n");
						}
					}
					"h1" => {
						if !is_closing {
							result.push_str("\n# ");
						} else {
							result.push('\n');
						}
					}
					"h2" => {
						if !is_closing {
							result.push_str("\n## ");
						} else {
							result.push('\n');
						}
					}
					"h3" => {
						if !is_closing {
							result.push_str("\n### ");
						} else {
							result.push('\n');
						}
					}
					"h4" | "h5" | "h6" => {
						if !is_closing {
							result.push_str("\n#### ");
						} else {
							result.push('\n');
						}
					}
					"p" | "div" | "section" | "article" => {
						if is_closing && !last_was_newline {
							result.push_str("\n\n");
							last_was_newline = true;
						}
					}
					"br" => {
						result.push('\n');
					}
					"li" => {
						if !is_closing {
							result.push_str("\n- ");
						}
					}
					"strong" | "b" => {
						result.push_str("**");
					}
					"em" | "i" => {
						result.push('*');
					}
					_ => {}
				}
			} else {
				tag_name.push(ch);
			}
			i += 1;
			continue;
		}

		if skip_content {
			i += 1;
			continue;
		}

		// Decode HTML entities
		if ch == '&' {
			let remaining: String = chars[i..].iter().take(10).collect();
			if remaining.starts_with("&amp;") {
				result.push('&');
				i += 5;
				continue;
			} else if remaining.starts_with("&lt;") {
				result.push('<');
				i += 4;
				continue;
			} else if remaining.starts_with("&gt;") {
				result.push('>');
				i += 4;
				continue;
			} else if remaining.starts_with("&quot;") {
				result.push('"');
				i += 6;
				continue;
			} else if remaining.starts_with("&#39;") || remaining.starts_with("&apos;") {
				result.push('\'');
				i += if remaining.starts_with("&#39;") { 5 } else { 6 };
				continue;
			} else if remaining.starts_with("&nbsp;") {
				result.push(' ');
				i += 6;
				continue;
			}
		}

		if in_pre {
			result.push(ch);
		} else if ch == '\n' || ch == '\r' {
			if !last_was_newline {
				result.push(' ');
			}
		} else {
			result.push(ch);
			last_was_newline = ch == '\n';
		}

		i += 1;
	}

	// Clean up: collapse multiple blank lines, trim
	let mut cleaned = String::new();
	let mut consecutive_newlines = 0;
	for ch in result.chars() {
		if ch == '\n' {
			consecutive_newlines += 1;
			if consecutive_newlines <= 2 {
				cleaned.push(ch);
			}
		} else {
			consecutive_newlines = 0;
			cleaned.push(ch);
		}
	}

	cleaned.trim().to_string()
}

/// Chunk markdown content by section headings.
/// Returns (title, content, level) tuples.
pub fn chunk_markdown_by_sections(content: &str) -> Vec<(String, String, usize)> {
	let mut sections = Vec::new();
	let mut current_title = String::new();
	let mut current_content = String::new();
	let mut current_level = 0;

	for line in content.lines() {
		let trimmed = line.trim();
		if let Some(heading) = parse_heading(trimmed) {
			// Save previous section if it has content
			if !current_content.trim().is_empty() || !current_title.is_empty() {
				sections.push((
					current_title.clone(),
					current_content.trim().to_string(),
					current_level,
				));
			}
			current_title = heading.1.to_string();
			current_level = heading.0;
			current_content.clear();
		} else {
			current_content.push_str(line);
			current_content.push('\n');
		}
	}

	// Don't forget the last section
	if !current_content.trim().is_empty() || !current_title.is_empty() {
		sections.push((current_title, current_content.trim().to_string(), current_level));
	}

	// Filter out empty sections
	sections.retain(|(_, content, _)| !content.is_empty());

	sections
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
	let trimmed = line.trim();
	if trimmed.starts_with('#') {
		let level = trimmed.chars().take_while(|c| *c == '#').count();
		if level <= 6 {
			let title = trimmed[level..].trim();
			if !title.is_empty() {
				return Some((level, title));
			}
		}
	}
	None
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_html_to_markdown_basic() {
		let html = "<h1>Title</h1><p>Hello world</p>";
		let md = html_to_markdown(html);
		assert!(md.contains("# Title"));
		assert!(md.contains("Hello world"));
	}

	#[test]
	fn test_html_to_markdown_strips_scripts() {
		let html = "<p>Keep this</p><script>remove();</script><p>And this</p>";
		let md = html_to_markdown(html);
		assert!(md.contains("Keep this"));
		assert!(md.contains("And this"));
		assert!(!md.contains("remove"));
	}

	#[test]
	fn test_html_to_markdown_code_blocks() {
		let html = "<pre><code>fn main() {}</code></pre>";
		let md = html_to_markdown(html);
		assert!(md.contains("```"));
		assert!(md.contains("fn main()"));
	}

	#[test]
	fn test_html_to_markdown_entities() {
		let html = "<p>&amp; &lt; &gt; &quot;</p>";
		let md = html_to_markdown(html);
		assert!(md.contains("& < > \""));
	}

	#[test]
	fn test_chunk_markdown_by_sections() {
		let md = "# Intro\nHello\n## Setup\nDo this\n## Usage\nRun that";
		let sections = chunk_markdown_by_sections(md);
		assert_eq!(sections.len(), 3);
		assert_eq!(sections[0].0, "Intro");
		assert_eq!(sections[1].0, "Setup");
		assert_eq!(sections[2].0, "Usage");
	}

	#[test]
	fn test_chunk_markdown_empty_sections_filtered() {
		let md = "# Title\n\n## Empty\n\n## Content\nSome text here";
		let sections = chunk_markdown_by_sections(md);
		// Empty section should be filtered
		assert!(sections.iter().all(|(_, content, _)| !content.is_empty()));
	}

	#[test]
	fn test_sources_config_roundtrip() {
		let mut config = SourcesConfig::default();
		config.add_source(Source {
			name: "rust-docs".to_string(),
			source_type: SourceType::Url,
			url: "https://doc.rust-lang.org/book/".to_string(),
			selector: Some("main".to_string()),
			max_depth: 5,
			last_indexed: 0,
		});

		let toml_str = toml::to_string_pretty(&config).unwrap();
		let parsed: SourcesConfig = toml::from_str(&toml_str).unwrap();
		assert_eq!(parsed.sources.len(), 1);
		assert_eq!(parsed.sources["rust-docs"].url, "https://doc.rust-lang.org/book/");
	}
}
