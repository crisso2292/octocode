// LLM-Generated Docstring Enhancement (Greptile's approach)
//
// Generates natural language descriptions for code blocks using an LLM.
// Embedding these descriptions alongside code yields ~12% better retrieval
// than embedding raw code alone, because search queries are in natural language
// and match better against natural language descriptions.
//
// How it works:
// 1. During indexing, code blocks above a size threshold are sent to an LLM
// 2. The LLM generates a 1-2 sentence description of what the code does
// 3. The description is prepended to the code content ONLY for embedding
// 4. The stored content remains the original code (for display)
//
// This is opt-in via config: [index.docstrings] enabled = true

use crate::config::Config;
use crate::llm::{LlmClient, Message};
use crate::store::CodeBlock;
use anyhow::Result;
use tracing::{debug, warn};

/// Minimum lines for a code block to qualify for docstring generation.
/// Small blocks (imports, single-line declarations) don't benefit from docstrings.
const DEFAULT_MIN_LINES: usize = 10;

/// Maximum code blocks to process per LLM batch call.
const BATCH_SIZE: usize = 5;

/// Maximum characters of code to send to the LLM per block.
const MAX_CODE_CHARS: usize = 3000;

const SYSTEM_PROMPT: &str = r#"You are a code documentation expert. For each code block provided, write a single concise sentence (max 30 words) describing WHAT the code does and WHY it exists. Focus on the purpose and behavior, not implementation details.

Respond with one description per line, in the same order as the input blocks. Each line should be a plain sentence, no numbering, no markdown, no code formatting.

Example input:
```
BLOCK 1:
pub async fn retry_with_backoff<F, T>(f: F, max_retries: u32) -> Result<T>
where F: Fn() -> Future<Output = Result<T>> {
    for attempt in 0..max_retries {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) if attempt < max_retries - 1 => {
                sleep(Duration::from_millis(100 * 2u64.pow(attempt))).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

Example output:
Retries an async operation with exponential backoff, returning the first success or the final error after exhausting all attempts."#;

/// Generate LLM docstrings for code blocks that meet the size threshold.
/// Returns a Vec of Option<String> parallel to the input blocks.
/// None means the block was too small or docstring generation failed.
pub async fn generate_docstrings(
	blocks: &[CodeBlock],
	config: &Config,
) -> Result<Vec<Option<String>>> {
	let min_lines = DEFAULT_MIN_LINES;
	let mut docstrings: Vec<Option<String>> = vec![None; blocks.len()];

	// Identify blocks that qualify for docstring generation
	let qualifying_indices: Vec<usize> = blocks
		.iter()
		.enumerate()
		.filter(|(_, block)| {
			let line_count = block.content.lines().count();
			line_count >= min_lines
		})
		.map(|(i, _)| i)
		.collect();

	if qualifying_indices.is_empty() {
		debug!("No code blocks qualify for docstring generation (min {} lines)", min_lines);
		return Ok(docstrings);
	}

	debug!(
		"Generating docstrings for {}/{} code blocks",
		qualifying_indices.len(),
		blocks.len()
	);

	// Create LLM client (catch panics from provider factory)
	let llm_client = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		LlmClient::from_config(config)
	})) {
		Ok(Ok(client)) => client,
		Ok(Err(e)) => {
			warn!("Failed to create LLM client for docstrings: {}. Skipping.", e);
			return Ok(docstrings);
		}
		Err(_) => {
			warn!("LLM client creation panicked. Skipping docstring generation.");
			return Ok(docstrings);
		}
	};

	// Process in batches
	for chunk in qualifying_indices.chunks(BATCH_SIZE) {
		let mut user_content = String::new();
		for (batch_idx, &block_idx) in chunk.iter().enumerate() {
			let block = &blocks[block_idx];
			let code = if block.content.len() > MAX_CODE_CHARS {
				// Find the nearest char boundary at or before MAX_CODE_CHARS
				let mut end = MAX_CODE_CHARS;
				while !block.content.is_char_boundary(end) && end > 0 {
					end -= 1;
				}
				&block.content[..end]
			} else {
				&block.content
			};
			user_content.push_str(&format!(
				"BLOCK {}:\n```{}\n{}\n```\n\n",
				batch_idx + 1,
				block.language,
				code
			));
		}

		let messages = vec![
			Message::system(SYSTEM_PROMPT),
			Message::user(&user_content),
		];

		match llm_client.chat_completion(messages).await {
			Ok(response) => {
				let lines: Vec<&str> = response
					.lines()
					.map(|l| l.trim())
					.filter(|l| !l.is_empty())
					.collect();

				for (batch_idx, &block_idx) in chunk.iter().enumerate() {
					if let Some(desc) = lines.get(batch_idx) {
						let desc = desc.to_string();
						if desc.len() > 10 && desc.len() < 500 {
							docstrings[block_idx] = Some(desc);
						}
					}
				}
			}
			Err(e) => {
				warn!("Docstring generation failed for batch: {}. Continuing without docstrings.", e);
			}
		}
	}

	let generated = docstrings.iter().filter(|d| d.is_some()).count();
	debug!("Generated {} docstrings for {} qualifying blocks", generated, qualifying_indices.len());

	Ok(docstrings)
}

/// Prepend docstrings to code content for embedding.
/// Returns the enriched content strings that should be embedded
/// (original content is NOT modified — this is for embedding only).
pub fn enrich_contents_for_embedding(
	blocks: &[CodeBlock],
	docstrings: &[Option<String>],
) -> Vec<String> {
	blocks
		.iter()
		.zip(docstrings.iter())
		.map(|(block, docstring)| {
			let mut parts = Vec::new();

			// Prepend LLM docstring if available (Greptile's insight)
			if let Some(desc) = docstring {
				parts.push(format!("// Description: {}", desc));
			}

			// Add symbols
			for symbol in &block.symbols {
				parts.push(symbol.clone());
			}

			// Add code content
			parts.push(block.content.clone());

			parts.join("\n")
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_enrich_contents_with_docstring() {
		let blocks = vec![CodeBlock {
			path: "test.rs".to_string(),
			language: "rust".to_string(),
			content: "fn hello() { println!(\"hello\"); }".to_string(),
			symbols: vec!["hello".to_string()],
			start_line: 1,
			end_line: 1,
			hash: "abc".to_string(),
			distance: None,
		}];

		let docstrings = vec![Some("Prints a greeting message to stdout.".to_string())];

		let enriched = enrich_contents_for_embedding(&blocks, &docstrings);
		assert!(enriched[0].contains("// Description: Prints a greeting"));
		assert!(enriched[0].contains("fn hello()"));
	}

	#[test]
	fn test_enrich_contents_without_docstring() {
		let blocks = vec![CodeBlock {
			path: "test.rs".to_string(),
			language: "rust".to_string(),
			content: "fn hello() {}".to_string(),
			symbols: vec![],
			start_line: 1,
			end_line: 1,
			hash: "abc".to_string(),
			distance: None,
		}];

		let docstrings = vec![None];

		let enriched = enrich_contents_for_embedding(&blocks, &docstrings);
		assert!(!enriched[0].contains("Description"));
		assert!(enriched[0].contains("fn hello()"));
	}
}
