pub mod config;
pub mod models;
pub mod download;
pub mod inference;
pub mod api;
pub mod gui;

// Library API for external use
pub mod lib_api;
pub use lib_api::LocalLLMClient;

// Re-export commonly used types
pub use inference::ChatResult;
pub use config::ModelEntry;