use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;

use crate::config::AppConfig;
use crate::inference::llm::{LlmBackend, LlmEngine};
use crate::models::{ModelCategory, ModelDef, MODEL_CATALOG};

// ============================================================================
// Events
// ============================================================================

enum AppEvent {
    Token(String),
    GenerationComplete(String),
    ModelLoaded(String, Arc<Mutex<LlmEngine>>),
    ModelLoadError(String),
    DownloadProgress(String, String),
    DownloadComplete(String),
    DownloadError(String, String),
    AsrResult(String),
    AsrError(String),
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivePanel {
    Chat,
    Models,
    ASR,
    TTS,
    ImageGen,
}

// ============================================================================
// Chat Message
// ============================================================================

#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

// ============================================================================
// App State
// ============================================================================

pub struct MofaApp {
    // Event system
    event_tx: mpsc::Sender<AppEvent>,
    event_rx: mpsc::Receiver<AppEvent>,

    // Navigation
    active_panel: ActivePanel,

    // Config
    config: AppConfig,

    // Chat state
    messages: Vec<ChatMessage>,
    input: String,
    is_generating: bool,
    current_response: Arc<Mutex<String>>,

    // Model state
    loaded_model_id: Option<String>,
    loaded_engine: Option<Arc<Mutex<LlmEngine>>>,
    is_loading_model: bool,
    downloading_models: HashMap<String, String>,

    // ASR state
    asr_text: String,
    is_recording: bool,
    audio_samples: Arc<Mutex<Vec<f32>>>,

    // Status
    status_message: String,
}

impl MofaApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (event_tx, event_rx) = mpsc::channel();
        let mut config = AppConfig::load(None);
        config.scan_models_dir();

        MofaApp {
            event_tx,
            event_rx,
            active_panel: ActivePanel::Chat,
            config,
            messages: Vec::new(),
            input: String::new(),
            is_generating: false,
            current_response: Arc::new(Mutex::new(String::new())),
            loaded_model_id: None,
            loaded_engine: None,
            is_loading_model: false,
            downloading_models: HashMap::new(),
            asr_text: String::new(),
            is_recording: false,
            audio_samples: Arc::new(Mutex::new(Vec::new())),
            status_message: "Ready. Select a model to begin.".to_string(),
        }
    }

    fn process_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::Token(token) => {
                    if let Ok(mut resp) = self.current_response.lock() {
                        resp.push_str(&token);
                    }
                    // Update last message
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == "assistant" {
                            if let Ok(resp) = self.current_response.lock() {
                                last.content = resp.clone();
                            }
                        }
                    }
                }
                AppEvent::GenerationComplete(text) => {
                    self.is_generating = false;
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == "assistant" {
                            last.content = text;
                        }
                    }
                    self.status_message = "Generation complete.".to_string();
                }
                AppEvent::ModelLoaded(id, engine) => {
                    self.loaded_model_id = Some(id.clone());
                    self.loaded_engine = Some(engine);
                    self.is_loading_model = false;
                    self.status_message = format!("Model '{}' loaded.", id);
                }
                AppEvent::ModelLoadError(e) => {
                    self.is_loading_model = false;
                    self.status_message = format!("Error: {}", e);
                }
                AppEvent::DownloadProgress(id, msg) => {
                    self.downloading_models.insert(id, msg);
                }
                AppEvent::DownloadComplete(id) => {
                    self.downloading_models.remove(&id);
                    self.config.scan_models_dir();
                    self.status_message = format!("Model '{}' downloaded.", id);
                }
                AppEvent::DownloadError(id, e) => {
                    self.downloading_models.remove(&id);
                    self.status_message = format!("Download failed for '{}': {}", id, e);
                }
                AppEvent::AsrResult(text) => {
                    self.asr_text = text;
                    self.status_message = "Transcription complete.".to_string();
                }
                AppEvent::AsrError(e) => {
                    self.status_message = format!("ASR error: {}", e);
                }
                AppEvent::Error(e) => {
                    self.status_message = format!("Error: {}", e);
                }
            }
        }
    }

    fn send_message(&mut self) {
        if self.input.trim().is_empty() || self.is_generating {
            return;
        }

        let engine = match &self.loaded_engine {
            Some(e) => e.clone(),
            None => {
                self.status_message = "No model loaded. Load a model first.".to_string();
                return;
            }
        };

        let user_msg = self.input.trim().to_string();
        self.input.clear();

        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_msg.clone(),
        });
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
        });

        self.is_generating = true;
        *self.current_response.lock().unwrap() = String::new();

        let messages: Vec<(String, String)> = self
            .messages
            .iter()
            .filter(|m| m.role != "assistant" || !m.content.is_empty())
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        let tx = self.event_tx.clone();

        std::thread::spawn(move || {
            let mut eng = engine.lock().unwrap();
            match eng.chat(&messages, 2048, 0.7) {
                Ok(result) => {
                    let _ = tx.send(AppEvent::GenerationComplete(result.text));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::Error(e));
                }
            }
        });
    }

    fn load_model(&mut self, model_id: &str) {
        if self.is_loading_model {
            return;
        }

        // Find model in config
        let entry = match self.config.models.iter().find(|m| m.id == model_id) {
            Some(e) => e.clone(),
            None => {
                self.status_message = format!("Model '{}' not found.", model_id);
                return;
            }
        };

        // Determine backend
        let backend = match detect_backend(&entry.model_type) {
            Some(b) => b,
            None => {
                self.status_message = format!(
                    "Unknown model type '{}'. Cannot load.",
                    entry.model_type
                );
                return;
            }
        };

        self.is_loading_model = true;
        self.status_message = format!("Loading '{}'...", model_id);

        let model_id_owned = model_id.to_string();
        let path = entry.path.clone();
        let tx = self.event_tx.clone();
        std::thread::spawn(move || {
            match LlmEngine::load(
                std::path::Path::new(&path),
                backend,
                &model_id_owned,
            ) {
                Ok(engine) => {
                    let engine = Arc::new(Mutex::new(engine));
                    let _ = tx.send(AppEvent::ModelLoaded(model_id_owned, engine));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::ModelLoadError(e));
                }
            }
        });
    }

    fn download_model(&mut self, catalog_entry: &ModelDef) {
        let repo_id = catalog_entry.repo_id.to_string();
        let model_id = catalog_entry.id.to_string();
        let models_dir = self.config.models_dir.clone();
        let tx = self.event_tx.clone();

        self.downloading_models
            .insert(model_id.clone(), "Starting...".to_string());

        std::thread::spawn(move || {
            let tx2 = tx.clone();
            let id2 = model_id.clone();
            let progress_cb = move |msg: &str| {
                let _ = tx2.send(AppEvent::DownloadProgress(id2.clone(), msg.to_string()));
            };

            match crate::download::download_model(
                &repo_id,
                std::path::Path::new(&models_dir),
                Some(&progress_cb),
            ) {
                Ok(_entry) => {
                    let _ = tx.send(AppEvent::DownloadComplete(model_id));
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::DownloadError(model_id, e));
                }
            }
        });
    }

    // ========================================================================
    // UI Drawing
    // ========================================================================

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("MOFA Local LLM");
        ui.separator();

        let panels = [
            (ActivePanel::Chat, "Chat"),
            (ActivePanel::Models, "Models"),
            (ActivePanel::ASR, "Speech"),
            (ActivePanel::TTS, "Voice"),
            (ActivePanel::ImageGen, "Image"),
        ];

        for (panel, label) in &panels {
            let selected = self.active_panel == *panel;
            if ui.selectable_label(selected, *label).clicked() {
                self.active_panel = panel.clone();
            }
        }

        ui.separator();

        // Status
        if let Some(ref id) = self.loaded_model_id {
            ui.label(format!("Loaded: {}", id));
        } else {
            ui.colored_label(egui::Color32::GRAY, "No model loaded");
        }
    }

    fn draw_chat_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Chat");

        if self.loaded_model_id.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("Load a model from the Models panel to start chatting.");
            });
            return;
        }

        // Messages area
        let available = ui.available_height() - 60.0;
        egui::ScrollArea::vertical()
            .max_height(available)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for msg in &self.messages {
                    let (prefix, color) = if msg.role == "user" {
                        ("You", egui::Color32::LIGHT_BLUE)
                    } else {
                        ("AI", egui::Color32::LIGHT_GREEN)
                    };

                    ui.horizontal(|ui| {
                        ui.colored_label(color, format!("{}:", prefix));
                        ui.label(&msg.content);
                    });
                    ui.add_space(4.0);
                }

                if self.is_generating {
                    ui.spinner();
                }
            });

        ui.separator();

        // Input area
        ui.horizontal(|ui| {
            let response = ui.add_sized(
                [ui.available_width() - 60.0, 30.0],
                egui::TextEdit::singleline(&mut self.input)
                    .hint_text("Type a message... (Enter to send)"),
            );

            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.send_message();
            }

            if ui
                .add_enabled(!self.is_generating, egui::Button::new("Send"))
                .clicked()
            {
                self.send_message();
            }
        });
    }

    fn draw_models_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Model Management");
        ui.separator();

        // Downloaded models
        ui.label(egui::RichText::new("Downloaded Models").strong());
        ui.add_space(4.0);

        let models = self.config.models.clone();
        if models.is_empty() {
            ui.colored_label(egui::Color32::GRAY, "No models downloaded yet.");
        } else {
            for model in &models {
                ui.horizontal(|ui| {
                    let is_loaded = self.loaded_model_id.as_deref() == Some(&model.id);

                    if is_loaded {
                        ui.colored_label(egui::Color32::GREEN, "●");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "○");
                    }

                    ui.label(&model.id);

                    if let Some(size) = model.size_bytes {
                        ui.colored_label(
                            egui::Color32::GRAY,
                            format!("({})", format_bytes(size)),
                        );
                    }

                    if !is_loaded && !self.is_loading_model {
                        if ui.button("Load").clicked() {
                            self.load_model(&model.id);
                        }
                    }
                });
            }
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);

        // Model catalog
        ui.label(egui::RichText::new("Available Models (Download)").strong());
        ui.add_space(4.0);

        for category in &[
            ModelCategory::LLM,
            ModelCategory::VLM,
            ModelCategory::ASR,
            ModelCategory::TTS,
            ModelCategory::ImageGen,
        ] {
            let models_in_cat = ModelDef::by_category(*category);
            if models_in_cat.is_empty() {
                continue;
            }

            ui.label(egui::RichText::new(format!("{}", category)).strong().small());

            for def in &models_in_cat {
                ui.horizontal(|ui| {
                    let is_downloaded = self
                        .config
                        .models
                        .iter()
                        .any(|m| m.repo_id == def.repo_id || m.id == def.id);

                    let is_downloading = self.downloading_models.contains_key(def.id);

                    ui.label(def.name);
                    ui.colored_label(egui::Color32::GRAY, def.size_hint);

                    if is_downloaded {
                        ui.colored_label(egui::Color32::GREEN, "Downloaded");
                    } else if is_downloading {
                        if let Some(msg) = self.downloading_models.get(def.id) {
                            ui.spinner();
                            ui.colored_label(egui::Color32::YELLOW, msg);
                        }
                    } else if ui.button("Download").clicked() {
                        self.download_model(def);
                    }
                });
            }
            ui.add_space(4.0);
        }
    }

    fn draw_asr_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Speech Recognition");
        ui.separator();

        ui.label("Upload or record audio to transcribe.");
        ui.add_space(8.0);

        if !self.asr_text.is_empty() {
            ui.group(|ui| {
                ui.label("Transcription:");
                ui.label(&self.asr_text);
                if ui.button("Copy to Chat").clicked() {
                    self.input = self.asr_text.clone();
                    self.active_panel = ActivePanel::Chat;
                }
            });
        }
    }

    fn draw_tts_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Voice Synthesis");
        ui.separator();
        ui.label("GPT-SoVITS voice cloning - download the model first from the Models panel.");
    }

    fn draw_image_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Image Generation");
        ui.separator();
        ui.label("Z-Image / FLUX.2-klein image generation - download a model first from the Models panel.");
    }
}

impl eframe::App for MofaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_events();

        if self.is_generating {
            ctx.request_repaint();
        }

        // Sidebar
        egui::SidePanel::left("sidebar")
            .resizable(false)
            .default_width(160.0)
            .show(ctx, |ui| {
                self.draw_sidebar(ui);
            });

        // Status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.is_loading_model {
                    ui.spinner();
                }
                ui.label(&self.status_message);
            });
        });

        // Main panel
        egui::CentralPanel::default().show(ctx, |ui| match self.active_panel {
            ActivePanel::Chat => self.draw_chat_panel(ui),
            ActivePanel::Models => self.draw_models_panel(ui),
            ActivePanel::ASR => self.draw_asr_panel(ui),
            ActivePanel::TTS => self.draw_tts_panel(ui),
            ActivePanel::ImageGen => self.draw_image_panel(ui),
        });
    }
}

fn detect_backend(model_type: &str) -> Option<LlmBackend> {
    match model_type {
        "qwen2" => Some(LlmBackend::Qwen2),
        "qwen3" | "qwen" => Some(LlmBackend::Qwen3),
        "mistral" => Some(LlmBackend::Mistral),
        "glm4" | "chatglm" => Some(LlmBackend::Glm4),
        "mixtral" => Some(LlmBackend::Mixtral),
        "minicpm" | "minicpm4" => Some(LlmBackend::MiniCpmSala),
        _ => None,
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{}KB", bytes / 1000)
    }
}
