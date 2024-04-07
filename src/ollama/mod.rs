use std::pin::Pin;

use langchain_rust::{
    language_models::{llm::LLM, GenerateResult, LLMError},
    llm::{OpenAI, OpenAIConfig},
    schemas::{Message, StreamData},
};
use tokio_stream::Stream;

pub struct Ollama {
    pub model_name: String,
    pub model: OpenAI<OpenAIConfig>,
}

impl Ollama {
    pub fn new(model_name: &str) -> Self {
        let ollama = OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model_name);

        Self {
            model_name: model_name.to_string(),
            model: ollama,
        }
    }
}

impl LLM for Ollama {
    #[must_use]
    #[allow(clippy::type_complexity, clippy::type_repetition_in_bounds)]
    fn generate<'life0, 'life1, 'async_trait>(
        &'life0 self,
        messages: &'life1 [Message],
    ) -> ::core::pin::Pin<
        Box<
            dyn ::core::future::Future<Output = Result<GenerateResult, LLMError>>
                + ::core::marker::Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        self.model.generate(messages)
    }

    #[must_use]
    #[allow(clippy::type_complexity, clippy::type_repetition_in_bounds)]
    fn stream<'life0, 'life1, 'async_trait>(
        &'life0 self,
        _messages: &'life1 [Message],
    ) -> ::core::pin::Pin<
        Box<
            dyn ::core::future::Future<
                    Output = Result<
                        Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>,
                        LLMError,
                    >,
                > + ::core::marker::Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        self.model.stream(_messages)
    }
}
