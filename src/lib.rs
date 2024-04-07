pub mod ollama;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use langchain_rust::language_models::llm::LLM;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_make_ollama() {
        let model_name = "gemma:latest";
        let ollama = ollama::Ollama::new(model_name);
        assert_eq!(ollama.model_name, model_name);
    }

    #[tokio::test]
    async fn test_generate() {
        let model_name = "gemma:latest";
        let ollama = ollama::Ollama::new(model_name);
        let result = ollama.invoke("Hi, can you ").await;
        println!("{:?}", result);
        assert!(result.is_ok());
    }
}
