# Use very small model for testing purpose. Recommened to use llama-2-7b-chat.Q4_K_M.gguf model to test the chatbot.
# huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q3_K_S.gguf --local-dir ./llama2-chat-7b --local-dir-use-symlinks False
huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf --local-dir ./phi-2 --local-dir-use-symlinks False