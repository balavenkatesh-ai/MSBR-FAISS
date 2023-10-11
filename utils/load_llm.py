from llama_cpp import Llama

def load_llm(model_type, model_path):
        if model_type == "LLaMA-7B":
                llm = Llama(model_path= model_path, 
                n_ctx= 4096,
                n_parts= -1,
                n_gpu_layers = 16,
                n_threads = 6,
                n_batch= 128,
                last_n_tokens_size = 64
                )
        
        elif model_type == "LLaMA-13B":
                llm = Llama(model_path= model_path, 
                n_ctx= 4096,
                n_parts= -1,
                n_gpu_layers = 16,
                n_threads = 6,
                n_batch= 128,
                last_n_tokens_size = 64
                )
        return llm