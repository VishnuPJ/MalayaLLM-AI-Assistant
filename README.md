## Steps to Run

1) Install llama.cpp and run llama-server ( Use the following instructions)
2) Install requirements.txt and run Malayalalm_AI_Assistant.py
   
## How to run GGUF

  - #### llama.cpp Web Server
    - The web server is a lightweight HTTP server that can be used to serve local models and easily connect them to existing clients.
  - #### Building llama.cpp
    - To build `llama.cpp` locally, follow the instructions provided in the [build documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md).
  - #### Running llama.cpp as a Web Server
    - Once you have built `llama.cpp`, you can run it as a web server. Below is an example of how to start the server:
        ```sh
        llama-server.exe -m gemma_2_9b_instruction.Q4_K_M.gguf -ngl 42 -c 128 -n 100
        ```
  - #### Accessing the Web UI
    - After starting the server, you can access the basic web UI via your browser at the following address:
      [http://localhost:8080](http://localhost:8080)
<img src="https://cdn-uploads.huggingface.co/production/uploads/64e65800e44b2668a56f9731/te7d5xjMrtk6RDMEAxmCy.png" alt="Baby MalayaLLM" width="600" height="auto">
   
# Model description
The MalayaLLM models have been improved and customized expanding upon the groundwork laid by the original Gemma-2-9B model.

- **Model type:** A 9B Gemma-2 finetuned model on Malayalam tokens.
- **Source Model:** [MalayaLLM_Gemma_2_9B_Base_V1.0](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_2_9B_Base_V1.0)
- **Instruct Model:** [MalayaLLM_Gemma_2_9B_Instruct_V1.0](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0)
- **GGUF Model:** [MalayaLLM_Gemma_2_9B_Instruct_V1.0_GGUF](https://huggingface.co/VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0_GGUF)

# Old Model
Gemma-7B trained model is here :[MalayaLLM:Gemma-7B](https://huggingface.co/collections/VishnuPJ/malayallm-malayalam-gemma-7b-66851df5e809bed18c2abd25)

##  Demo Video

<video controls autoplay src=""></video>

