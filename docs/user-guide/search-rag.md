---
sidebar_position: 1
---

# Realtime information for the LLM

For a digital assistant to be truly useful, it requires access to realtime information.
Established information retrieval methods like [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) tend to work great with long-form knowledge, such as books and research papers, but generally tend to lack methods to access information that isn't found in those texts. To supplement this, we require a means of accessing up to date information.

The Internet, accessed using sophisticated search engines, is the perfect candidate for such an information source. Search Engines such as Google and Bing update information such as breaking news in real time. This behavior can be utilized by the LLMs to access up to date information in scenarios where RAG fails.

The LlamaEdge API server, along with the llama-core crate, provides application components that developers can reuse to  supplement their LLM applications with up to date information from the Internet. 
These features have been built into the [llama-core](https://github.com/llamaedge/llamaedge) crate and demonstrated with the [search-api-server](https://github.com/LlamaEdge/search-api-server) project. 
The result is an OpenAI compatible LLM service that performs internet search on the server-side. The client application can simply chat with it and have it supplement gaps in its trained knowledge with the content fetched from the internet at runtime.

The [rag-api-server](https://github.com/LlamaEdge/rag-api-server) also has the search feature that can be enabled on compilation with the `search` feature flag, which will enable it to perform internet search on user chat queries where RAG fails. 

## Prerequisites

*Note:* Running the Search Enabled RAG API Server and some of its dependencies entirely locally requires a hefty chunk of VRAM. Consider hosting the LLamaEdge Query Server on a separate machine if you run into VRAM issues. 

Install the [WasmEdge Runtime](https://github.com/WasmEdge/WasmEdge), our cross-platform LLM runtime.

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s
```

Download the pre-built binary for the LlamaEdge API server with RAG support.

## Requirements for the Search API Server

### Build it

The following commands will add the `wasm32-wasip1` target to your current `rustup` toolchain and build and copy the Search API Server to your current directory.

```
git clone --depth=1 https://github.com/LlamaEdge/search-api-server.git
cd search-api-server

# (Optional) Add the `wasm32-wasip1` target to the Rust toolchain
rustup target add wasm32-wasip1

# Build `search-api-server.wasm` with both internet search support
cargo build --target wasm32-wasip1 --release --release
cp target/wasm32-wasip1/release/search-api-server.wasm .
```

And the chatbot web UI for the API server.

```
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz
```

Download a chat model.

```
# The chat model is Llama3 8b chat
curl -LO https://huggingface.co/second-state/Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf
```

This chat model is what will be fed the search results from the internet searches performed. Use a model that fits your use case! The Llama3 8B model should be fine for regular chatting.

### Start the API server

Let's start the Search Enabled LlamaEdge Search API server on port 8080. The default search engine is Tavily. You will have to register for an API Key in order to make use of this API endpoint, and pass it with the `--api-key`.

```
wasmedge --dir .:.  --env llama_log="info" \
	--nn-preload default:GGML:AUTO:Llama-2-7b-chat-hf-Q5_K_M.gguf \
	./search-api-server.wasm \
	--ctx-size 4096,384 \
	--prompt-template llama-2-chat \
	--model-name Llama-2-7b-chat-hf-Q5_K_M \
	--api-key "tvly-xxx" 
```

The CLI arguments are self-explanatory.
Notice that those arguments are different from the [llama-api-server.wasm](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server) app.

* The `--nn-proload` loads two models we just downloaded. The chat model is named `default` and the embedding model is named `embedding` .
* The `search-api-server.wasm` is the API server app. It is written in Rust using LlamaEdge SDK, and is already compiled to cross-platform Wasm binary.
* The `--model-name` specifies the names of those two models so that API calls can be routed to specific models.
* The `--ctx-size` specifies the max input size for each of those two models listed in `--model-name`.
* The `--batch-size` specifies the batch processing size for each of those two models listed in `--model-name`. This parameter has a large impact on the RAM use of the API server.
* The `--api-key` is the API key to be supplied to the endpoint, if the endpoint in use supports it.

### Chat with supplemental real-time knowledge

Just go to `http://localhost:8080/` from your web browser, and you will see a chatbot UI web page. Any question you ask here will be answered with a supplemental internet search!

> While the LLM runs locally, any queries you make to the chatbot will go to the currently selected search API endpoint. This endpoint is what performs the internet search. The default is Tavily, which is a proprietary search API. You will have to register for an API Key in order to make use of this API endpoint, and pass it with the `--api-key` as mentioned above.

Or, you can access it via the API. 

```
curl -X POST http://localhost:8080/v1/chat/completions \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the new OpenAI o1 model?"}], "model":"Llama-2-7b-chat-hf-Q5_K_M"}'

{
  "id": "chatcmpl-edd22922-f7aa-4866-8b18-e13c322faa2b",
  "object": "chat.completion",
  "created": 1727463938,
  "model": "Llama-2-7b-chat-hf-Q5_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "The OpenAI O1 model is a recent advancement in the field of natural language processing (NLP) by OpenAI, a leading AI research organization. It is designed to handle complex and diverse tasks, such as question answering, text classification, and language translation.\n\nThe OpenAI O1 model is based on a novel architecture that combines the strengths of both transformer-based and recurrent neural network (RNN) architectures. The transformer architecture is well-suited for handling sequential data, while the RNN architecture can capture long-term dependencies in sequences. By combining these two architectures, the OpenAI O1 model can effectively handle a wide range of NLP tasks.\n\nSome key features of the OpenAI O1 model include:\n\n1. Multitask learning: The model is trained on multiple tasks simultaneously, allowing it to learn shared representations across tasks and improve overall performance.\n2. Attention mechanism: The model uses an attention mechanism to focus on specific parts of the input sequence, allowing it to selectively attend to the most relevant information for each task.\n3. Encoder-decoder architecture: The model uses an encoder-decoder architecture, which allows it to effectively handle sequential data and generate coherent output.\n4. Efficient use of parameters: The model uses a novel parameterization scheme that allows it to efficiently learn from large datasets while using fewer parameters than other state-of-the-art models.\n\nThe OpenAI O1 model has achieved state-of-the-art results on several benchmark datasets, including the GLUE benchmark, which is a collection of tasks that assess a model's ability to perform various NLP tasks such as question answering, sentiment analysis, and text classification.\n\nOverall, the OpenAI O1 model represents a significant advancement in the field of NLP, and it has the potential to enable new applications and use cases for NLP models.",
        "role": "assistant"
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 37,
    "completion_tokens": 409,
    "total_tokens": 446
  }
}
```

### Requirements for the search-enabled RAG API Server

#### Build it

The following commands will add the `wasm32-wasip1` target to your current `rustup` toolchain and build and copy the Search API Server to your current directory.

```
git clone --depth=1 https://github.com/LlamaEdge/rag-api-server.git
cd rag-api-server

# (Optional) Add the `wasm32-wasi` target to the Rust toolchain
rustup target add wasm32-wasip1

# Build `rag-api-server.wasm` with both internet search support
cargo build --target wasm32-wasip1 --release --features full
cp target/wasm32-wasip1/release/rag-api-server.wasm .
```

And the chatbot web UI for the API server.

```
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz
```

Download a chat model and an embedding model.

```
# The chat model is Llama3 8b chat
curl -LO https://huggingface.co/second-state/Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf

# The embedding model is nomic-embed-text-v1.5
curl -LO https://huggingface.co/second-state/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf
```

The embedding model is a special kind of LLM that turns sentences into vectors. The vectors can then be stored in a vector database and searched later. When the sentences are from a body of text that represents a knowledge domain, that vector database becomes our RAG knowledge base.

### Prepare a vector database

By default, we use Qdrant as the vector database. You can start a Qdrant instance on your server using Docker. The following command starts it in the background.

```
mkdir qdrant_storage
mkdir qdrant_snapshots

nohup docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    -v $(pwd)/qdrant_snapshots:/qdrant/snapshots:z \
    qdrant/qdrant
```

Delete the `default` collection if it exists.

```
curl -X DELETE 'http://localhost:6333/collections/default'
```

Next, download a knowledge base, which is in the form of a vector snapshot. For example, here is an vector snapshot
created from a guidebook for Paris. It is a 768-dimension vector collection created by the embedding model [nomic-embed-text](https://huggingface.co/second-state/Nomic-embed-text-v1.5-Embedding-GGUF), which you have already downloaded.

```
curl -LO https://huggingface.co/datasets/gaianet/paris/resolve/main/paris_768_nomic-embed-text-v1.5-f16.snapshot
```

> You can create your own vector snapshots using tools discussed in the next several chapters.

Import the vector snapshot file into the local Qdrant database server's `default` collection.

```
curl -s -X POST http://localhost:6333/collections/default/snapshots/upload?priority=snapshot \
    -H 'Content-Type:multipart/form-data' \
    -F 'snapshot=@paris_768_nomic-embed-text-v1.5-f16.snapshot'
```

### Start the LlamaEdge Query Server

The search functionality utilized by the search enabled RAG API Server is powered by the LlamaEdge Query Server. Just as the RAG portion is handled by Qdrant, the internet search portion of the RAG API Server is handled by the LlamaEdge Query Server.

#### Build it

```
git clone --depth=1 https://github.com/LlamaEdge/llamaedge-query-server.git
cd llamaedge-query-server
cargo build --release --target wasm32-wasip1
cp target/wasm32-wasip1/release/llamaedge-query-server.wasm .
```

We then download the [Mistral 7B Model](https://huggingface.co/second-state/Mistral-7B-Instruct-v0.3-GGUF) to support the query generation and summarization capabilities of the LlamaEdge Query Server. Note that it's possible to use other tool supported models, as long as they're in the GGUF format.
Store it in the git folder.

#### Execute it

This will expose the LlamaEdge Query Server on the port 8081 on your local machine by default.

```
$ wasmedge --dir .:.  --env LLAMA_LOG="info" \
    --nn-preload default:GGML:AUTO:Mistral-7B-Instruct-v0.3-Q5_K_M.gguf \
    llamaedge-query-server.wasm \
    --ctx-size 4096 \
    --prompt-template mistral-tool \
    --model-name Mistral-7B-Instruct-v0 \
    --temp 1.0 \
    --log-all
```

Please note that you can also choose to use an external LlamaEdge Query Server instance.

### Start the API server

Let's start the Search Enabled LlamaEdge RAG API server on port 8080. By default, it connects to the local Qdrant server.

```
$ wasmedge --dir .:. --nn-preload default:GGML:AUTO:Llama-2-7b-chat-hf-Q5_K_M.gguf \
    --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
    rag-api-server.wasm \
    --model-name Llama-2-7b-chat-hf-Q5_K_M,all-MiniLM-L6-v2-ggml-model-f16 \
    --ctx-size 4096,384 \
    --prompt-template llama-2-chat,embedding \
    --rag-prompt "Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n" \
    --api-key "xxx" \                               # Use if your chosen LlamaEdge query server endpoint requires one.
    --query-server-url "http://0.0.0.0:8081/" \     # URL of the LlamaEdge query server of your choosing. This is the default local endpoint.
    --log-prompts \ 
    --log-stat
```

The CLI arguments are self-explanatory.
Notice that those arguments are different from the [llama-api-server.wasm](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server) app.

* The `--nn-proload` loads two models we just downloaded. The chat model is named `default` and the embedding model is named `embedding` .
* The `rag-api-server.wasm` is the API server app. It is written in Rust using LlamaEdge SDK, and is already compiled to cross-platform Wasm binary.
* The `--model-name` specifies the names of those two models so that API calls can be routed to specific models.
* The `--ctx-size` specifies the max input size for each of those two models listed in `--model-name`.
* The `--batch-size` specifies the batch processing size for each of those two models listed in `--model-name`. This parameter has a large impact on the RAM use of the API server.
* The `--rag-prompt` specifies the system prompt that introduces the context of the vector search and returns relevant context from qdrant.

There are a few optional `--qdrant-*` arguments you could use.

* The `--qdrant-url` is the API URL to the Qdrant server that contains the vector collection. It defaults to `http://localhost:6333`.
* The `--qdrant-collection-name` is the name of the vector collection that contains our knowledge base. It defaults to `default`.
* The `--qdrant-limit` is the maximum number of text chunks (search results) we could add to the prompt as the RAG context. It defaults to `3`.
* The `--qdrant-score-threshold` is the minimum score a search result must reach for its corresponding text chunk to be added to the RAG context. It defaults to `0.4`.

The following are search-specific arguments.

* The `--api-key` is the API key to be supplied to the endpoint, if the endpoint in use supports it.
* The `--query-server-url` is the URL for the LlamaEdge query server.
* The `--search-api-backend` is the search API backend on the previously specified LlamaEdge Query Server to use for internet search. We use "Tavily" by default.

## Chat with supplemental knowledge

Just go to `http://localhost:8080/` from your web browser, and you will see a chatbot UI web page. You can now
ask any question about Paris and it will answer based on the Paris guidebook in the Qdrant database!

> While the LLM runs locally, any queries you make to the chatbot will go to the currently selected search API endpoint in the supplied LlamaEdge Query Server. This endpoint is what performs the internet search. The default is Tavily, which is a proprietary search API. You will have to register for an API Key in order to make use of this API endpoint, and pass it with the `--api-key` as mentioned above.

Or, you can access it via the API. 

```
curl -X POST http://localhost:8080/v1/chat/completions \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Where is Paris?"}]}'

{
    "id":"18511d0f-b760-437f-a87f-8e95645822a0",
    "object":"chat.completion",
    "created":1711519741,
    "model":"Meta-Llama-3-8B-Instruct-Q5_K_M",
    "choices":[{"index":0,
      "message":{"role":"assistant","content":"Based on the provided context, Paris is located in the north-central part of France, situated along the Seine River. According to the text, people were living on the site of the present-day city by around 7600 BCE, and the modern city has spread from the island (the Île de la Cité) and far beyond both banks of the Seine."},
    "finish_reason":"stop"}],"usage":{"prompt_tokens":387,"completion_tokens":80,"total_tokens":467}
}
```

## Next steps

Now it is time to build your own LLM API server withboth short and long-term memory! You can start by using the same embedding model but with a different document. Remember to also supply it your API keys.

Good luck!
