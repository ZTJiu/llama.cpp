# 功能
使用 llamacpp 为一段文本生成高维词嵌入向量。

# 用法
```bash
./llama-embedding -m ./path/to/model --pooling mean --log-disable -p "Hello World!" 2>/dev/null
```

# 核心代码流程
## 第一步：初始化和加载模型：
```c++
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    const llama_vocab * vocab = llama_model_get_vocab(model);
```

## 第二步：prompt to token
```c++
    std::vector<std::vector<int32_t>> inputs;
    for (const auto & prompt : prompts) {
        auto inp = common_tokenize(ctx, prompt, true, true);
        inputs.push_back(inp);
    }
```

## 第三步：batch_decode
```c++
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_self_clear(ctx);

    // run model
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }
```
