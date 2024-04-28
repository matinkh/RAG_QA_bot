# My Notes (Draft)
See todo.txt for the actual plan.

---
---

##

* Difficulty in visualizing and inspecting the execution flow of LLMs
* Challenges in debugging LLM chains and prompts
* Inefficiency in evaluating the performance of LLM chains and prompts
* Lack of secure storage and management of prompts and LLM chain configurations

What **W&B** offers:

* See and track LLM inputs/outputs
* Track token usage
* Understand tool usage or more complex chains
* View chain/model configuration
* Store and version prompt templates
* ...

---

### 3. Notebook Notes

1. Typically, you want to use the **same LLM model** for your **embeddings and inference**.

2. The good thing about `Artifact` in WandB is that we can load the artifacts we stored previously and go to inference/prod. That's helpful if there's something wrong with the new data, training process, etc, where our new artifacts are corrupted.
3. Artifacts make everything reproducible.
4. We can log the artifact as a csv or as table.



For summarization

1. We don't hard-code the artifact. We pass it to `get_data`. So anyone who's pointing to WandB's project have access to artifacts and can choose one. That's good for reproducibility.
2. The whole summarization, is a multi-prompt application.
   1. Summarize each token. (`map_prompt_template`)
   2. Summarize the summarizations of the tokens into one nice summary. (`combine_prompt_template`)
3. **Summarization** takes a long time. But QA will be much faster.
