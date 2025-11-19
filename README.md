# Model Blueprint

## Architecture Selection
- **Model size**: ~700M-parameter decoder-only transformer with 24 layers, 3072 hidden dimension, 16 attention heads, and an 8k context window. The configuration is sized so a single A100 80GB (or comparable) can host the model with headroom for activation-checkpointing, enabling rapid iteration without distributed inference. Depth/width are tuned to maintain code-aware reasoning (24 layers) while keeping total parameters below the point where pretraining cost would exceed the available budget.
- **Tokenizer**: SentencePiece unigram with a 65k subword vocabulary. The unigram algorithm handles identifier-heavy source files and multilingual documentation by jointly optimizing for frequent programming tokens (e.g., `::`, `camelCase`, `snake_case`) and natural-language words. A 65k vocab keeps embedding matrices manageable while avoiding the fragmentation seen with 32k vocabularies on CJK text and shell commands.
- **Training objectives**: Baseline objective is causal language modeling over contiguous 8k windows. We also reserve capacity for an auxiliary contrastive document-grounding head that scores retrieved passages against generated responses; enabling it during supervised/RAG fine-tuning keeps the base checkpoint unchanged while providing a drop-in module for enterprise deployments that demand grounded citations.

## Pretraining Data Blend
- **70% coding corpora**: Open-source code datasets such as The Stack and StarCoderData (via Hugging Face). We ingest repo metadata to drop GPL/AGPL/LGPL projects, honor removal requests, and exclude binaries/tests that leak secrets. The coding-heavy mix is critical because downstream workflows expect synthesis and refactoring of enterprise SDKs.
- **20% general text**: Broad-coverage corpora including The Pile and SlimPajama ensure the model keeps world knowledge and conversational fluency. These sources are filtered for hate speech and duplicated fragments before mixing to prevent overwhelming the code signal.
- **10% synthetic instruction traces**: Self-instruct and templated dialogues teach the base model multi-turn assistant behavior even before supervised alignment. Synthetic traces are generated with strict prompt hygiene (no customer names, no PHI) to stay within privacy guidelines.
- **Filtering pipeline**: License-aware metadata parsing, automatic GPL/AGPL exclusion, near-duplicate removal (MinHash + SimHash), profanity/PII scrubbing, and language-based bucketing. We store provenance hashes so compliance can reproduce the exact blend or remove offending shards if takedown requests arrive.

## Fine-Tuning Phases
1. **Public alignment SFT**: Fine-tune on instruction datasets such as databricks-dolly-15k, OpenAssistant, and CodeAlpaca. Training emphasizes refusal style, tool-use templates, and step-by-step reasoning so the checkpoint is safe for early internal pilots. Mix in contrastive grounding batches to warm up the auxiliary head without proprietary data.
2. **Private RAG-grounded tuning**: After proprietary documentation, API references, and embedding indices are available, run retrieval-augmented fine-tuning (RA-FT). Prompts include citations and chunk IDs so the model learns to quote sources, disfavor hallucinations, and leverage the optional contrastive head for reranking. Access controls and audit logging are mandatory for this phase.

## Evaluation Targets
- **Code generation**: HumanEval and MBPP pass@1 >= baseline open 7B models. Stretch goal is +5 absolute points over StarCoderBase-1B after pretraining, with further gains from RA-FT.
- **General reasoning**: MMLU subsets covering business, computer security, and software engineering; target ≥60% on relevant subsets to ensure sufficient knowledge transfer from the general-text mix.
- **Chatbot success criteria**: Internal red-teaming should report ≥85% helpful/harmless ratings, <5% unsupported claims in scripted support scenarios, and ≥90% citation coverage when RAG context is present. Customer-ready gating requires a manual evaluation that the contrastive grounding head downranks hallucinated passages for at least 80% of sampled questions.
