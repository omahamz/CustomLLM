# Security Practices for Private RAG Deployments

This document captures the minimum controls necessary for safeguarding proprietary
repositories while performing retrieval augmented generation.

## Air-gapped Training

* Keep the fine-tuning and embedding training loops inside an air-gapped network
  that mirrors production dependencies.
* Use one-way data diodes when exporting model checkpoints so private weights
  never leave the enclave without approval.

## Audit Logging

* Every ingestion event from `src/serving/api.py` must emit structured logs that
  capture operator identity, commit hash, and document fingerprint.
* Store the logs in an immutable bucket with a 13-month retention policy to
  satisfy enterprise compliance audits.

## Encryption at Rest

* Encrypt FAISS/LanceDB indices and any serialized metadata with AES-256 keys
  stored in a hardware security module.
* Rotate keys quarterly and re-encrypt indices immediately when a credential is
  revoked.
