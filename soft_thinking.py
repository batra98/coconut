import math
import torch
import torch.nn.functional as F


class SoftThinking:
    """Inference-time Soft Thinking wrapper.

    This implements probability-weighted "concept tokens" by computing the
    probability-weighted sum of the model's token embeddings and feeding that
    continuous vector back as the next step's input embedding. Works with
    base causal LM or the `Coconut` wrapper (detects presence of
    `base_causallm`).

    Usage:
        outputs = SoftThinking.generate(model, tokenizer, input_ids, attention_mask, ...)

    Returns a tensor of token ids shaped (1, seq_len) to be compatible with
    the rest of the codebase. Optionally returns the final embeddings.
    """

    @staticmethod
    def generate(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens=32,
        device=None,
        cold_stop_threshold=0.1,
        cold_stop_patience=2,
        temperature=1.0,
        output_embedding=False,
        synced_gpus=False,
        eos_token_id=None,
    ):
        """Generate tokens using soft concept embeddings.
        
        Args:
            cold_stop_threshold: Entropy threshold for Cold Stop (0.0 to 1.0, default 0.1).
                When normalized entropy falls below this, increment low-entropy counter.
            cold_stop_patience: Consecutive low-entropy steps before stopping (default 2).
                Stops generation if entropy is low for this many consecutive steps.
            temperature: Softmax temperature for probability computation (default 1.0).
                Lower values make the distribution sharper, higher make it softer.
        """
        # Determine base model and embedding
        if hasattr(model, "base_causallm") and hasattr(model, "embedding"):
            base_model = model.base_causallm
            embedding = model.embedding
        else:
            base_model = model
            try:
                embedding = base_model.get_input_embeddings()
            except Exception:
                # fallback: try attribute
                embedding = model.get_input_embeddings()

        vocab_size, embed_dim = embedding.weight.shape

        if device is None:
            device = input_ids.device

        # initial inputs_embeds
        inputs_embeds = embedding(input_ids.to(device))

        tokens = input_ids[0].detach().tolist()

        # keep track of normalized entropy for Cold Stop
        low_entropy_count = 0

        # Keep generating
        new_inputs_embeds = inputs_embeds

        for step in range(max_new_tokens):
            outputs = base_model(inputs_embeds=new_inputs_embeds)
            logits = outputs.logits[:, -1, :]

            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

            # entropy per batch (we assume batch size 1 in current codebase)
            logp = torch.clamp(torch.log(probs + 1e-12), min=-1e9)
            entropy = -torch.sum(probs * logp, dim=-1)  # (bs,)

            # normalized entropy in [0,1] dividing by log(V)
            norm_entropy = entropy / math.log(max(vocab_size, 2))

            # cold stop: if normalized entropy is low for consecutive steps
            if (norm_entropy < cold_stop_threshold).all():
                low_entropy_count += 1
            else:
                low_entropy_count = 0

            # compute probability-weighted embedding: batch x embed_dim
            # probs: (bs, vocab), embedding.weight: (vocab, embed_dim)
            concept_embed = probs @ embedding.weight.to(device)

            # append concept_embed as the next token embedding
            concept_embed = concept_embed.view(concept_embed.size(0), 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds.to(device), concept_embed), dim=1)

            # for readability / evaluation, also pick discrete token by argmax
            next_token = torch.argmax(logits, dim=-1).item()
            tokens.append(next_token)

            # stop on eos
            if eos_token_id is not None and next_token == eos_token_id:
                break

            if low_entropy_count >= cold_stop_patience:
                break

        result = torch.tensor(tokens, device=device).view(1, -1)

        if output_embedding:
            return result, new_inputs_embeds
        return result
