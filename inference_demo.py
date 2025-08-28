import torch
from tokenizers import Tokenizer
from transformer import Transformer

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

BOS, EOS, PAD, UNK = "[BOS]", "[EOS]", "[PAD]", "[UNK]"

def greedy_decode(model, tok, src_sentence, max_len=50):
    model.eval()
    with torch.no_grad():
        src_ids = [tok.token_to_id(BOS)] + tok.encode(src_sentence).ids + [tok.token_to_id(EOS)]
        src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

        # Start target with BOS
        tgt_ids = [tok.token_to_id(BOS)]
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=DEVICE)
            logits = model(src, tgt)  # [1, T, vocab]
            next_id = logits[0, -1].argmax(-1).item()
            tgt_ids.append(next_id)
            if next_id == tok.token_to_id(EOS):
                break

        return tok.decode(tgt_ids[1:])  # skip BOS when decoding

def main():
    checkpoint = torch.load("transformer_en_fr.pt", map_location=DEVICE)
    vocab_size = checkpoint["vocab_size"]

    # rebuild tokenizer
    tok = Tokenizer.from_str(checkpoint["tokenizer_json"])

    # rebuild model (match training hyperparams!)
    model = Transformer(
        d_model=512, num_heads=8, num_encoders=6, num_decoders=6,
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, max_len=16384
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])

    # test inference
    sentence = "Analyse the gait of the horse"
    translation = greedy_decode(model, tok, sentence)
    print("EN:", sentence)
    print("FR:", translation)

if __name__ == "__main__":
    main()