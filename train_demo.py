# train_demo.py
import math
import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm

from transformer import Transformer
from lr_warmup import WarmupLR

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
BOS, EOS, PAD, UNK = "[BOS]", "[EOS]", "[PAD]", "[UNK]"

def build_tokenizer(text_iter, vocab_size=8000):
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=[PAD, BOS, EOS, UNK])
    tok.train_from_iterator(text_iter, trainer=trainer)
    return tok

def encode_sentence(tok, s):
    ids = tok.encode(s).ids
    bos = tok.token_to_id(BOS); eos = tok.token_to_id(EOS)
    return [bos] + ids + [eos]

def collate(batch, tok, max_len=128):
    pad_id = tok.token_to_id(PAD)
    src_batch, tgt_in_batch, tgt_out_batch = [], [], []
    for en, fr in batch:
        src = encode_sentence(tok, en)[:max_len]
        tgt = encode_sentence(tok, fr)[:max_len]
        tgt_in  = tgt[:-1]
        tgt_out = tgt[1:]
        src_batch.append(src)
        tgt_in_batch.append(tgt_in)
        tgt_out_batch.append(tgt_out)

    def pad_to_max(lst):
        m = max(len(x) for x in lst)
        return [x + [pad_id]*(m - len(x)) for x in lst]

    src_t = torch.tensor(pad_to_max(src_batch), dtype=torch.long)
    tgt_in_t = torch.tensor(pad_to_max(tgt_in_batch), dtype=torch.long)
    tgt_out_t = torch.tensor(pad_to_max(tgt_out_batch), dtype=torch.long)
    return src_t.to(DEVICE), tgt_in_t.to(DEVICE), tgt_out_t.to(DEVICE), pad_id

def main():
    # 1) Load EN–FR
    ds = load_dataset("opus_books", "en-fr", split="train")  # small & demo-friendly
    pairs = [(ex["translation"]["en"], ex["translation"]["fr"]) for ex in ds]

    # 2) Train a tiny byte-level BPE tokenizer on both languages
    text_iter = (txt for pair in pairs for txt in pair)
    tok = build_tokenizer(text_iter, vocab_size=8000)
    vocab_size = tok.get_vocab_size()

    # 3) Dataloader
    random.shuffle(pairs)
    train_pairs = pairs
    loader = DataLoader(train_pairs, batch_size=32, shuffle=True,
                        collate_fn=lambda b: collate(b, tok))

    # 4) Model
    model = Transformer(
        d_model=512, num_heads=8, num_encoders=6, num_decoders=6,
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, max_len=16384
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = WarmupLR(opt, d_model=model.d_model, warmup_steps=1000)

    # 5) Train a couple of steps to demonstrate updates
    model.train()
    for step, (src, tgt_in, tgt_out, pad_id) in enumerate(tqdm(loader, total=2000)):
        if step == 2000: break  # short demo
        opt.zero_grad()
        loss = model.training_step(src, tgt_in, tgt_out, pad_id=pad_id)
        loss.backward()
        opt.step()
        sched.step()

        if step % 50 == 0:
            lr_now = sched.get_last_lr()[0]
            print(f"step {step} | loss {loss.item():.4f} | lr {lr_now:.6f}")

            # (Optional) Peek at attention of first encoder layer, head 0
            enc0 = model.encoder.encoding_layers[0].self_attention
            attn = enc0.last_attn_per_head[0] if enc0.last_attn_per_head else None
            if attn is not None:
                print("Encoder layer 0, head 0 attention map:", tuple(attn.shape))  # [B, S, S]

    # quick qualitative check
    model.eval()
    with torch.no_grad():
        src, tgt_in, tgt_out, pad_id = collate([train_pairs[0]], tok)
        logits = model(src, tgt_in)
        print("Logits shape:", tuple(logits.shape))  # [B, T, vocab]

    # 6) Save model + tokenizer
    save_path = "transformer_en_fr.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "tokenizer_json": tok.to_str(),  # serialize tokenizer
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
