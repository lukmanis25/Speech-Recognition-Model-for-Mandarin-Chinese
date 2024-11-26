# embeddings

(tutorial)[https://www.youtube.com/watch?v=5MaWmXwxFNQ]

*word embeddings are mathematical representations of words*

*każda kolumna wychodząca z pierwszej Conv to już jest embedding*

# Speech-Transformer

Speech-Transformer implementation:

(Speech-Transformer)[https://github.com/foamliu/Speech-Transformer/blob/master/transformer/encoder.py]

^ (ref)[https://sci-hub.se/10.1109/ICASSP.2018.8462506]

*L. Dong, S. Xu and B. Xu, "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018*

```python
n_tgt_vocab = N_TOKENS = 3393 + 3
d_word_vec = 512
self.emb = nn.Embedding(n_tgt_vocab, d_word_vec) # N_TOKENS x emb_vector_length
```

## Positional Encoding

przekształcenia trygonometryczne zwracające unikatowy wektor iksów dla każdego y w 2D

## Recognize

Bierze się encoder_output
jego długość sekwencji to maxlen
dec_output = [<SOS>]
teraz for i in maxlen
    dec_output.append = decoder(dec_output)
    topk = torch.topk(beam)

```python
maxlen = encoder_output.size(0)
ys = torch.ones(1,1).fill_(SOS_TOKEN) # SOS
yseq = ys
for i in range maxlen
    ys = yseq # 1*i
    dec_output = decoder(ys)
    # take top K hypothesis with torch.topk() or just use the best one each time
    topk = 1
    ys = cat(ys, torch.topk(dec_output))
    if ys[-1] == "<EOS>"
        break
```


