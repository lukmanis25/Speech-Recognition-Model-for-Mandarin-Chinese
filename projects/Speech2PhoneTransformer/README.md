# Multi-Task Learning approach

## Papers with code

[ASR on AISHELL-1](https://paperswithcode.com/sota/speech-recognition-on-aishell-1)
->
[Multi-Task Encoder-Decoder](https://paperswithcode.com/paper/mmspeech-multi-modal-multi-task-encoder)

## Data

- nasze dane
- [AISHELL](https://www.openslr.org/33/)
    ```python
    ! huggingface-cli login
    from datasets import load_dataset
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("shenberg1/aishell3")
    ```
    ./AISHELL-3/train/train_content.txt:
    ```none
    SSB00050009.wav	北 bei3 郊 jiao1
    ```
    wav -> phonemes (chinese and pinyin with tones denoted by numbers)

## PLAN

## Encoder

- .ogg -> .wav -> MFCC  -> embedding
- words (pinyin)        -> embedding

words to phonems:
[english to pinyin](https://www.thepurelanguage.com/englishtranslationfree.aspx)

word | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
-|-|-|-|-|-|-|-|-|-|-
pinyin | yī | èr | sān | sì | wǔ | liù | qī | bā | jiǔ | shí

```python
tone = unicodedata.normalize('ì')[1]
tones = {1: ' ̄ ', 2: ' ́ ', 3: ' ̌ ', 4: ' ̀ ', 0: ''}
# fifth (zeroth) tone = normal (no diacritic marks)
```

### Embedding

pinyin phoneme embeddings, e.g. yī = yi1 -> vector

as for labels map phone (e.g. yī / yi1) - integer (e.g. 0)
<!-- has to include pronunciation and tone info

say the fixed width is 8 and the syllable is 'sān'

append the tone: 'sān ̄ ' and flex

let's go with [ s s ā ā n n ̄  ̄  ]

^ emphasis on vowels, tone (in form of diacritic character or a number) in the end -->

## Decoder

- <|START|> <|TASK|> [embd] <|END|>
- if <|TASK|> = <|PRON|> -> score(0-10)
- if <|TASK|> = <|TONE|> -> tone(-,1-4)

## Output

- score / tone

## BRUDNOPIS

- robic speech2pinyin -> redukcja do score, czy speech+pinyin2score ?
```none
Corpus.train = tensor([[L1T1, L2T1, ..], [L2TQ, L2T2, ..], ..])
file_wav = 'SSB06930002.wav'
```