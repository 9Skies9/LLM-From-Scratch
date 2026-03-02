The simple problem that machines can't deal with numbers, solved by giving text 'embeddings', a vector representation of certain chunks of text.

The idea of embeddings also exists for other high dimensional data types that cannot be directly turned into numbers, like videos or audio, at the end of the day, it's a mapping tool from these mediums into numbers (maybe meaningful numbers), a format which neural networks can process.

A way people like to think of embeddings is that similar ones stay together, different ones stay apart, and there might be relationships between them alike words.

![[Pasted image 20260131003138.png|300]]

- The typical king, man, royal idea never gets old, it's like vector addition with words

There is a long history of making, and improving these word embeddings, from *Word2Vec*, *Glove*, *FastText*, *Elmo*... to eventually making these embeddings part of the training of an LLM directly.

Still, breaking the text down into individual units and assigning numbers to them are necessary before the conversion into their respective embeddings, this process is called tokenization.

![[110 Learning/112 CS/112.3 AI/112.3.2 NLP/2 LLM From Scratch/attachments/Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 6.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=43&rect=89,211,455,548|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.21]]

This chapter aims to:
- import a piece of text
- make a simple tokenizer and run it on the text
- import an existing tokenizer and run it on the text
- create a dataset from the text
- create embeddings and positional encodings for the tokens

The rest is all done in code rather than notes.

---
### Converting Tokens into IDs

Assuming you have tokens from the regular expressions steps he talked about, the next part is assigning unique tokens in the dataset a respective ID.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 7.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=47&rect=97,344,466,619|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.25]]

With a unique 'vocabulary' built mapping tokens to IDs, you can now write out a simple tokenizer process that takes in text, breaks it down, and maps tokens to ID based on the existing vocabulary.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 8.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=48&rect=86,185,463,468|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.26]]

And inversely, turn tokens back into words.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 9.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=50&rect=85,381,467,615|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.28]]

---
# Special Tokens

If you've ever dived a little behind the conversation of a human and an LLM, you'll find special little pieces of language that doesn't look quite like english.

![[Pasted image 20260131111311.png|500]]

`<|im_start|>`, `<|im_sep|>`, `<|im_end|>`... these are special 'words' added to the vocabulary to help an LLM (and, probably the fancy interfaces) to determine which is by the user, the LLM, and the system (a reminder for the LLM).

Right now we aren't there yet, but special tokens like `<|unk|>` and `<|end_of_text|>` could be useful.

- `<|unk|>` represents unknown, it defaults to this character when it encounters an unknown piece of text not in the vocab
- `<|end_of_text|>` represents... end of a piece of text, helps the LLM in pre-training to realize certain text sources are unrelated.

Other tokens will also eventually come into play during training, like `[BOS]`, [EOS], [PAD]... these are all arbitrary names used to help divide the text for LLMs

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 10.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=52&rect=86,392,444,614|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.30]]

---
## BPE Algorithm

As we said, tokenization through regular expressions is limited, very limited considered the vast amount of tokens possible, maybe it's an emoji, maybe it's a language outside of english.

So BPE (Bye Pair Encoding), originally a text compression algorithm, was used for constructing the vocabulary for GPT-2, GPT-3, ChatGPT.

You can go read about how PBE works here, we aren't concerned about the details of it's implementation, and will use existing implementations.

https://www.youtube.com/watch?v=HEikzVL-lZU

---
## Text in Pre-training

For the pre-training of the LLM, text inputs and outputs should work like this.

Essentially, given some input of text, predict the next word, we will repeat this for a certain *context size*.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 11.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=57&rect=95,334,408,493|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.35]]

For a large corpus of text, this obviously won't be implemented in basic python, and will be implemented in Pytorch instead.

The details of how Pytorch handles data... like iterating, batching, is for appendix A.

---
## Token Embeddings

As we said, token embeddings are usually trained as a part of the LLM now, the book has decided to put these alongside the tokens.

But the idea is simple, text map to tokens, tokens map to embeddings.

There are 2 parts to embeddings:
- word embeddings (tokens to embeddings)
- positional embeddings (position to embeddings)

Word embeddings is essentially a look up operation in the embedding matrix.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 12.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=66&rect=83,425,396,621|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.44]]

Idea for positional embeddings is simple, it makes order of words matter:
- I eat fish, fish eat I, not the same idea.

And so this is achieved by adding token embeddings to positional embeddings, to reach the final input embeddings.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 13.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=67&rect=96,323,441,449|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.45]]

And this creates the input embedding pipeline.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 14.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=70&rect=86,278,393,616|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.48]]