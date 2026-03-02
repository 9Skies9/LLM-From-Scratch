Attention, is the core heart of what makes an LLM, and in this chapter, we will implement the 4 variations of attention, building up to the model that exists inside an LLM, the Multi-head attention.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 15.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=73&rect=89,128,479,310|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.51]]

We know prior to attention, there has always been a bottle neck problem between language translation, we can't just simply translate word to word due to grammatical structure of language.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 16.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=74&rect=84,193,475,479|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.52]]

We've known the history of trying to solve this 'how to get the entire context over' problem in sequence to sequence translation, from recurrent neural networks to long short term memory neural networks... this 'hidden state' idea really didn't go very far.
- the encoder processes the entire input text into a hidden state
- the decode takes in the hidden state to produce the output

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 17.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=75&rect=91,265,480,465|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.53]]

> [!PDF|yellow] [[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=75&selection=21,0,25,30&color=yellow|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.53]]
> > The big limitation of encoder–decoder RNNs is that the RNN can’t directly access earlier hidden states from the encoder during the decoding phase. Consequently, it relies solely on the current hidden state, which encapsulates all relevant information. This can lead to a loss of context, especially in complex sentences where dependencies might span long distances

And then comes the idea of attention, which allows the text generating decoder part of the network access all input tokens selectively, and so the decoder can 'choose' which parts of the inputs context is important to the output.
> It was a really old paper back in 2014!

It used:
- hidden states for encoder states to encoder states
- attention for encoder states to decoder states
- hidden states for decoder states to decoder states

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 18.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=76&rect=87,179,468,398&color=yellow|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.54]]

But then the paper 'Attention is all you need' just decided that there's no point to use any mechanism outside of attention for this entire encoding decoding process.

It used:
- self attention for encoder states to encoder states
- attention for encoder states to decoder states
- self-masked attention for decoder states to decoder states

And as we remember, the decoder section of the transformer makes the GPT architecture, and we will be making the attention mechanism in this note, and the GPT decoder architecture in the next note.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 20.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=77&rect=90,249,393,492|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.55]]

---
## Writing Out Attention

In this chapter, they've written out attention in 4 ways:
- simple self-attention (no trainable weights)
- self-attention
- casual attention (masked attention)
- multi-head attention

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 19.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=86&rect=54,109,474,361&color=yellow|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.64]]

It's mostly code, so I'm not going to write about it in the notes.