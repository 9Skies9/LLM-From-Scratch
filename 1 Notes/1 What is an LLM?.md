Large language models, like ChatGPT, are fundamentally neural networks, and neural networks, fundamentally are high dimensional function approximator, approximating the function of 'language' (the communication of information).

The terminologies that revolve around 'AI' is very confusing, so I'm glad there was a diagram breaking down what's what.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=25&rect=62,330,483,486|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.3]]

We know LLMs can do many things today, writing code, summarizing emails, cheating on writing HW... the TLDR is that it saves time for people, and people are pushing for LLMs to do even more things today.

The goal of the book is to make an LLM (duh), and this is roughly the steps of creating an LLM:
- data gathering & filtering
- pre-training
- fine-tuning

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 1.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=28&rect=86,265,475,477|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.6]]

All modern LLMs rely on a single idea of the 'transformer' from a 2017 paper 'Attention is All You Need'.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 2.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=30&rect=50,304,481,618|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.8]]


This was divided down into BERT and GPT, the transformer's encoder and decoder respectively. BERT's original training goal was to 'fill in the missing words', while GPT's original training goal was to 'generate the next word 1 by 1'.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 4.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=35&rect=62,232,482,537|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.13]]


And in training these language models, large datasets were scraped from the internet, for example, this is the datasets that OpenAI report to have used for training GPT-3.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 3.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=33&rect=98,479,478,618|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.11]]


Training these models take a lot of money too! In rough estimates, it would've taken $4.6M in computing resources to train GPT-3.

After that, researchers in OpenAI followed up in creating InstructGPT, which followed up with creating a supervised-training/fine-tuning dataset for GPT-3, and reinforcement learning algorithms.

![[Pasted image 20260130234710.png|500]]

GPT-3 was created in 2020, and we know that today this LLM tree is blossoming in every direction, countless companies and individuals want a piece of this pie, and have developed countless papers and projects that sparked this AI era.

However, the underlying heart of LLMs remain the same (as I'm writing this, and I hope someone flips the table someday), and we will go through the framework of building an LLM from 0 up.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 5.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=36&rect=56,143,474,371|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.14]]