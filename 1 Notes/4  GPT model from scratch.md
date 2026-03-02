The LLM architecture consists of several building blocks, and we've already coded out the heart of multi-head attention, so we'll be implementing the other parts of the GPT architecture here.

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 21.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=115&rect=53,425,483,620|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.93]]

This is an official diagram of the GPT architecture:

![[Pasted image 20260212225414.png|200]]

And there will be 7 things to implement in this chapter of the book:
1. GPT Structure
2. Layer Normalization
3. GELU
4. Feed Forward
5. Shortcut Connections
6. Transformer Block

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 22.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=118&rect=83,455,425,619|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.96]]

Then the rest of this chapter is mostly just in code, not much notes I can take here except a mental model for how data flows through the GPT model.

- start by tokenizing the text into IDs
- convert IDs into embeddings
- let the embeddings run through the GPT model
- The model should return an output similar to the size of the input
- Post-processing steps allows us to receive text outputs

![[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 23.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=120&rect=48,257,481,623|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.98]]























