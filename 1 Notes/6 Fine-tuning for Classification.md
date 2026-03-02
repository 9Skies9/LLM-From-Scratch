Can you believe that we are at the point of fine tuning? Technically, this is a very annoying stage due to how all the different prompts of language being different lengths to one another, something that the GPU absolutely hates.

This chapter's fine tuning will simply be done towards classification, and the simplest type of classification of 'yes or no'. Next chapter's fine tuning will be directed towards being a useful personal assistant.

Basically... can you imagine in 2026 a person is still building a spam classification service with an LLM? That's what we are doing in this chapter...

![[110 Learning/112 CS/112.3 AI/112.3.2 NLP/3 LLM From Scratch/1 Notes/attachments/Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library) 25.jpg|500]]

[[Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library).pdf#page=193&rect=98,299,423,501|Build a Large Language Model (From Scratch) (Sebastian Raschka) (Z-Library), p.171]]

Big idea goes as such:
1. get dataset
2. process it, make data loaders
3. get a pre-trained model (that has done pre-training, at a much larger scale than the one we had done locally)
4. modify model for fine tuning (we need a classification head)
5. fine tune it using the dataset
6. evaluate it
7. see how it works on new data

Again, not much in the notes section, so much of this is just following along the code in the book.