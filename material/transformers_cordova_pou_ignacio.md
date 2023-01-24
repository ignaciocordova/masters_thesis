---
output:
  pdf_document: default
  html_document: default
---

# Transformers  

### by Ignacio Cordova Pou

--------------------------------------------------------

1. Introduction 
  
    1.1. Where does the hype come from?

    1.3. What problems does it solve?

2. Attention is all you need (2017)

    2.1. Encoder-Decoder architecture

    2.2. Attention mechanism

    2.3. The Transformer step by step

3. Transformer as a general architecture

\pagebreak

## 1. Introduction
### 1.1. Where does the hype come from?

The transformer architecture has revolutionized the world of Artificial Intelligence since it was first presented to the world in the paper "Attention is all you need" by Vaswani et al. in 2017. The main reason for this is that it has been able to achieve state of the art results in a wide variety of tasks, such as machine translation, speech recognition, image captioning, etc.  

During the last years, the Transformer architecture has been applied to a wide variety of tasks and projects. Some of them are:

- BERT, a language model that has been able to achieve state of the art results in a wide variety of NLP tasks. 
- GPT-3, a language model that has been able to generate text that is indistinguishable from human written text.
- Github Copilot, a language model that has been able to generate complex code forom simple prompts.
- Protein folding has been a problem that has been unsolved for decades. The Transformer architecture has been able to provide the best solution so far to this problem. 
- Tesla Autopilot uses a Transformer architecture to convert the raw data from the cameras into a representation that is easier to understand for the neural network. 

I hope this gives you an idea of the impact that the Transformer architecture has had in the world of Artificial Intelligence. In this article, I will try to explain the Transformer architecture in a simple way and I will try to explain the main concepts that are needed to understand it. 

### 1.2. What problems does it solve?

The Transformer architecture is a general architecture that can be applied to a wide variety of tasks. However, it is important to understand the problems that it solves in order to understand why it is so powerful. Knowing that it was born for Natural Language Processing (NLP) we can lay out the problems that it solves in this context. 

The first problem that it solves is the problem of long term dependencies. In NLP, we usually deal with sequences of words. For example, in the sentence "The dog stayed at home because it was tired" we have a sequence of words. The problem is that the meaning of a word is not only related to the word that is next to it. For example, the word "tired" is related to the word "dog" but it is not in the immediate context of the word. That is what a long term dependency is. It is hard for RNNs or even LSTMs to capture long term dependencies since they have no direct acces to the information that is far away in the sequence. In other words, the information of "dog" has to travel al long way through multiple connections until it is required. We will see that the Transformer architecture solves this problem by using attention and allowing the decoder to access the information of the encoder at any point in the sequence. 

The mechanism of attention also allows for parallel processing. In RNNs, the information has to travel through the network sequentially. In the Transformer architecture, the information can travel in parallel. This allows for faster processing particularly in the case of long sequences. Besides that, we all like parallelization, don't we?  

Interpretability is another curtial and sometimes overlooked problem. Hidden states of both RNNs and Transformers are hard to interpret. However, the Transformer architecture provides attention scores that can be used to understand which parts of the input are more important for the output. This can be understood as what is the architecture paying attention to when making predictions. 

\pagebreak
## 2. Attention is all you need

The paper "Attention is all you need" by Vaswani et al. is the paper that introduced the Transformer architecture to the world. In this section, I will try to explain the main concepts of the paper. 

First I will dive in the encoder-decoder architecture. Then I will explain the attention mechanism and how it is used in the model presented in the paper. On next sections, I will show how the architecture can be applied to other problemas like image captioning, image classification, video understanding, etc.

### 2.1. Encoder-Decoder architecture

The encoder-decoder architecture is a general architecture that can be applied to a wide variety of tasks. It is composed of two parts: an encoder and a decoder. The encoder is used to encode the input into a representation that is easier to understand for the decoder. The decoder is used to decode the representation into the output. For example, in the case of machine translation, the encoder would encode the input sentence into a representation that is easier to understand for the decoder. The decoder would then decode the representation into the translated sentence. 

![Transformer architecture presented at "Attention is all you need" (2017)](./figures/transformer.png){height="300" width="200"}

Transformers are a special case of encoder-decoder architectures. Imagine we had an encoder that could capture information about the context of the input. Not only local context (like CNNs do with images) but also global context. Imagine that we could use this information to decode the input into the output taking advantage of the relationships between different elements of both the input and the output. The attention mechanism used by the Transformer architecture allows us to do exactly that. 

The problem we are trying to solve is to find an adequate representation of the input that contains information about the relationships between the different elements of the input. 

### 2.2. Attention mechanism

This is a key concept of the Transformer architecture. Please read this section carefully and make sure you understand it.

I stated that the problem we are trying to solve is to find an adequate representation of the input that contains information about the context. We can think of the context as the relationships between the different elements of the input. For example, in the sentence "The dog stayed at home because it was tired", when encoding the word "tired", we want to take into account the pressence of the word "dog" in the sentence. In other words, we want to pay attention to the word "dog" when encoding the word "tired". How is this achieved? From now one we must regard the words as embedded into a vector space so that we have a vector representation of each word. In "Transformer terminology" we refer to these vectors as embeddings and in this particular problem, the words would be our tokens. In the paper, one can read the following: "... we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model} = 512$". It is common for the embeddings to be learnable and to change during training as the model learns. It is important to note that this embeddings are context-independent, meaning that the word "dog" will always be represented by the same vector. We will see later that information about the position of the word in the sentence is captured by the positional encoding and that information about the context is captured by the attention mechanism. 






