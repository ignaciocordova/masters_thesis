# Useful references 

## General

- Artículo explicando Transformers: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Video (16min) explicando Transformers [https://www.youtube.com/watch?v=XSSTuhyAmnI&ab_channel=AriSeff](https://www.youtube.com/watch?v=XSSTuhyAmnI&ab_channel=AriSeff)
- Curso de Transformers: [https://www.youtube.com/watch?v=0SmNEp4zTpc&ab_channel=LennartSvensson](https://www.youtube.com/watch?v=0SmNEp4zTpc&ab_channel=LennartSvensson)
- Vision Transformers (1h 6min) [https://www.youtube.com/watch?v=J-utjBdLCTo&ab_channel=AICamp](https://www.youtube.com/watch?v=J-utjBdLCTo&ab_channel=AICamp)

## Papers 

### Video Vision Transformers 

- ViViT: A Video Vision Transformer [https://arxiv.org/pdf/2103.15691.pdf](https://arxiv.org/pdf/2103.15691.pdf)
- Is Space-Time Attention All You Need for Video Understanding? [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095)

### Transformers for images 

- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale https://arxiv.org/pdf/2010.11929.pdf

### Transformers 

- Attention is all you need https://arxiv.org/pdf/1706.03762.pdf

## Notebooks and code 
- Video Vision Transforming for classification of different organs https://keras.io/examples/vision/vivit/

- Video Vision Transformers for classification of videos https://keras.io/examples/vision/video_transformers/

----

## Key ideas 

### HISTORIA

2017 Transformers in NLP

2020 ViT

2021 T for VIDEO

### RNN

Already uses encoder-decoder architecture

Encoder produces a WHOLE SENTENCE EMBEDDING 

 The decoder has a hard job! 

Computationally, we need to wait for the previous result to compute the next one. 

Thus, paralellization is not possible. 

### Transformers

AT EACH DECODING STEP WE HAVE ACCES TO THE WHOLE MEMORY 

¿Can we parallelize? 

Solution: TO COMPLETLY BREAK THE RECURRENT CONNECTIONS! Attention is all you need (2017) 

Attention: inter-sequence dependencies

Self-attentio: intra-sequence dependecies 

Positional Encoding allows for attention to be permutationally invariant 

Probelms: attentio is EXPENSIVE to compute O(n^2)  each token dots product with each token! 

### IN VISION

CNNs analyze neighbourhoods of pixels while Transformers allow to find important relations between distant regions of an image. 

Tokens can be retrived from an Object detector 

IMAGE PATCHES: An image is worth 16x16 words 

When I have a very large data set I will use vision transformers. Otherwise CNNs are better. The two networks look at the image in a different way. Each one focuses on a specific part.