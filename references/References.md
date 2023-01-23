# Useful references 

## General

- (MUY BUENO,BÁSICO) Artículo explicando Transformers: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Video (16min) explicando Transformers [https://www.youtube.com/watch?v=XSSTuhyAmnI&ab_channel=AriSeff](https://www.youtube.com/watch?v=XSSTuhyAmnI&ab_channel=AriSeff)
- Video (15min) explicando Transformers (muy bueno) https://www.youtube.com/watch?v=4Bdc55j80l8&ab_channel=TheA.I.Hacker-MichaelPhi
- Curso de Transformers: [https://www.youtube.com/watch?v=0SmNEp4zTpc&ab_channel=LennartSvensson](https://www.youtube.com/watch?v=0SmNEp4zTpc&ab_channel=LennartSvensson)
- Vision Transformers (1h 6min) [https://www.youtube.com/watch?v=J-utjBdLCTo&ab_channel=AICamp](https://www.youtube.com/watch?v=J-utjBdLCTo&ab_channel=AICamp)

## Papers 

### Next frame prediction 

- Applying attention in next-frame and time series forecasting https://arxiv.org/abs/2108.08224


### Video Vision Transformers 

- ViViT: A Video Vision Transformer [https://arxiv.org/pdf/2103.15691.pdf](https://arxiv.org/pdf/2103.15691.pdf)
- Is Space-Time Attention All You Need for Video Understanding? [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095)

### Transformers for images 

- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale https://arxiv.org/pdf/2010.11929.pdf

### Transformers 

- Attention is all you need https://arxiv.org/pdf/1706.03762.pdf

### TFG María Barroso

- https://repositorio.uam.es/bitstream/handle/10486/698026/barroso_honrubia_maria_tfg.pdf?sequence=1&isAllowed=y

## Notebooks and code 

- Video predictions using Transformers https://github.com/iamrakesh28/Video-Prediction 

- Video Vision Transforming for classification of different organs https://keras.io/examples/vision/vivit/

- Video Vision Transformers for classification of videos https://keras.io/examples/vision/video_transformers/

- Image classification with ViT https://keras.io/examples/vision/image_classification_with_vision_transformer/
(explained step by steo in this video -> https://www.youtube.com/watch?v=i2_zJ0ANrw0&t=5s&ab_channel=ConnorShorten)


----
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
