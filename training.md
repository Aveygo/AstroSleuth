# Training details

Astrosleuth_v1 is a 6b RealESR-GAN model, trained on 15 thousand images of various deep space objects, taken from multiple sources including:
 - [AstroBin](https://welcome.astrobin.com/)
 - [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html)
 - [r/Astrophotography](https://www.reddit.com/r/astrophotography/)
 - [r/Astronomy](https://www.reddit.com/r/astronomy/)

Astrosleuth_v2 is a continuation of v1, where I add the following improvements:
 - [Projected discriminator](https://github.com/autonomousvision/projected-gan)
 - Greator emphasis on visual quality rather than accuracy
 - Custom VGG model for perceptual loss
 - Less JPG degredations and more motion / guassian blur
 - Trained with a much lower learning rate
 - Pruned the dataset with CLIP for more high quality ground truths

As of writing, v2 is not publicly available, but will be added to the hugging face repository in the very near future.

## Future plans

Astrosleuth_v3 will be v2 with more emphasis on accuracy (no discriminator) to complete with BlurXTerminator as I believe that the community will appreciate such changes.

I am also considering a true "zero-knowledge" model, as described by [Deep Image Prior](https://arxiv.org/abs/1711.10925), but will leave alone for now to focus on current work.