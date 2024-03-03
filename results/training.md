# Training details

## AstroSleuthV1
Is a 6b RealESR-GAN model, trained on 15 thousand images of various deep space objects, taken from multiple sources including:
 - [AstroBin](https://welcome.astrobin.com/)
 - [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html)
 - [r/Astrophotography](https://www.reddit.com/r/astrophotography/)
 - [r/Astronomy](https://www.reddit.com/r/astronomy/)

## AstroSleuthV2 
This is strictly a continuation of v1, where I add the following improvements:
 - [Projected discriminator](https://github.com/autonomousvision/projected-gan)
 - Greator emphasis on visual quality rather than accuracy
 - Custom VGG model for perceptual loss
 - Less JPG degredations and more motion / guassian blur
 - Trained with a much lower learning rate
 - Pruned the dataset with CLIP for more high quality ground truths

## AstroSleuthNEXT 
This model was founded on a completely different basis from the first two. Rather than trying to fight against 3 (or more) different loss functions, the discriminator was set with this responsibility. 

This does result in some undesired effects, notably some color or structural artifacts. This is why I also added conditioning to the models to try and mitigate this, however as of 2024/3/3, this functionality is commented-away until the UI is up to scratch.

## AstroSleuthFAST
A VGG based upscaler, a complete copy from the realesr repo. Despite being a quarter the size of V1, it performs similarly, but suffers from worse high-frequency detail and some star "shadows". 

## Future plans
I really hope to get FAST up to speed with V1, then work on the ability to condition NEXT with at least some presets.

## Notes
Checked out [Deep Image Prior](https://arxiv.org/abs/1711.10925). Too slow and bad/very finicky results.