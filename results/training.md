# Training details

## Dataset

Each model (so far) has been trained from a variety of sources, including:
 - [AstroBin](https://welcome.astrobin.com/)
 - [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html)
 - [r/Astrophotography](https://www.reddit.com/r/astrophotography/)
 - [r/Astronomy](https://www.reddit.com/r/astronomy/)

 The total size of the dataset is 15 thousand images, with an addition 5 thousand of non-deep-space targets.

## AstroSleuthV1
The first model that was created - It was intended to be a "dipping my toes in" kind of thing for upscaling in general, as there wasnt any good solutions before hand. The underlying architecture is is a 6-block RealESRGAN model, which was original used for anime. 

At the time I didn't realise it, but the training had actually failed as the discriminator got too good at identifing the generator so it "gave up" and stuck to minimising the error between the ground truth and predicted image.

## AstroSleuthV2 
Identifying that V1 could see some improvements on the discriminator front, I wrote my own pipeline and used a [projected discriminator](https://github.com/autonomousvision/projected-gan) as it seems to perform well in low-compute environments. I also trained a custom VGG model for the perceptual loss to better fit the dataset, and used clip as kind of a shot-in-the-dark. The biggest difficulty was balancing all these loss function's weights, as making one too large would result in the generator abusing it and throwing the rest off.

## AstroSleuthNEXT 

This model was founded on a completely different basis from the first two. Rather than trying to fight against 3 (or more) different loss functions, the discriminator was set with this responsibility. 

This does result in some undesired effects, notably some color or structural artifacts, which is why I also added conditioning to the model to try and mitigate this. 

An important note is that conditioning does not work with ncnn (the binary release), as layer normalization is not yet supported.  

## AstroSleuthFAST
A VGG based upscaler, which is borrows from the realesr repo. Despite being a quarter the size of V1, it performs similarly, but admittedly suffers from worse high-frequency detail and some star "shadows". 

## Future plans
I really hope to get FAST up to speed (in terms of quality) with V1, then pull experimental into master.

## Notes
Checked out [Deep Image Prior](https://arxiv.org/abs/1711.10925). Too slow and bad/very finicky results.