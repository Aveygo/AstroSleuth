# Comparing against BlurXTerminator

Images taken from https://www.rc-astro.com/


The results below are the outputs from the RC-Astro suite of BlurX and NoiseX, against AstroSluthNEXT.

## Debluring
I believe that the results of NEXT outperforms BlurXTerminator, as well as producing a more realistic result (albeit, perhaps a little too 'creative"). 

| Original  | RC-Astro  | AstroNEXT |
| --- | --- | --- |
| <img src="blurx/blur_input.jpg"> | <img src="blurx/blur_blurx.jpg"> | <img src="blurx/blur_next.jpg"> |


## Denoising
Denoising is a little more difficult to judge. This is because I had to turn off YR/CB upsampling for a decent result, displaying NEXT's biggest flaw; color matching. While I think the "per-pixel" quality is better, NoiseX is certainly more faithful to the original. Only AstroSleuthV1 was able to perform in a similar manner to NoiseX within my suite. 

One theortical solution is this is to use V1's YR/CB color channels and combine them into NEXT's luminance for the most optimal result, but again, this is a band aid fix on a fundamental problem of NEXT. 

| Original  | RC-Astro  | AstroNEXT |
| --- | --- | --- |
| <img src="blurx/noise_input.jpg"> | <img src="blurx/noise_blurx.jpg"> | <img src="blurx/noise_next.jpg"> |