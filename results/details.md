## Details

The provided sample images were selected from the daily top of r/astrophotography on 4/9/2023.
To minise cherry picking, my only criteria was that they must cover a broad range of deep space targets.
They have not been resized or had noise added to them, but were cropped to a 512x512 tile of my choosing based on what I deemed as the most important aspect. eg: lots of noise/stars, nebulosity, or artifacts.

## Tips, Tricks & Issues

After upscaling many images, in general it almost always helps to **reduce the image size** before feeding it into the model. It is slightly counterintuitive as one might think that larger is better, but the main failure point is the original size. I believe this is due to the fact that the model is trained to expect images that are 4x smaller than they should be because it's task is to reverse this process.

**After reducing image size such that the stars are at no more than 6pixels wide, performance can drastically improve.**

To get to the expected size, the upscaled "crushed" image can be upscaled once again.

### Workflow for difficult cases

1. Original -> +downsample 4x -> +noise -> +blur -> Crushed
2. Crushed -> +upscale -> +upscale -> Output

Please note that the tile size parameter for the model was reduced from 512x512 to 128x128 to fit the input image better.

### Summary

1. Reduce image size and upscale twice
2. AstroSleuthV1 generally works better for smoother results, V2 for anything else
3. Don't be afraid to add noise or blur the image to improve results
4. Diffraction spikes are almost always troublesome

### Workflow result on original/sample2.png



| Raw AstroSleuthV2  | Workflow with V2  |  Workflow with V1  |
| --- | --- | --- |
| <img src="resize_test/original_output.jpg"> | <img src="resize_test/crushed_output_v2.jpg"> | <img src="resize_test/crushed_output_v1.jpg"> |

As you can see, this workflow works much better with AstroSleuthV1 for this particular image. However I find results can drastically differ based on the amount of noise/blur applied to the crushed image, but were kept the same for the sake of comparison.


### Image credits / sources
https://www.reddit.com/r/astrophotography/comments/12fnhb3/m35_and_ngc_2158/
https://www.reddit.com/r/astrophotography/comments/12flzf1/messier_106_in_lrgb/
https://www.reddit.com/r/astrophotography/comments/12febjp/orion_nebula/
https://www.reddit.com/r/astrophotography/comments/12fewff/northwestern_cygnus_nebulae/
