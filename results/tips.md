### TLDR

1. Manually reduce the input image size and upscale twice to get better results
2. AstroSleuthV1 generally works better for smoother results, V2 for anything else
3. Don't be afraid to add noise or blur the input image to potentially improve results
4. Diffraction spikes are almost always troublesome

## Tips, Tricks & Issues

The model that tries to improve the image is trained on really noisy and blurry inputs. If the input image is too clean, you may get artifacts and sub-par performance. That is why I recommended making the image *worse* by manually adding blur/noise/downscaling with an image processing tool such as gimp or photoshop. 

The most significant factor that determines the quality of the output, is the input size. 
**After reducing image size such that the stars are at no more than 6pixels wide, performance can drastically improve.**

Another factor is the presence of diffraction spikes, as these are rarely present in the training data and as such, are more "troublesome" and can get a little wavy or split up into smaller stars. 
I unfortunately have to recommend *not* having diffraction spikes in the input image to begin with.

### Workflow for difficult cases

1. Original -> +downsample 4x -> +noise -> +blur -> "Crushed" image
2.  "Crushed" image -> +upscale -> +upscale -> Output


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
