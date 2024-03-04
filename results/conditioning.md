### Conditioning result on cond/raeikja9d7mc1.jpeg

[Image source](https://www.reddit.com/r/astrophotography/comments/1b5vfie/reprocessed_crescent_nebula_ngc_6888/)

TLDR: Conditioning looks successfull!

It certainly failed to keep the same color, however I believe this is because the color information may be contained within the condition embedding, and because I used a generalised one for inference, the output is inaccurate.

Will be very interesting to see what other possible conditioning prompts could do? 

Ideas:
 - Saturation
 - Contrast
 - Nebula detail
 - Diffraction spikes

| Original  | AstroNEXT (no cond)  |
| --- | --- |
| <img src="cond/raeikja9d7mc1.jpeg"> | <img src="cond/raeikja9d7mc1_neutral.jpeg"> |

| Low detail, high stars  | Low detail, low stars  |
| --- | --- |
| <img src="cond/raeikja9d7mc1_down_up.jpeg"> | <img src="cond/raeikja9d7mc1_down_down.jpeg"> |

| high detail, high stars  | high detail, low stars  |
| --- | --- |
| <img src="cond/raeikja9d7mc1_up_up.jpeg"> | <img src="cond/raeikja9d7mc1_up_down.jpeg"> |

