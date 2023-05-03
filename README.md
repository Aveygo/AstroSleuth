---
title: AstroSleuth
emoji: 🌖
colorFrom: pink
colorTo: yellow
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
license: gpl-2.0
---

# AstroSleuth

<p align="center">
  <img src="https://media.githubusercontent.com/media/Aveygo/AstroSleuth/master/sample.png">
</p>

[![Aveygo - AstroSleuth](https://img.shields.io/static/v1?label=Aveygo&message=AstroSleuth&color=black&logo=github)](https://github.com/Aveygo/AstroSleuth "Go to GitHub repo")
[![stars - AstroSleuth](https://img.shields.io/github/stars/Aveygo/AstroSleuth?style=social)](https://github.com/Aveygo/AstroSleuth)[![Python 3.9.9](https://img.shields.io/badge/python-3.9.9-black.svg)](https://www.python.org/downloads/release/python-399/)

The (only?) free, zero bulls**t, 200 line, open source astrophotgraphy upscaler.

Sick of the commercialisation of deep space tools, I wanted a solution that can run on almost any hardware with epic results.

I started this project a regrettably long time ago. A lot has changed since then. I tried to share my work, got burned, removed it, perfected it, and fell into a well of "is it good enough".

I present my original model, a finetuned realesr-gan model trained on 15k images of astrophotography. It is behind my works on [reddit](https://www.reddit.com/user/CodingCoda), my [youtube](https://www.youtube.com/channel/UCHode4WV0hteze-ZDEG5atQ) attempt
and my [cloudy nights post](https://www.cloudynights.com/topic/816869-astrosleuth-image-denoiser-upscaler/), and I hope it will suit you well.

## Running

### Colab (recommended)
1. Visit [colab](https://colab.research.google.com/drive/1LxiNsnokF-6OmICSxWNvTeFEEZvRM2Lp?usp=sharing)
2. Enjoy!

### Locally (best, complicated)
1. Install [python](https://www.python.org/downloads/) (and [pip](https://phoenixnap.com/kb/install-pip-windows))
2. Download and unzip latest [release](https://github.com/Aveygo/AstroSleuth/archive/refs/heads/master.zip)
3. Open the terminal (right-click -> terminal) and run ```pip install -r requirements.txt```
4. Run the streamlit interface with ```streamlit run app.py```

### Hugging face (last resort, simple)
Go [here](https://huggingface.co/spaces/CodingCoda/AstroSleuth). Please note that hugging face servers use 2 core cpus, so large images may take a very long time, even timing out.

## Extra information

Please see [details](https://github.com/Aveygo/AstroSleuth/blob/master/results/details.md) for image samples and potential workflow improvements and [training](https://github.com/Aveygo/AstroSleuth/blob/master/training.md) for details on how the models are trained.

## Known issues

Results are now more comparable with BlurXterminator after training improvements (see [training](https://github.com/Aveygo/AstroSleuth/blob/master/training.md)). AstroSleuthV2 weights will be on the hugging face repo, but not automatically downloaded for the time being.

Currently investigating a "zero-knowledge" solution.

## Concerns and Personal Notes

Its not a understatement that this tool has changed my life. It was my first machine learning project. I even built full-stack applications searching for the perfect way to share my work.
I will continue to do so. Ask for any improvements and I will likely impliment them. I am begging for an excuss to work on it so any feedback is appreciated. I am interested in creating a Photoshop/Pixinsight plugin if thats what even a single person wants, just open a git issue [here](https://github.com/Aveygo/AstroSleuth/issues) and I'll see to it.

For the redditors, this tool is presented as is, free as long as it stays free, I cannot convey though words how much I dont care that its not "scientifically accurate".

<!---If it wasnt for https://www.rc-astro.com/ I wouldnt have built up the effort though spite to go though redeveloping this project. "Does BlurXTerminator fabricate detail? No" is full of s**t, when I got s**t for being honest and saying my model does-->
