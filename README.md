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
  <img src="results/gui.png">
</p>

[![Aveygo - AstroSleuth](https://img.shields.io/static/v1?label=Aveygo&message=AstroSleuth&color=black&logo=github)](https://github.com/Aveygo/AstroSleuth "Go to GitHub repo")
[![stars - AstroSleuth](https://img.shields.io/github/stars/Aveygo/AstroSleuth?style=social)](https://github.com/Aveygo/AstroSleuth)[![Python 3.9.9](https://img.shields.io/badge/python-3.9.9-black.svg)](https://www.python.org/downloads/release/python-399/)

The (only?) free, no-strings-attached, open source astrophotgraphy upscaler toolset. I wanted a solution that can run on almost any hardware with epic results, and I hope this repo serves you well.

If you ever want to support the project, please give the repo a star so that it's easier for others to discover it.

## Running AstroSleuth

### Colab - Best method if you don't have a GPU 
1. Visit [colab](https://colab.research.google.com/drive/1LxiNsnokF-6OmICSxWNvTeFEEZvRM2Lp?usp=sharing)
2. Enjoy!

### Locally (Docker) - Somewhat recommended, not very "efficent"
1. Download [docker](https://www.docker.com/products/docker-desktop/)
2. In a terminal, run ```docker run -p 7860:7860 --gpus all --rm -ti --ipc=host aveygo/astrosleuth:experimental```
4. Go to [127.0.0.1:7860](http://127.0.0.1:7860)

### Locally (Python) - Highly recommended for experienced users
1. Install [python](https://www.python.org/downloads/) (and [pip](https://phoenixnap.com/kb/install-pip-windows))
2. Follow the instructions on the [pytorch](https://pytorch.org/get-started/locally/) website to install pytorch.
3. Download and unzip the latest [release](https://github.com/Aveygo/AstroSleuth/archive/refs/heads/master.zip) of AstroSleuth
4. Open the terminal (right-click -> terminal) and run ```pip install -r requirements.txt```
5. Run the streamlit interface with ```streamlit run app.py```

### Hugging face - Good for testing/playing around
1. Go [here](https://huggingface.co/spaces/Aveygo/AstroSleuth). Please note that hugging face servers use 2 core cpus and that you'll likely be sharing so large images may take a while.

### Locally ( Binaries ) - Not recommended, but good for fast inference and 'bare minimum' needs
1. Go to the [binary releases](https://github.com/Aveygo/AstroSleuth/releases/tag/v0.1.0) 
2. Download the latest zip for your platform, eg: astrosleuth-v0.1.0-windows.zip
3. Unzip and enter the folder
4. To add additional models, download the ".bin" and ".param" file from a model release, eg [FAST](https://github.com/Aveygo/AstroSleuth/releases/tag/FAST)  and place them in the "models/" directory. Use them with the "-n" tag, eg ```-n AstroSleuthFAST``` for step 6.

### Locally ( Binaries, continued ) - Windows
5. To run, double click ```run.bat``` and follow from there.
6. If you want to change the model, edit run.bat, line 50, to match step 4.

### Locally ( Binaries, continued ) - Linux/Macos
5. Set executable permissions with ```sudo chmod +x astrosleuth```
6. Run with ```./astrosleuth -n astrosleuth -i [input image path] -o [output path]```

## Extra information

Please see [details](results/details.md) for image samples and potential workflow improvements, as well as [training](results/training.md) for details on how the models are trained.

## Recent changes

 - AstroNEXT and AstroFAST models to replace V2 and V1 respectively. Read model.json for more details.
 - Minor UI changes for reading model descriptions.
 - Scaled the preview image for faster loading.

## Why

I started this project a regrettably long time ago. I tried different ways to share my work, got some hate, some love, and setted for what you see now.

I present an acculmination of multiple ideas, improvements, and lessons; trained on 15 thousand images of various astrophotography targets. 

It is behind my works on [reddit](https://www.reddit.com/user/CodingCoda), my [youtube](https://www.youtube.com/channel/UCHode4WV0hteze-ZDEG5atQ) attempt and my [cloudy nights post](https://www.cloudynights.com/topic/816869-astrosleuth-image-denoiser-upscaler/), and I hope it leads the way for any other future attempts; for anyone.

## Known issues


AstroSleuthNEXT likes to really invent lots of small stars in the background and struggles with large stars by turning them into star clusters as well as turning diffraction spikes into waves or smaller stars. Like with most ml problems, this looks like a garbage in - garbage out kind of situation. 

No known solutions other than to increase pixel loss weight during training. Currently waiting for any better ideas to pop up. Best way to get around this is by using AstroSleuthFAST.

## Concerns and Personal Notes

A lot has happend throughout this project. I guess cause it was my first machine learning application and it's been a nice distraction every now and again.

Ask for any improvements and I will likely implement them. Any feedback is appreciated, such as creating a Photoshop/Pixinsight plugin? Just open a git issue [here](https://github.com/Aveygo/AstroSleuth/issues) and I'll see to it.

This tool is presented as is, free as long as it stays free. "Scientific accuracy" was never the goal of this project - it was made to help others, and to make astronomy a little easier. 

<!--
Hi there, if you are reading this, then maybe you are interested in helping out. Make your changes, create a pull request, and I'll see to it.
-->

<!--Analytics-->
![Alt](https://repobeats.axiom.co/api/embed/dbcc73880aef11e4e7f6f0ae8c8f22557ea67e21.svg "Repobeats analytics image")

<!--git push hf HEAD:main-->