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
Quick start on [colab](https://colab.research.google.com/drive/1LxiNsnokF-6OmICSxWNvTeFEEZvRM2Lp?usp=sharing), but checkout [running](results/running.md) for more details.

## Extra information

 - [Tips](results/tips.md) - for image samples and potential workflow improvements 
 - [Training](results/training.md) - for more information on each model

## Recent changes

 - AstroNEXT and AstroFAST models to replace V2 and V1 respectively. Read [training](results/training.md) for more details.
 - Minor UI changes for reading model descriptions.
 - Scaled the preview image for faster loading.
 - Ability to 'coerce' NEXT into stars vs details. See [conditioning](results/conditioning.md). 

## Known issues

 - Most models convert large stars into 'star clusters'
 - Diffraction spikes may turn into diffraction stars
 - Fine nebula details may "smush" together (varies from image-to-image)

## Why

I started this project a regrettably long time ago. I tried different ways to share my work, got some hate, some love, and setted for what you see now.

I present an acculmination of multiple ideas, improvements, and lessons; trained on 15 thousand images of various astrophotography targets. 

It is behind my works on [reddit](https://www.reddit.com/user/CodingCoda), my [youtube](https://www.youtube.com/channel/UCHode4WV0hteze-ZDEG5atQ) attempt and my [cloudy nights post](https://www.cloudynights.com/topic/816869-astrosleuth-image-denoiser-upscaler/), and I hope it leads the way for any other future attempts; for anyone.

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
