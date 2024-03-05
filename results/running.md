## Running Astrosleuth

### Colab - Best method if you don't have a GPU 
1. Visit [colab](https://colab.research.google.com/drive/1LxiNsnokF-6OmICSxWNvTeFEEZvRM2Lp?usp=sharing)
2. Enjoy!

### Locally (Docker) - Somewhat recommended, may not have most recent features
1. Download [docker](https://www.docker.com/products/docker-desktop/)
2. In a terminal, run ```sudo docker run -it -p 7860:7860 registry.hf.space/aveygo-astrosleuth:latest streamlit run app.py```
4. Go to [127.0.0.1:7860](http://127.0.0.1:7860)

### Locally (Python) - Highly recommended, especially for experienced users
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
