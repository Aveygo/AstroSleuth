## Running Astrosleuth

### Colab - Best method if you don't have a GPU 
1. Visit [colab](https://colab.research.google.com/drive/1LxiNsnokF-6OmICSxWNvTeFEEZvRM2Lp?usp=sharing)
2. Enjoy!

### Locally (Docker) - Somewhat recommended, may not have most recent features
1. Download [docker](https://www.docker.com/products/docker-desktop/)
2. In a terminal, run ```docker run -it -p 8501:8501 --gpus all --ipc=host registry.hf.space/aveygo-astrosleuth:latest streamlit run app.py -- --ignore_hf```
4. Go to [127.0.0.1:8501](http://127.0.0.1:8501)

### Locally (chaiNNer) - Highly recommended for new users
<!--Onnx not recommended as it doesnt play nice with anything other than a CPU or Nvidia GPU-->
1. Download [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) and run it.
2. On the left, scroll down to "dependency manager", and install the NCNN package
3. Download [NCNN.chn](https://raw.githubusercontent.com/Aveygo/AstroSleuth/master/chainner/NCNN.chn) (right click, save as) and open it in chaiNNer (file -> open)
4. Download your desired model in the ```.bin``` **AND** ```.param``` format from [releases](https://github.com/Aveygo/AstroSleuth/releases). Make sure the model files are placed next to each other. 
5. In chaiNNer, select your image, the model ```.param``` file, and output folder/filename.
6. Hit the green play button at the top.

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
