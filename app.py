import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.web.server.websocket_headers import _get_websocket_headers
from PIL import Image
import time, threading, io, os, sys

from file_queue import FileQueue
from main import AstroSleuth

IS_HF = os.getenv("HF_HOME") is not None
WARNING_SIZE = 1024 if IS_HF else 4096 
MAX_SIZE = 2048 if IS_HF else None
USE_DETECTOR = True if IS_HF else False

if USE_DETECTOR:
    print("WARNING: Space detector is being used! It's possible for a space image to be upscaled with the incorrect model if it gets misclassified!")

import argparse

parser = argparse.ArgumentParser(description='AstroSleuth')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
parser.add_argument('--torch', action='store_true', help='Use Torch')

args = parser.parse_args()
USE_GPU = args.gpu
USE_TORCH = args.torch

class App:
    def __init__(self):
        self.upscaling = False
        self.queue = None
        self.running = True

    def upscale(self, image):
        self.upscaling = True
        bar = st.progress(0)
        
        model = AstroSleuth(use_detector=USE_DETECTOR, use_onnxruntime=not USE_TORCH, device="cuda" if USE_GPU else "cpu")

        result = None
        for i in model.enhance_with_progress(image):
            if type(i) == float:
                bar.progress(i)
            else:
                result = i
                break
            
            if not self.running:
                break

        bar.empty()

        self.upscaling = False
        return result

    def heart(self):
        while self.running and self.queue.should_run():
            if _get_websocket_headers() is None:
                self.close()
                return

            self.queue.heartbeat()
            time.sleep(1)
    
    def render(self):
        st.title('AstroSleuth')
        st.subheader("Upscale deep space targets with AI")

        with st.form("my-form", clear_on_submit=True):
            file = st.file_uploader("FILE UPLOADER", type=["png", "jpg", "jpeg"])
            submitted = st.form_submit_button("Upscale!")
        
        if submitted and file is not None:
            image = Image.open(file)
            
            if MAX_SIZE is not None and (image.width > MAX_SIZE or image.height > MAX_SIZE):
                st.warning("Your image was resized to save on resources! To avoid this, run AstroSleuth with colab or locally: https://github.com/Aveygo/AstroSleuth#running", icon="⚠️")
                if image.width > image.height:
                    image = image.resize((MAX_SIZE, MAX_SIZE * image.height // image.width))
                else:
                    image = image.resize((MAX_SIZE * image.width // image.height, MAX_SIZE))

            elif image.width > WARNING_SIZE or image.height > WARNING_SIZE:
                st.info("Woah, that image is quite large! You may have to wait a while and/or get unexpected errors!", icon="🕒")

            self.queue = FileQueue()
            
            queue_box = None
            while not self.queue.should_run():    
                if queue_box is None:
                    queue_box = st.warning("Experincing high demand, you have been placed in a queue! Please wait...", icon ="🚦") 
                time.sleep(1)
                self.queue.heartbeat()
            
            t = threading.Thread(target=self.heart)
            add_script_run_ctx(t)
            t.start()

            if queue_box is not None:
                queue_box.empty()

            info = st.info("Upscaling image...", icon="🔥")

            image = self.upscale(image)
            if image is None:
                st.error("Internal error: Upscaling failed, please try again later?", icon="❌")
                self.close()
                return     
            
            if queue_box is not None:
                queue_box.empty()
            info.empty()

            st.success('Done! Loading result... (Please use download button to save result for the highest resolution)', icon="🎉")
            
            b = io.BytesIO()
            file_type = file.name.split(".")[-1].upper()
            file_type = "JPEG" if not file_type in ["JPEG", "PNG"] else file_type
            image.save(b, format=file_type)
            st.download_button("Download", b.getvalue(), file.name, "image/" + file_type)

            st.image(image, caption='Upscaled preview', use_column_width=True)
            self.close()
        
    def close(self):
        self.running = False
        if self.queue is not None:
            self.queue.quit()
            self.queue = None

app = App()
app.render()