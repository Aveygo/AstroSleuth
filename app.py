import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.web.server.websocket_headers import _get_websocket_headers

from PIL import Image
import time, threading, io, warnings, argparse, json
from os import listdir

from file_queue import FileQueue
from main import AstroSleuth

known_models = list(json.load(open("models.json"))["data"].keys())
known_models = ' '.join(known_models)

parser = argparse.ArgumentParser(description='AstroSleuth')

parser.add_argument('--cpu', action='store_true', help='Force CPU')
parser.add_argument('--ignore_hf', action='store_true', help='Ignore hugging face enviornment')
parser.add_argument('--modelname', default='astrosleuthv2', help=f'Select a model, available: {known_models}')

args = parser.parse_args()
FORCE_CPU = args.cpu
IGNORE_HF = args.ignore_hf
MODEL_NAME = args.modelname

# Check if we are running in huggingface environment
try: IS_HF = listdir('/home/')[0] == 'user'
except: IS_HF = False

# Set image warning and max sizes
IS_HF = IS_HF if not IGNORE_HF else False
WARNING_SIZE = 1024 if IS_HF else 4096 
MAX_SIZE = 2048 if IS_HF else None
Image.MAX_IMAGE_PIXELS = None if IS_HF else Image.MAX_IMAGE_PIXELS

if IS_HF: warnings.warn(f"Running in huggingface environment! Images will be resized to cap of {MAX_SIZE}x{MAX_SIZE}")

class App:
    def __init__(self):
        self.queue = None
        self.running = True
        self.model_name = MODEL_NAME
    
    def on_download(self, model_name):
        self.download_info = st.info(f"Downloading the model: {model_name} (this may take a minute...)", icon ="☁️")
    
    def off_download(self):
        self.download_info.empty()

    def upscale(self, image:Image, model_name:str)->Image.Image:
        # Convert to RGB if not already
        image_rgb = Image.new("RGB", image.size, (255, 255, 255))
        image_rgb.paste(image)
        del image

        # Start the model (downloading is done here)
        model = AstroSleuth(force_cpu=FORCE_CPU, model_name=model_name, on_download=self.on_download, off_download=self.off_download)

        # Show that upscale is starting
        self.info = st.info("Upscaling image...", icon="🔥")

        # Set the bar to 0
        bar = st.progress(0)

        # Run the model, yield progress
        result = None
        for i in model.enhance_with_progress(image_rgb):
            if type(i) == float:
                bar.progress(i)
            else:
                result = i
                break
            
            # Early exit if we are no longer running (user closed the page)
            if not self.running:
                break
        
        # Clear the bar
        bar.empty()
        return result

    def heart(self):
        # Beacause multiple users may be using the app at once, we need to check if
        # the websocket headers are still valid and to communicate with other threads
        # that we are still "in line"
        
        while self.running and self.queue.should_run():
            if _get_websocket_headers() is None:
                self.close()
                return

            self.queue.heartbeat()
            time.sleep(1)
    
    def render(self):
        st.title('AstroSleuth')
        st.subheader("Upscale deep space targets with AI")

        #st.text(f"Using {self.model_name} for processing!")

        # Selecting the model
        model_src:dict = json.load(open("models.json"))["data"]
        model_name = st.radio(
            "Select a model use use for upscaling",
            list(model_src.keys()),
        )

        st.write(f"{model_name}: {model_src[model_name]['description']}")

        # Show the file uploader and submit button
        with st.form("my-form", clear_on_submit=True):
            file = st.file_uploader("FILE UPLOADER", type=["png", "jpg", "jpeg"])
            submitted = st.form_submit_button("Upscale!")
        
        if submitted and file is not None:
            image = Image.open(file)
            
            # Resize the image if it is too large
            if MAX_SIZE is not None and (image.width > MAX_SIZE or image.height > MAX_SIZE):
                st.warning("Your image was resized to save on resources! To avoid this, run AstroSleuth with colab or locally: https://github.com/Aveygo/AstroSleuth#running", icon="⚠️")
                if image.width > image.height:
                    image = image.resize((MAX_SIZE, MAX_SIZE * image.height // image.width))
                else:
                    image = image.resize((MAX_SIZE * image.width // image.height, MAX_SIZE))

            elif image.width > WARNING_SIZE or image.height > WARNING_SIZE:
                st.info("Woah, that image is quite large! You may have to wait a while and/or get unexpected errors!", icon="🕒")

            # Start the queue
            self.queue = FileQueue()
            queue_box = None

            # Wait for the queue to be empty
            while not self.queue.should_run():    
                if queue_box is None:
                    queue_box = st.warning("Experincing high demand, you have been placed in a queue! Please wait...", icon ="🚦") 
                time.sleep(1)
                self.queue.heartbeat()
            
            # Start the heart thread while we are upscaling
            t = threading.Thread(target=self.heart)
            add_script_run_ctx(t)
            t.start()

            # Empty the queue box
            if queue_box is not None:
                queue_box.empty()

            # Start the upscale
            image = self.upscale(image, model_name)

            # Check if the upscale failed for whatever reason
            if image is None:
                st.error("Internal error: Upscaling failed, please try again later?", icon="❌")
                self.close()
                return     

            # Empty the info box
            self.info.empty()

            st.success('Done! Please use the download button to get the highest resolution', icon="🎉")
            
            # Convert to bytes
            b = io.BytesIO()
            file_type = file.name.split(".")[-1].upper()
            file_type = "JPEG" if not file_type in ["JPEG", "PNG"] else file_type
            image.save(b, format=file_type)
            st.download_button("Download Full Resolution", b.getvalue(), file.name, "image/" + file_type)

            # Show preview
            if (image.width*image.height) < (4096*4096):
                st.image(image, caption='Upscaled preview', use_column_width=True)
            
            self.close()
        
    def close(self):
        # Exit from queue and stop running
        self.running = False
        if self.queue is not None:
            self.queue.quit()
            self.queue = None

app = App()
app.render()