import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.web.server.websocket_headers import _get_websocket_headers
from PIL import Image
import time, threading

from file_queue import FileQueue
from main import AstroSleuth

WARNING_SIZE = 4096 # ~7gb

class App:
    def __init__(self):
        self.upscaling = False
        self.queue = None
        self.running = True

    def upscale(self, image):
        self.upscaling = True
        bar = st.progress(0)
        
        model = AstroSleuth()
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
        st.subheader("Upscale astrophotography images with AI")

        with st.form("my-form", clear_on_submit=True):
            file = st.file_uploader("FILE UPLOADER", type=["png", "jpg", "jpeg"])
            submitted = st.form_submit_button("Upscale!")
        
        if submitted and file is not None:
            image = Image.open(file)
            
            if image.width > WARNING_SIZE or image.height > WARNING_SIZE:
                st.info("Woah, that image is quite large! You may encounter memory issues...", icon="🕒")

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

            st.success('Done! Loading result... (To download, right-click image -> Save image as)', icon="🎉")
            
            # Uncomment to get download button (not working on huggingface?)
            #b = io.BytesIO()
            #file_type = file.name.split(".")[-1].upper()
            #file_type = "JPEG" if not file_type in ["JPEG", "PNG"] else file_type
            #image.save(b, format=file_type)
            #st.download_button("Download", b.getvalue(), file.name, "image/" + file_type)

            st.image(image, caption='Upscaled preview', use_column_width=True)
            self.close()
        
    def close(self):
        self.running = False
        if self.queue is not None:
            self.queue.quit()
            self.queue = None

app = App()
app.render()