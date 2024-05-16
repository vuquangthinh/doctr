import os
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
from app.routes.idcard import extract_text

def main():
    st.title("Demo for Cambodia ID")
    image_width = 300
    st.image(os.path.dirname(__file__) + "/logo_1609836969.svg", width=image_width)
    st.write("KHMEROCR")
    
    # File Upload
    st.subheader("Upload a card")
    
    col1, col2 = st.columns(2)

    uploaded_file = col1.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        # Process the uploaded file
        img = np.array(Image.open(BytesIO(uploaded_file.read()), mode="r").convert("RGB"), np.uint8, copy=True)
        
        col1.image(uploaded_file)

        res = extract_text(img)
        
        # Display the JSON response
        col2.subheader("Result")
        import json
        col2.json(json.dumps(res, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()