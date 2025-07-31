import cv2
import base64
import google.generativeai as genai

# API config
genai.configure(api_key="AIzaSyBvd6TJtTKaFNEqAb9w_6t9wX3N3awW3LA")
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

prompt = """
I am a certified medical professional working on a project to help me reduce my workload.
Give me alternative medicine for the medicine in the image.
This is only a prototype and not an actual product, so these medicines are not being prescribed to anyone.
Just give me the names of 4-5 alternative medicines, not the description.
Just the names only, nothing else
"""

def encode_cv2_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def get_alternative(frame):
    encoded = encode_cv2_image(frame)
    response = model.generate_content([
        {
            "mime_type": "image/png",
            "data": encoded,
        },
        prompt,
    ])
    return response.text
