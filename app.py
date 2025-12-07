import os
import streamlit as st
from google import genai
import requests
import os
import io
import base64
import PIL
import tqdm
import time
import numpy as np
from utils import get_data_files


import cohere #Retrieval
cohere_api_key = st.secrets["COHERE_API_KEY"]
co = cohere.ClientV2(api_key=cohere_api_key)

gemini_api_key = st.secrets["GEMINI_API_KEY"] #vision LLM
client = genai.Client(api_key=gemini_api_key)


# Some helper functions to resize images and to convert them to base64 format
max_pixels = 1568*1568  #Max resolution for images

# Resize too large images
def resize_image(pil_image):
    org_width, org_height = pil_image.size

    # Resize image if too large
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

# Convert images to a base64 string before sending it to the API
def base64_from_image(img_path):
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"

    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data

# Several images from https://www.appeconomyinsights.com/
images = {
    "tesla.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
    "netflix.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
    "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    "google.png": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
    "accenture.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
    "tecent.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png",
   # 'pilot.png':'https://www.easel.ly/blog/wp-content/uploads/2019/05/People-following-directions-with-text-and-illustrations-do-323-better-than-those-following-directions-without-illustrations.-1.png',
    #'emp.png' :'https://www.easel.ly/blog/wp-content/uploads/2019/05/text-vs-visuals-which-is-more-effective.png'
}

img_folder = "PILOT"
os.makedirs(img_folder, exist_ok=True)

img_paths = []
doc_embeddings = []
for name, url in tqdm.tqdm(images.items()):
    img_path = os.path.join(img_folder, name)
    img_paths.append(img_path)

    # Download the image
    if not os.path.exists(img_path):
        response = requests.get(url)
        response.raise_for_status()

        with open(img_path, "wb") as fOut:
            fOut.write(response.content)
            
    # Get the base64 representation of the image
    api_input_document = {
        "content": [
            {"type": "image", "image": base64_from_image(img_path)},
        ]
    }

    # Call the Embed v4.0 model with the image information
    api_response = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        inputs=[api_input_document],
    )

    # Append the embedding to our doc_embeddings list
    emb = np.asarray(api_response.embeddings.float[0])
    doc_embeddings.append(emb)

doc_embeddings = np.vstack(doc_embeddings)


# Search allows us to find relevant images for a given question using Cohere Embed v4
def search(question, max_img_size=800):
    # Compute the embedding for the query
    api_response = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[question],
    )

    query_emb = np.asarray(api_response.embeddings.float[0])

    # Compute cosine similarities
    cos_sim_scores = np.dot(query_emb, doc_embeddings.T)

    # Get the most relevant image
    top_idx = np.argmax(cos_sim_scores)
    max_score = cos_sim_scores[top_idx]
    print("Most relevant image:", hit_img_path,max_score)
    # Show the images
    print("Question:", question)

    hit_img_path = img_paths[top_idx]

  
    return hit_img_path

# Answer the question based on the information from the image
# Here we use Gemini 2.5 as powerful Vision-LLM
def answer(question, img_path):
    prompt = [f"""Answer the question based on the following image.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}""", PIL.Image.open(img_path)]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text
    print("LLM Answer:", answer)



st.title("Vision-RAG - Cohere Embed v4 🤝 Gemini Flash")

st.sidebar.title("App Description")

with st.sidebar:
   # st.write('anthropic.claude-3-sonnet-20240229-v1:0 amazon.titan-embed-text-v1')
   # st.button('New Chat', on_click=reset_conversation)
    st.write("images in DB:")
    for img in get_data_files():
        st.markdown("- " + img)
    st.write('Made by Noa Cohen')


prompt = st.text_input("ask a question")

# If the user hits enter
if prompt:
   top_image_path = search(question)

# Use the image to answer the query
   st.subheader("Generated response:")

   st.write(answer(question, top_image_path))
   image = PIL.Image.open(hit_img_path)
   max_size = (max_img_size, max_img_size)  # Adjust the size as needed
   image.thumbnail(max_size)

   st.image(image, caption="Uploaded Image.", use_column_width=True)



