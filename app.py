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

import cohere #Retrieval
cohere_api_key = st.secrets["COHERE_API_KEY"]
co = cohere.ClientV2(api_key=cohere_api_key)

gemini_api_key = userdata.get("GEMINI_API_KEY")  #LLM
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
   # st.write("files in DB:")
    #for file in get_data_files():
     #   st.markdown("- " + file)
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



