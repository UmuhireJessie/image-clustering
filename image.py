import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Set up the Unsplash API key
API_KEY_UNSPLASH = "Jzv-RluJG-3vj4RvvMSUx_qjWm0SzjzyxWKgVZGROVc"

def retrieve_images(search_term, image_count=10):
    api_endpoint = f"https://api.unsplash.com/search/photos/?query={search_term}&client_id={API_KEY_UNSPLASH}&per_page={image_count}"
    response = requests.get(api_endpoint)
    result_json = response.json()
    image_urls = [img['urls']['regular'] for img in result_json['results']]
    image_descriptions = [img.get('description') for img in result_json['results']]
    return image_urls, image_descriptions

def save_images(image_urls, target_folder='downloaded_images'):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for index, url in enumerate(image_urls):
        img_data = requests.get(url).content
        with open(f"{target_folder}/img_{index}.jpg", 'wb') as img_file:
            img_file.write(img_data)

def group_images(folder_path, image_texts, cluster_nums=3, text_based=False, image_size=(100, 100)):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Ensure image_texts align with files in the folder
    if text_based:
        # Filter out None descriptions and corresponding files
        filtered_files_and_texts = [(file, text) for file, text in zip(files, image_texts) if text is not None]
        if not filtered_files_and_texts:
            st.write("No valid descriptions available for text-based clustering.")
            return [], []
        filtered_files, valid_texts = zip(*filtered_files_and_texts)
        
        text_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
        text_clusters = text_vectorizer.fit_transform(valid_texts)
        cluster_algo = KMeans(n_clusters=cluster_nums)
        cluster_algo.fit(text_clusters)
        labels = cluster_algo.labels_
        return filtered_files, labels
    else:
        images = []
        for file_name in files:
            image_path = os.path.join(folder_path, file_name)
            with Image.open(image_path) as img:
                img_resized = img.resize(image_size)
                img_array = np.array(img_resized).flatten()
                images.append(img_array)
        
        img_cluster = KMeans(n_clusters=cluster_nums)
        img_cluster.fit(images)
        labels = img_cluster.labels_
        return files, labels

# Streamlit interface setup
st.title("Explore and Cluster Unsplash Images")

search_query = st.text_input("Search for images:", "nature")
images_to_fetch = st.number_input("How many images to retrieve?", min_value=1, max_value=20, value=10)
clusters_to_create = st.number_input("How many clusters?", min_value=2, max_value=10, value=3)
cluster_by_description = st.checkbox("Use descriptions for clustering?")

if st.button("Start Process"):
    st.write("Collecting images...")
    urls, descriptions = retrieve_images(search_query, images_to_fetch)
    
    st.write(f"Downloading {len(urls)} images...")
    save_images(urls)
    
    st.write("Clustering images now...")
    files, labels = group_images('downloaded_images', descriptions, clusters_to_create, cluster_by_description)
    
    st.write("Cluster assignment complete.")
    for label in set(labels):
        st.subheader(f"Cluster {label + 1}")
        for file, cluster_label in zip(files, labels):
            if cluster_label == label:
                st.image(os.path.join('downloaded_images', file), width=300)

