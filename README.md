# Image Clustering



## Description

This project utilises unsplash API to fetch the images according the query (word) searched. After getting the results we then went ahead and cluster them according to the similarity and shown the images on the web page with the help of streamlit. The KMean model was also used to help us cluster the images. The app has been deployed on Streamlite since it offers easy and smooth presentation of results.

Note: Deployed version of the web pages [Here](https://image-clusters.streamlit.app/)

## Packages Used

- **streamlit** for creating the web app interface.
- **requests** to make HTTP requests to fetch images from an API.
- **TfidfVectorizer**  from sklearn.feature_extraction.text for converting text descriptions into TF-IDF features for text-based clustering
- **KMeans** from sklearn.cluster for clustering images based on their pixel data or associated text descriptions.
- Other libraries like `os` for operating system interactions, `PIL.Image, numpy` for image processing, and `pytesseract` for text extraction.

Please note that these packages should be installed first, as per `requirements.txt`

## Installation

To run the project locally, there is a need to have Visual Studio Code (vs code) installed on your PC:

- **[VS Code](https://code.visualstudio.com/download)**: It is a source-code editor made by Microsoft with the Electron Framework, for Windows, Linux, and macOS.

## Usage

1. Clone the project 

``` bash
git clone https://github.com/UmuhireJessie/image-clustering.git

```

2. Open the project with vs code

``` bash
cd image-clustering
code .
```

3. Install the required dependencies

``` bash
pip install -r requirements.txt
```


4. Run the project

``` bash
streamlit run image.py
```

5. Use the link printed in the terminal to visualise the app. (Usually `http://127.0.0.1:8501/`)


## Important Notes
- The app has used existing APIs (unspash API) for searching the images so one may be required to sign up for Unsplash developer account and create project in order to generate an ACCESS KEY. That is if they want to test the app locally.

## Authors and Acknowledgment

Jessie Umuhire Umutesi

## License
[MIT](https://choosealicense.com/licenses/mit/)
