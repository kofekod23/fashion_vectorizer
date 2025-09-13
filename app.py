import streamlit as st
import os, zipfile
from dotenv import load_dotenv
from PIL import Image
from src.pipeline import process_one_image
from src.weaviate_utils import connect_weaviate, create_collections_if_needed, insert_tenue, insert_vetement
from dotenv import load_dotenv  # charge les variables d'environnement du .env si dispo
load_dotenv()  # charge les variables d'environnement du .env si dispo

def extract_zip(uploaded_zip, out_dir):
    with zipfile.ZipFile(uploaded_zip) as zf:
        zf.extractall(out_dir)

def make_dirs():
    os.makedirs("content/train", exist_ok=True)
    os.makedirs("content/crops", exist_ok=True)

st.title("Traitement tenues 🕺🏼/ vêtements automatisé ⚙️")

uploaded_zip = st.file_uploader("Upload ZIP images tenues", type="zip")

# 1. Récupéreration valeurs .env
default_url = os.getenv("WEAVIATE_URL", "")
default_apikey = os.getenv("WEAVIATE_APIKEY", "")

# 2. Permettre override depuis Streamlit seulement si pas présent
WCS_URL = st.text_input("Weaviate Cloud URL", value=default_url)
WCS_APIKEY = st.text_input("Weaviate API Key", value=default_apikey, type="password")
client = None

if WCS_URL and WCS_APIKEY:
    client = connect_weaviate(WCS_URL, WCS_APIKEY)
    create_collections_if_needed(client)

if uploaded_zip and client:
    make_dirs()
    extract_zip(uploaded_zip, "content/train")
    st.success("Zip extrait et base connectée !")

    for imgfile in os.listdir("content/train"):
        img_path = os.path.join("content/train", imgfile)
        if not imgfile.lower().endswith((".jpg",".png")): continue
        img = Image.open(img_path).convert("RGB")
        results = process_one_image(img, imgfile)
        for item in results:
            if item["collection_name"]=="Tenues_v2025_c":
                insert_tenue(client, item)
            elif item["collection_name"]=="Vetements_v2025_c":
                crop_img = item["image_data"]
                crop_path = os.path.join("content/crops", item["properties"]["imageCroppedFile"])
                crop_img.save(crop_path)
                insert_vetement(client, item)
    st.success("Pipeline terminé ! Résultats sauvegardés et insérés.")

    crops = os.listdir("content/crops")
    if crops:
        st.image([os.path.join("content/crops", c) for c in crops[:6]], caption="Aperçu premiers crops générés")
    st.write(f"{len(os.listdir('content/train'))} tenues traitées / {len(os.listdir('content/crops'))} crops générés")
