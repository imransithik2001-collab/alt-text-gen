import io
import csv
from typing import List, Tuple
import streamlit as st
from PIL import Image
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Alt Text Generator (BLIP)", page_icon="🖼️", layout="wide")

# -------- Load model once --------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_model()

# -------- Helpers --------
def fetch_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, stream=True, timeout=20)
    resp.raise_for_status()
    img = Image.open(resp.raw).convert("RGB")
    return img

def cap_words(text: str, max_words: int = 10) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words])

@torch.inference_mode()
def generate_alt_text_from_image(img: Image.Image, max_words: int = 10) -> str:
    try:
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=25)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return cap_words(caption, max_words=max_words)
    except Exception as e:
        return f"Error: {e}"

def batch_from_urls(urls: List[str], max_words: int = 10) -> List[Tuple[str, str]]:
    results = []
    for url in urls:
        u = url.strip()
        if not u:
            continue
        try:
            img = fetch_image_from_url(u)
            alt = generate_alt_text_from_image(img, max_words=max_words)
            results.append((u, alt))
        except Exception as e:
            results.append((u, f"Error: {e}"))
    return results

def to_csv_bytes(rows: List[Tuple[str, str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["image_url", "alt_text"])
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")

# -------- UI --------
st.markdown("<h1 style='text-align:center;'>🖼️ Alt Text Generator (BLIP)</h1>", unsafe_allow_html=True)
st.write("Generate short, SEO-friendly alt text from image **URLs** or **uploads**.")

with st.sidebar:
    st.header("⚙️ Settings")
    max_words = st.slider("Max words in alt text", 5, 16, 10)
    st.caption("Best practice: keep alt text concise and descriptive.")

tab1, tab2, tab3 = st.tabs(["🔗 Single URL", "📝 Multiple URLs", "📂 Upload Images"])

# CSS for hover effect
st.markdown("""
    <style>
    .alt-card:hover {
        background-color: #f9f9f9;
        transform: scale(1.01);
        transition: 0.3s ease;
    }
    .alt-card {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid #e5e5e5;
    }
    a {
        color: #0066cc;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- Single URL ---
with tab1:
    url = st.text_input("Paste an Image URL")
    if st.button("Generate", use_container_width=True):
        if not url.strip():
            st.warning("Please paste an image URL.")
        else:
            try:
                img = fetch_image_from_url(url.strip())
                alt = generate_alt_text_from_image(img, max_words=max_words)
                st.markdown(f"""
                    <div class="alt-card">
                        <p><b>Alt Text:</b> {alt}</p>
                        <p><a href="{url.strip()}" target="_blank">View Image</a></p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not load/process image: {e}")

# --- Multiple URLs ---
with tab2:
    st.write("Paste one image URL **per line**:")
    urls_text = st.text_area("URLs", height=160, placeholder="https://.../image1.jpg\nhttps://.../image2.png")
    if st.button("Generate for All", use_container_width=True):
        urls = [u for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.warning("Please paste at least one URL.")
        else:
            results = batch_from_urls(urls, max_words=max_words)
            for u, alt in results:
                st.markdown(f"""
                    <div class="alt-card">
                        <p><b>Alt Text:</b> {alt}</p>
                        <p><a href="{u}" target="_blank">View Image</a></p>
                    </div>
                """, unsafe_allow_html=True)
            csv_bytes = to_csv_bytes(results)
            st.download_button("⬇ Download CSV", data=csv_bytes, file_name="alt_text_results.csv", mime="text/csv")

# --- Upload Images ---
with tab3:
    uploads = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Generate for Uploads", use_container_width=True):
        if not uploads:
            st.warning("Please upload at least one image.")
        else:
            rows = []
            for f in uploads:
                try:
                    img = Image.open(f).convert("RGB")
                    alt = generate_alt_text_from_image(img, max_words=max_words)
                    st.markdown(f"""
                        <div class="alt-card">
                            <p><b>File:</b> {f.name}</p>
                            <p><b>Alt Text:</b> {alt}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    rows.append((f.name, alt))
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if rows:
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["filename", "alt_text"])
                for r in rows:
                    w.writerow(r)
                st.download_button("⬇ Download CSV", data=buf.getvalue().encode("utf-8"),
                                   file_name="alt_text_uploads.csv", mime="text/csv")








