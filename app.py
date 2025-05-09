from flask import Flask, render_template, request, make_response
import os
import fitz  # PyMuPDF for PDF extraction
from groq import Groq  # For LLaMA via Groq API
from sentence_transformers import SentenceTransformer  # For CLIP embeddings
import chromadb  # For vector storage
from PIL import Image
import io
import re
import logging

# Set up logging for error handling and feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize the Groq client with your API key
groq_client = Groq(api_key="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")  



# Directory to store uploaded PDFs and images
DOCUMENT_DIR = "documents"
IMAGE_DIR = "static/images"
if not os.path.exists(DOCUMENT_DIR):
    os.makedirs(DOCUMENT_DIR)
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Initialize CLIP model for multimodal embeddings
clip_model = SentenceTransformer('clip-ViT-B-32')

# Initialize ChromaDB for vector storage
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="multimodal_collection")

# Global variables to store extracted content
extracted_text = ""
extracted_images = []
image_paths = []

def clean_text(text):
    """Clean up extracted text with generic rules for common OCR errors and formatting issues."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    text = text.replace(' rn ', ' m ')
    text = text.replace('1', 'l')
    text = text.replace('0', 'o')
    text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
    text = re.sub(r'(\w+)(gmail\.com)', r'\1@gmail.com', text)
    if text and re.search(r'[a-zA-Z0-9]$', text):
        logging.warning("Extracted text appears genuinely truncated. Last character suggests a cut-off word.")
    return text

def extract_content(pdf_path):
    """Extract text and images from a PDF file."""
    global extracted_text, extracted_images, image_paths
    extracted_text = ""
    extracted_images = []
    image_paths = []

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # Extract text
                raw_text = page.get_text("text")
                extracted_text += clean_text(raw_text) + " "

                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    extracted_images.append(image)
                    # Save image to disk for display
                    image_path = os.path.join(IMAGE_DIR, f"image_{len(image_paths)}.png")
                    image.save(image_path)
                    image_paths.append(f"images/image_{len(image_paths)-1}.png")

        extracted_text = extracted_text.strip()
        if not extracted_text and not extracted_images:
            logging.warning("No text or images extracted from the PDF.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error in content extraction from {pdf_path}: {str(e)}")
        return False

def embed_content():
    """Embed text and images into a shared multimodal vector space using CLIP."""
    try:
        # Clear existing data in the collection if there are any IDs
        existing_ids = collection.get()['ids']
        if existing_ids:  # Only call delete if there are IDs to delete
            collection.delete(ids=existing_ids)

        # Embed text
        if extracted_text:
            text_embedding = clip_model.encode(extracted_text, convert_to_tensor=False)
            collection.add(
                documents=[extracted_text],
                embeddings=[text_embedding.tolist()],
                ids=["text_0"],
                metadatas=[{"type": "text"}]
            )

        # Embed images
        for i, image in enumerate(extracted_images):
            image_embedding = clip_model.encode(image, convert_to_tensor=False)
            collection.add(
                documents=[f"image_{i}"],
                embeddings=[image_embedding.tolist()],
                ids=[f"image_{i}"],
                metadatas=[{"type": "image"}]
            )
        logging.info("Content embedded and stored in vector store")
    except Exception as e:
        logging.error(f"Error in embedding content: {str(e)}")
        raise

def semantic_retrieval(query, top_k=3):
    """Perform semantic retrieval based on the query."""
    try:
        query_embedding = clip_model.encode(query, convert_to_tensor=False)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_texts = []
        retrieved_image_indices = []

        for metadata, document, doc_id in zip(results["metadatas"][0], results["documents"][0], results["ids"][0]):
            if metadata["type"] == "text":
                retrieved_texts.append(document)
            else:
                img_idx = int(doc_id.split("_")[-1])
                retrieved_image_indices.append(img_idx)

        if not retrieved_texts and not retrieved_image_indices:
            logging.warning("No relevant content retrieved for the query")
        
        return retrieved_texts, retrieved_image_indices
    except Exception as e:
        logging.error(f"Error in semantic retrieval: {str(e)}")
        return [], []

def generate_image_description(image):
    """Generate a description of the image using CLIP (placeholder for multimodal LLM)."""
    try:
        return "A visual representation related to the document content."
    except Exception as e:
        logging.error(f"Error in image description generation: {str(e)}")
        return "Error describing image."

def generate_answer_with_llama(query, retrieved_texts, retrieved_image_indices):
    """Generate an answer using LLaMA via Groq API, incorporating text and image descriptions."""
    try:
        # Combine retrieved texts
        context = "\n".join(retrieved_texts) if retrieved_texts else "No relevant text found."

        # Generate image descriptions
        image_descriptions = []
        for idx in retrieved_image_indices:
            if idx < len(extracted_images):
                desc = generate_image_description(extracted_images[idx])
                image_descriptions.append(f"Image {idx+1}: {desc}")

        image_context = "\n".join(image_descriptions) if image_descriptions else "No relevant images found."

        # Create a prompt for LLaMA
        prompt = f"Given the following context:\n\nText Context:\n{context}\n\nImage Context:\n{image_context}\n\nAnswer the question: {query}\n\nProvide a concise and accurate answer based on the context."

        # Call the Groq API with LLaMA
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            max_tokens=150,
            temperature=0.5
        )

        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logging.error(f"Error in answer generation with LLaMA: {str(e)}")
        return f"Error generating answer: {str(e)}"

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to avoid 404 errors."""
    return make_response("", 204)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle PDF upload and content extraction."""
    if 'pdf_file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['pdf_file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")
    
    if file and file.filename.endswith('.pdf'):
        # Clear existing images in the static/images directory
        for existing_file in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, existing_file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except PermissionError:
                    return render_template('index.html', error=f"Cannot delete existing file '{existing_file}' because it is in use by another process.")
        
        # Ensure the documents directory exists
        if not os.path.exists(DOCUMENT_DIR):
            os.makedirs(DOCUMENT_DIR)
        
        # Clear existing PDFs in the documents directory
        for existing_file in os.listdir(DOCUMENT_DIR):
            file_path = os.path.join(DOCUMENT_DIR, existing_file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except PermissionError:
                    return render_template('index.html', error=f"Cannot delete existing file '{existing_file}' because it is in use by another process.")
        
        # Save the uploaded PDF
        file_path = os.path.join(DOCUMENT_DIR, file.filename)
        file.save(file_path)
        
        # Extract text and images
        if not extract_content(file_path):
            return render_template('index.html', error="No text or images could be extracted from the PDF.")
        
        # Embed the extracted content
        try:
            embed_content()
        except Exception as e:
            return render_template('index.html', error=f"Error embedding content: {str(e)}")
        
        return render_template('index.html', extracted_text=extracted_text, filename=file.filename)
    
    return render_template('index.html', error="Please upload a valid PDF file.")

@app.route('/query', methods=['POST'])
def query():
    """Handle query submission and answer generation."""
    query = request.form['query']
    
    # Ensure a PDF has been uploaded
    pdf_files = [f for f in os.listdir(DOCUMENT_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        return render_template('index.html', error="No PDF found. Please upload a PDF first.")
    
    pdf_path = os.path.join(DOCUMENT_DIR, pdf_files[0])
    if not extract_content(pdf_path):
        return render_template('index.html', error="No text or images could be extracted from the PDF.")
    
    # Embed the content
    try:
        embed_content()
    except Exception as e:
        return render_template('index.html', error=f"Error embedding content: {str(e)}")
    
    # Perform semantic retrieval
    retrieved_texts, retrieved_image_indices = semantic_retrieval(query)
    
    # Generate the answer
    answer = generate_answer_with_llama(query, retrieved_texts, retrieved_image_indices)
    
    # Prepare images for display
    display_images = [image_paths[idx] for idx in retrieved_image_indices if idx < len(image_paths)]
    
    return render_template('index.html', extracted_text=extracted_text, filename=pdf_files[0], query=query, answer=answer, images=display_images)

if __name__ == '__main__':
    app.run(debug=False)