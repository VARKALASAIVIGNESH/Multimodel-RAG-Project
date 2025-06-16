# Multimodal RAG Application
https://drive.google.com/file/d/1m48w5qIr9mP_OYj0DFNhs0QEqI2-f4J4/view?usp=sharing
## Overview

This project implements a Multimodal Retrieval-Augmented Generation (RAG) pipeline that processes both text and image inputs from PDF documents to answer user queries. The application extracts text and images, embeds them into a shared multimodal vector space using CLIP, stores the embeddings in ChromaDB for semantic retrieval, and generates answers using a combination of CLIP-generated image descriptions and LLaMA (via Groq API). The app avoids using OCR for image content, as per the task requirements, and provides a user-friendly web interface built with Flask.

### Features
- Upload a PDF and extract text and images.
- Ask questions about the PDF content, with answers generated based on both text and images.
- Display retrieved images alongside the answer for a multimodal experience.
- Robust error handling for extraction, embedding, and answer generation.

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate a Virtual Environment**:
   - On Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```
   pip install flask pymupdf groq sentence-transformers chromadb pillow
   ```

4. **Add Your Groq API Key**:
   - Open `app.py` in a text editor.
   - Replace `"your-groq-api-key"` with your actual Groq API key:
     ```python
     groq_client = Groq(api_key="your-actual-groq-api-key")
     ```

5. **Run the Application**:
   ```
   python app.py
   ```
   - Access the app at `http://127.0.0.1:5000`.

## Usage

1. **Upload a PDF**:
   - Navigate to `http://127.0.0.1:5000`.
   - Use the upload form to select a PDF file (e.g., `Assignment for interns.pdf`).
   - Click "Upload" to extract text and images.

2. **View Extracted Text**:
   - The extracted text from the PDF will be displayed on the page.

3. **Ask a Question**:
   - Enter a question about the PDF content (e.g., "What is the hint?").
   - Click "Ask" to get an answer based on the text and images.

4. **View the Answer**:
   - The answer will be displayed, along with any relevant images retrieved from the PDF.

5. **Start Over**:
   - Click "Start Over" to upload a new PDF and ask more questions.

## Project Structure

```
TASK-2/
│
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # HTML template with internal CSS
├── static/
│   └── images/           # Directory to store extracted images
├── documents/            # Directory to store uploaded PDFs
├── chroma_db/            # Directory for ChromaDB vector storage
└── venv/                 # Virtual environment
```

## Libraries Used

- **Flask**: Web framework for building the application.
- **PyMuPDF**: For extracting text and images from PDFs.
- **Groq**: To access LLaMA for text generation via the Groq API.
- **Sentence-Transformers**: For CLIP (`clip-ViT-B-32`) to generate multimodal embeddings.
- **ChromaDB**: Vector database for storing and retrieving embeddings.
- **Pillow**: For handling image processing.

## Thought Process

- **Multimodal RAG Pipeline**:
  - **Extraction**: Used `PyMuPDF` to extract text and images from PDFs without relying on OCR for image content, as per the task requirements.
  - **Embedding**: Employed CLIP (`clip-ViT-B-32`) to embed both text and images into a shared vector space, enabling multimodal retrieval.
  - **Retrieval**: Stored embeddings in ChromaDB and performed semantic retrieval based on user queries to fetch relevant text and images.
  - **Answer Generation**: Due to Groq’s LLaMA being text-only, used CLIP to generate image descriptions and combined them with LLaMA for answer generation. This is a workaround since a true multimodal LLM wasn’t feasible with the given constraints.

- **Error Handling**:
  - Added robust error handling for cases like empty ChromaDB collections, failed extractions, and embedding issues.
  - Fixed a bug where ChromaDB’s `delete` method failed on empty collections by adding a check for existing IDs.

- **Frontend Design**:
  - Built a single-page web app with `index.html` using internal CSS for a modern, user-friendly interface.
  - Displayed retrieved images alongside answers to emphasize the multimodal aspect.

## Limitations and Future Improvements

- **Multimodal LLM Limitation**:
  - Groq’s LLaMA is text-only, so images are processed indirectly via CLIP-generated descriptions. This doesn’t fully meet the requirement of using a multimodal LLM to process images directly.
  - **Future Improvement**: Use a true multimodal LLM like LLaVA or Florence-2, which can process text and images together. This could be achieved by:
    - Running LLaVA locally if compute resources (e.g., a GPU) are available.
    - Using an API that supports multimodal models, such as Hugging Face Spaces hosting LLaVA.

- **Image Descriptions**:
  - The current app uses a placeholder description for images due to the lack of a multimodal LLM. A proper multimodal LLM would generate more meaningful image descriptions or directly use image content in answer generation.

- **Performance**:
  - Embedding and retrieval can be slow for large PDFs with many images. Optimizing the pipeline (e.g., batch processing embeddings) could improve performance.

## Example Usage

- **PDF**: `Assignment for interns.pdf`
- **Query**: "What is the hint?"
- **Answer**:
  ```
  The hint is to use ChatGPT for architecture and technical help, but you should not copy-paste from there and must write your own code.
  ```

- **Query**: "What are the requirements?"
- **Answer**:
  ```
  The requirements are:  
  1. Input: Either an image (.jpg, .png) or a text prompt (plain text).  
  2. Processing: For photo input, extract the object clearly (background removal optional but preferred); for text input, use any open-source text-to-3D model generation method.  
  3. Output: A downloadable .stl or .obj file, and a simple visualization (e.g., a basic 3D plot using matplotlib or pyrender).  
  4. Code and Documentation: Code must be in Python, use virtualenv, document all dependencies in requirements.txt, and include a README.md with steps to run, libraries used, and your thought process.
  ```

## Troubleshooting

- **ChromaDB Error on First Upload**:
  - If you see an error like `Expected IDs to be a non-empty list, got 0 IDs in delete`, it means the ChromaDB collection is empty. This has been fixed by checking for existing IDs before deletion.
- **Groq API Key**:
  - Ensure your Groq API key is correctly added in `app.py`. Without it, answer generation will fail.
- **Dependencies**:
  - If a library fails to install, ensure you’re using a compatible Python version (e.g., Python 3.8 or higher) and check for package conflicts.

## Acknowledgments

- Built as part of a task to create a Multimodal RAG application.
- Utilized open-source libraries like `sentence-transformers` and `chromadb` for multimodal embeddings and vector storage.
- Leveraged Groq’s API for LLaMA-based text generation.
