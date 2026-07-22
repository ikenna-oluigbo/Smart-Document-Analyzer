# Contact me on callme-smartai@proton.me 
# Program to analyze document, summarize, and provide chat responses, suggest related articles online, and download images from documents

#Importing Libraries 
import fitz
import pdfplumber
from openai import OpenAI
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, OllamaEmbeddingFunction
import time

OPENAI_API_KEY = "=== Your OpenAI API KEY ==="

pdfupload_path = "./pdf_files/"

class LLMModel:
    def __init__(self, model_type = "openai"): 
        self.model_type = model_type 
        if model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = "gpt-5.4"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llava"
        
    def generate_completion(self, message):
        try:
            response = self.client.chat.completions.create(
                messages=message,
                model=self.model_name             
                )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating content: {str(e)}"


class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small", 
                dimensions=768
            )
            
        elif model_type == "ollama_local":
            self.embedding_fn = OllamaEmbeddingFunction(
                url="http://localhost:11434", 
                model_name="qwen3-embedding"
            )
        
        elif model_type == "ollama_openai":
            self.embedding_fn = OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text"
            )    
            

def select_model_embedding():
    # LLM Model Selection
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT")
    print("2. Ollama")
    while True:
        llm_choice = eval(input("Enter your choice of model (1 or 2): "))
        if llm_choice in [1, 2]:
            model_type = "openai" if llm_choice == 1 else "ollama"
            print(f"\nYou have selected: {model_type} LLM model")
            break
        print("Selection must be 1 or 2")
        
    # Select Embedding Model
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embeddings")
    print("2. Ollama local (Qwen 3 Embedding)")
    print("3. Ollama OpenAI (nomic-embed-text Embedding)")
    while True:
        emb_choice = input("Enter your choice of embedding model (1, 2, 3): ").strip()
        if emb_choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "ollama_local", "3": "ollama_openai"}[emb_choice]
            print(f"\nYou have selected: {embedding_type} Embedding model")
            break
        print("Selection must be 1, 2, or 3")
        
    return model_type, embedding_type


# Function to split text
def make_chunks(texts : list, chunk_size: int = 1000, chunk_overlap: int = 200):
    chunks = []
    start = 0
    while start < len(texts):
        end = start + chunk_size
        chunks.append(texts[start:end])
        start = end - chunk_overlap
    
    return chunks

# Function to parse pdf texts
def read_pdfuploaded_text():
    all_documents = []
    for pdf in os.listdir(pdfupload_path):
        doc = fitz.open(pdfupload_path + pdf)
        print(f"Total lenght of Document Pages: {len(doc)} ")
        for i in range(len(doc)):
            curr_page = doc[i]
            text = [curr_page.get_text()]
            all_documents.extend(text)
            
    all_documents = " ".join(all_documents)         #Merge all strings into one
    
    return all_documents


#Extract Title
def read_pdf_title():
    all_titles = []
    #Method 1: Extract from Metadata
    for pdf in os.listdir(pdfupload_path):
        with pdfplumber.open(pdfupload_path + pdf) as p:
            page = p.pages[0]
            # Filter characters with a size greater than a threshold (e.g., 20)
            # Adjust threshold based on your PDF's font sizes
            filtered_page = page.filter(lambda x: x.get("size", 0) > 12)
            title = filtered_page.extract_text()
            all_titles.append(title)
            print(f"Heuristic Title: {title}")
    
    return all_titles


#Image Extraction        
def read_pdf_image():
    for pdf in os.listdir(pdfupload_path):
        doc = fitz.open(pdfupload_path + pdf)
        for page_num in range(len(doc)):   #Loop through all pages and find images from images (jpeg, jpg, png. OCR images cant be seen)
            page = doc[page_num]
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                with open(f"image_{page_num}.{image_ext}", "wb") as f:
                    f.write(image_bytes)
                    

#Split Text and assign ID
def split_text_id():
    messages_with_id = dict()
    texts = read_pdfuploaded_text()
    chunks = make_chunks(texts=texts, chunk_size=1000, chunk_overlap=200)
    for i, chunk in enumerate(chunks):
        idx = i + 1
        messages_with_id[str(idx)] = chunk
    
    return messages_with_id


def setup_chromadb(texts, embedding_model):
    collection_name = "doc_analyzer"
    db_path = "./ChromaDB/llm_docdb"
    
    client = chromadb.PersistentClient(path=db_path)
  
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_model.embedding_fn
    )
    
    
    ids = list(texts.keys())
    documents = list(texts.values())
    
    collection.add(ids=ids,
                      documents=documents)
   
    print("\nDocuments added to ChromaDB collection successfully!")
    return collection


def find_related_chunks(query, collection, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    #print(results)
    return results["documents"][0]
    

def augment_prompt(query, relevant_chunks):
    context = "\n".join([chunks for chunks in relevant_chunks])
    #print(relevant_chunks)
    augmented_prompt = f"Context: \n{context} \n\nQuestion: {query}:"
    
    print("Augmented prompt: ⤵️")
    print(augmented_prompt)
    
    return augmented_prompt


def rag_pipeline(query, collection, llm_model, top_k=5):
    print(f"\nNow processing query: {query}")
    
    relevant_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, relevant_chunks)
    
    response = llm_model.generate_completion(
        [
        {"role": "system", "content": f"""You are a very smart and assistant.
                    You can answer questions about Subgraphs, Graphs embedding, Knowledge Graphs, Graph Neural Networks, Node translation, and lots more. 
                    You answers questions that are directly related to the sources/documents given, but can also be to add more knowledge to the source document. 
                    Use your base knowledge and the context information in {augmented_prompt} to answer questions"""},
        {"role": "user", "content": query}
        ]
    )
    
    print("\nGenerated response:")
    print(response)

    references = [chunk for chunk in relevant_chunks]        
    return response, references


def main():
    print("Starting the RAG pipeline...")

    # Select models
    llm_type, embedding_type = select_model_embedding()

    # Initialize models
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)

    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"Using Embeddings: {embedding_type.upper()}")

    
    print("\nSplitting data into IDs and Chunks")
    texts = split_text_id()

    # Setup ChromaDB
    collection = setup_chromadb(texts, embedding_model)

    # Process Queries:
    while True:
        print("\n")
        print("=== Enter your Query ===")
        print("=== Ask questions about your files to gain more Insights ===")
        print("=== To download images in the document, type images ===")
        print("=== To search for related papers, type papers ===")
        print("=== Type Quit to exit program ===")
        query = input("You: ")
        
        if query == "quit".lower():
            print("Goodbye! 👋👋")
            break
            
        elif query == "papers".lower():
            print("Searching the web ...")
            from search_articles import execute_search_papers
            all_titles = read_pdf_title()
            
            for title in all_titles:
                tar = title.split(" ")[:10]
                print("\n", tar)
                execute_search_papers(" ".join(tar))
            
        elif query == "images".lower():
            read_pdf_image()
            print("All Images Downloaded Successfully! 😎")
            
        else:
            print("\n" + "=" * 50)
            print(f"Processing query: {query}")
            response, references = rag_pipeline(query, collection, llm_model)

            print("\nFinal Results:")
            print("-" * 30)
            print("Response:", response)
            print("\nReferences used:")
            for ref in references:
                print(f"- {ref}")
            print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat Session ended by user. Goodbye! 👋")
