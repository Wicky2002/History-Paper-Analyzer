import os
import fitz  # PyMuPDF for fast text extraction
from pandukabhaya import Converter # Sinhala FM-to-Unicode converter

# Define absolute or relative paths based on your current project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def convert_pdf_to_unicode(pdf_path, output_path):
    print(f"Extracting: {os.path.basename(pdf_path)}...")
    
    # Initialize the converter with the FM Abhaya mapping dictionary
    converter = Converter("fm_abhaya")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Explicitly cast to string to resolve Pylance typing warnings
        raw_text = str(page.get_text("text"))
        
        # Process line by line to avoid breaking the converter's regex
        for line in raw_text.split('\n'):
            line = line.strip()
            
            # Filter out empty lines and isolated page numbers
            if not line or line.isnumeric():
                continue
                
            # Convert line by line
            unicode_text = converter.convert(line)
            full_text += unicode_text + "\n"
        
        full_text += "\n" # Add paragraph break between pages
        
    # Save the processed Sinhala text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Saved cleanly to: {os.path.basename(output_path)}\n")

def main():
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Iterate through all PDFs in the raw folder
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, filename)
            # Change file extension from .pdf to .txt
            output_filename = filename.replace(".pdf", ".txt")
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            
            convert_pdf_to_unicode(pdf_path, output_path)

if __name__ == "__main__":
    main()