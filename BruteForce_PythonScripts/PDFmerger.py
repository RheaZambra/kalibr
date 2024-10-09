import os
from PyPDF2 import PdfMerger

def merge_pdfs(folder_path, output_filename):
    # Create a PdfMerger object
    merger = PdfMerger()

    # Get all PDF files from the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()  # Optional: sort files if needed (e.g., by name or number)

    # Append each PDF to the merger
    for pdf in pdf_files:
        pdf_path = os.path.join(folder_path, pdf)
        merger.append(pdf_path)

    # Write out the merged PDF
    with open(output_filename, 'wb') as output_pdf:
        merger.write(output_pdf)

    print(f"Merged PDF saved as {output_filename}")

# Example usage
merge_pdfs('Big_AccelNoise', 'BigCalib_AccelNoise.pdf')

