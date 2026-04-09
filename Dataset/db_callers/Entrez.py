import os

def EntrezCaller(EntrezIds):
    from Bio import Entrez
    
    Entrez.email = os.getenv('email')  # NCBI requires this
    
    # Join IDs into a comma-separated string
    id_str = ','.join(EntrezIds)
    
    # Fetch summaries in batch
    handle = Entrez.esummary(db="gene", id=id_str)
    records = Entrez.read(handle)
    
    # Extract symbols
    symbols = [doc['Name'] for doc in records['DocumentSummarySet']['DocumentSummary']]
    return symbols

