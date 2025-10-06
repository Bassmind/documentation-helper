import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # For Vector Storing Locally
from langchain_openai import OpemAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

#TavilyCrawl to get TavilyDocumentation, TavilyExtract to extract the content, TavilyMap to map the content
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

#Configure SSL context to use certifi certificates
#We will require these to use tons of requests to Tavily API
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

embeddings = OpemAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

#Chroma if you want to store the vectors locally
#chroma = Chroma(embedding_function=embeddings, persist_directory="chroma_db")

#Pinecone if you want to store the vectors in Pinecone
vector_store = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)

#Tavily Variables
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=5, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process"""
    print("Starting the ingestion process...")




if __name__ == "__main__":
    asyncio.run(main())