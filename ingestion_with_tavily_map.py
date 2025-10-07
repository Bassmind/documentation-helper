#Dependencies required: langchain-tavily, certifi, rich

import asyncio
import os
import ssl
import requests
import time
from typing import Any, Dict, List
import json
from datetime import datetime
from dotenv import load_dotenv

import certifi
from langchain_tavily import TavilyExtract, TavilyMap
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

load_dotenv()

#Configure SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

#Initialize Rich Console for pretty printing
console = Console()


"""
This file is to test Crawling using TavilyMap and TavilyExtract.
TavilyMap is used to map a given url and get a map o the URLs found on the pages
TavilyExtract is used to extract the content from the URLs found by TavilyMap
"""

async def main():
    print("All Imports successful!")

    tavily_map = TavilyMap(
        max_depth=3,    #Crawl up to 3 levels deep
        max_breadth=15, #Explore up to 15 links per page
        limit=500       #Limit to 500 pages total
    )

    # Example website to map
    demo_url = "https://python.langchain.com/docs/introduction/"

    console.print(f"Mapping website structure for: {demo_url}", style="bold blue")
    console.print("This may take a moment...")

    #Map the website structure
    site_map = tavily_map.invoke(demo_url)

    #Display results
    urls = site_map.get("results", [])
    console.print(f"\nSuccessfully mapped {len(urls)} URLs", style="bold green")

    #Show first 10 URLs as example
    console.print("\nFirst 10 URLs found:", style="bold yellow")
    for i, url in enumerate(urls[:10], 1):
        console.print(f"{i:2d}. {url}")
    
    if(len(urls) > 10):
        console.print(f"    ...   and {len(urls) - 10} more URLs")

    console.print("TavilyMap Initialized successfully!", style="bold green")

    #Now use TavilyExtract to extract content from the mapped URLs
    tavily_extract = TavilyExtract()

    print("✅ Tavily Extract Initialized successfully!")

    #Select a few URLs to extract content from
    sample_urls = [urls[20]]

    console.print(f"Extracting content from {len(sample_urls)} sample URLs...", style="bold blue")

    #Extract content
    extraction_result = await tavily_extract.ainvoke(input={"urls": sample_urls})

    #Display extraction results
    extracted_docs = extraction_result.get('results', [])
    console.print(f"\nSuccessfully extracted content from {len(extracted_docs)} documents", style="bold green")

    #Show summary of each extracted document
    for i, doc in enumerate(extracted_docs, 1):
        url = doc.get('url', 'Unknown URL')
        content = doc.get('raw_content', '')

        #Create a panel for each document
        panel_content = f"""URL: {url}
        Content Length: {len(content):,} characters
        Preview: {content}..."""

        console.print(Panel(panel_content, title=f"Document {i}", border_style="blue"))
        print() #Spacing

    def chunk_urls(urls: List[str], chunk_size: int = 3) -> List[List[str]]:
        """Utility function to chunk URLs into smaller lists"""
        chunks = []

        for i in range(0, len(urls), chunk_size):
            chunk = urls[i:i + chunk_size]
            chunks.append(chunk)

        return chunks

    async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
        """Extract documents from a batch of URLs."""
        try:
            console.print(f"Processing batch {batch_num} with {len(urls)} URLs...", style="blue")
            docs = await tavily_extract.ainvoke(input={"urls": urls})
            results = docs.get("results", [])
            console.print(f"Batch {batch_num} completed: Extracted {len(results)} documents.", style="green")
            return results
        except Exception as e:
            console.print(f"Error in batch {batch_num}: {e}", style="red")
            return []

    #Process a larger set of URLs in batches
    url_batches = chunk_urls(urls[:9], chunk_size=3)  # Limit to first 9 URLs for demo

    console.print(f"\nProcessing 9 URLs in {len(url_batches)} batches...", style="bold yellow")

    #Process batches concurrently
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    batch_results = await asyncio.gather(*tasks)

    #Flatten results
    all_extracted = []
    for batch_result in batch_results:
        all_extracted.extend(batch_result)

    console.print(f"\nBatch processing complete!: Total documents extracted: {len(all_extracted)}", style="bold green")

if __name__ == "__main__":
    asyncio.run(main())