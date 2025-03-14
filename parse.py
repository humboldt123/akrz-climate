#!/usr/bin/env python3
import mailbox
import csv
import json
import re
import time
import requests
import os
import argparse
from typing import Dict, List, Tuple, Optional

# set environment variables for hf cache
os.environ["HF_HOME"] = "/local-ssd/hf_cache/"

def geocode_address(address: str) -> Optional[str]:
    """Convert address to latitude,longitude using Nominatim API."""
    if not address:
        return None
        
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "EnvReportParser/1.0"  # required by nominatim ToS
    }
    
    try:
        # rate limit but do we care? I could totally get rid of this cos
        # the llm is gonna take a while too
        time.sleep(1)
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        if results and len(results) > 0:
            lat = results[0]["lat"]
            lon = results[0]["lon"]
            return f"({lat}, {lon})"
        return None
    except Exception as e:
        print(f"Geocoding error for address '{address}': {e}")
        return None

def clean_text(text: str) -> str:
    """Remove email headers, signatures, formatting artifacts, etc. etc."""
    lines = []
    skip_patterns = [
        r'^-+ Forwarded message -+$',
        r'^From:',
        r'^Date:',
        r'^Subject:',
        r'^To:',
        r'^Contact:',
        r'^Thanks',
        r'@',
        r'advocate',
        r'cleanair\.org',
        r'Southeast Region:',
        r'^\s*>+'
    ]
    
    for line in text.split('\n'):
        if not any(re.search(pattern, line, re.I) for pattern in skip_patterns):
            lines.append(line)
            
    return '\n'.join(lines)

def extract_date(text: str) -> str:
    """Extract date from text using regex."""
    date_patterns = [
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'\d{1,2}/\d{1,2}/\d{4}'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""

def init_llama_model():
    """Initialize Llama 3.1 70B model using HuggingFace Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
    
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("Creating text generation pipeline...")
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        do_sample=False
    )
    
    return llm

def parse_block(llm, text: str) -> Tuple[List[Dict], List[str]]:
    """Parse text via Llama model."""
    system_prompt = "You are a parser specializing in environmental reports. Return only valid JSON."
    
    user_prompt = f"""Parse this environmental report text into structured data. Extract:
1. Facility Name
2. Address (must include street number, street name, city, state, zip)
3. Type of contamination/environmental issue
4. Date (if present)
5. Full description of the issue

If multiple facilities or issues are mentioned, return data for each separately.
Ignore contact information, email headers, and administrative details.

Text to parse:
{text}

Return JSON in this format:
{{
    "records": [
        {{
            "name": "facility name",
            "address": "full address",
            "type": "contamination type",
            "date": "date if present",
            "description": "full description"
        }}
    ],
    "unclassified_text": ["any text that couldn't be parsed into records"]
}}

Ensure you only return valid JSON."""

    # Format prompt according to Llama 3's expected format
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"
    
    try:
        print("Generating response...")
        response = llm(prompt)[0]['generated_text']
        
        # Strip the prompt from the response to just get the assistant's reply
        response = response.split("<|assistant|>")[-1].strip()
        
        # Extract JSON from response if it's embedded in text
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
        
        print("Parsing JSON response...")
        result = json.loads(response)
        return result["records"], result["unclassified_text"]
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response}")
        return [], [text]

def process_mbox(mbox_path: str, output_csv: str, unclassified_txt: str, email_limit: int = float('inf')):
    """Process .mbox file, writing results to CSV and excess text file."""
    print("Initializing Llama model...")
    llm = init_llama_model()
    records = []
    unclassified = []
    
    print(f"Opening mbox file: {mbox_path}")
    mbox = mailbox.mbox(mbox_path)
    for i, message in enumerate(mbox):
        if i >= email_limit:
            break
            
        print(f"Processing email {i+1}/{min(email_limit, len(mbox))}")
        
        # email body
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == 'text/plain':
                    text = part.get_payload(decode=True).decode(errors='replace')
                    break
            else:
                text = ""  # no text/plain part found
        else:
            text = message.get_payload(decode=True).decode(errors='replace')
        
        if not text:
            print("  No text content found, skipping")
            continue
            
        print("  Cleaning text...")
        clean = clean_text(text)
        
        print("  Parsing with Llama...")
        parsed_records, unclassified_text = parse_block(llm, clean)
        
        print("  Geocoding addresses...")
        for record in parsed_records:
            record["latlong"] = geocode_address(record["address"])
        
        records.extend(parsed_records)
        unclassified.extend(unclassified_text)
        
        print(f"  Found {len(parsed_records)} records")
    
    print(f"Writing {len(records)} records to {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'address', 'type', 'date', 'description', 'latlong'])
        writer.writeheader()
        writer.writerows(records)
    
    print(f"Writing unclassified text to {unclassified_txt}")
    with open(unclassified_txt, 'w', encoding='utf-8') as f:
        for text in unclassified:
            f.write(text + '\n\n' + '-'*80 + '\n\n')
    
    print("Processing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process environmental report emails using Llama 3.1 70B.')
    parser.add_argument('mbox_file', help='Input mbox file')
    parser.add_argument('output_csv', help='Output CSV file')
    parser.add_argument('unclassified_txt', help='File for unclassified text')
    parser.add_argument('--limit', type=int, default=float('inf'), 
                      help='Maximum number of emails to process (default: no limit)')
    parser.add_argument('--gpus', default="0,1", 
                      help='GPU devices to use (default: "0,1")')
    
    args = parser.parse_args()
    
    # set visible GPUs before importing torch
    print(f"Setting CUDA_VISIBLE_DEVICES={args.gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    process_mbox(args.mbox_file, args.output_csv, args.unclassified_txt, args.limit)