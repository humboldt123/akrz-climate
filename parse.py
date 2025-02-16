import mailbox
import csv
import json
import re
from openai import OpenAI
from typing import Dict, List, Tuple
import argparse

def get_api_key() -> str:
   """Read OpenAI API key from secret file."""
   try:
       with open('secret', 'r') as f:
           return f.read().strip()
   except FileNotFoundError:
       raise Exception("'secret' file not found in current directory :(")
   except Exception as e:
       raise Exception(f"error reading api key: {e}")

def clean_text(text: str) -> str:
   """Remove email headers, signatures, formatting artifacts, etc. etc."""
   lines = []
   # TODO: add to this list of skip patterns
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

def parse_block(client: OpenAI, text: str) -> Tuple[List[Dict], List[str]]:
   """Parse text via OpenAI."""
   prompt = f"""Parse this environmental report text into structured data. Extract:
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
}}"""

   response = client.chat.completions.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": "You are a parser specializing in environmental reports. Return only valid JSON."},
           {"role": "user", "content": prompt}
       ],
       temperature=0
   )
   
   try:
       result = json.loads(response.choices[0].message.content)
       return result["records"], result["unclassified_text"]
   except:
       return [], [text]

def process_mbox(mbox_path: str, output_csv: str, unclassified_txt: str, email_limit: int = float('inf')):
   """Process .mbox file, writing results to CSV and excess text file."""
   api_key = get_api_key()
   client = OpenAI(api_key=api_key)
   records = []
   unclassified = []
   
   mbox = mailbox.mbox(mbox_path)
   for i, message in enumerate(mbox):
       if i >= email_limit:
           break
           
       # email body
       if message.is_multipart():
           for part in message.walk():
               if part.get_content_type() == 'text/plain':
                   text = part.get_payload(decode=True).decode()
                   break
       else:
           text = message.get_payload(decode=True).decode()
       
       # clean and parse
       clean = clean_text(text)
       parsed_records, unclassified_text = parse_block(client, clean)
       
       records.extend(parsed_records)
       unclassified.extend(unclassified_text)
       
       print(f"Processed email {i+1}/{min(email_limit, len(mbox))}")
   
   with open(output_csv, 'w', newline='', encoding='utf-8') as f:
       writer = csv.DictWriter(f, fieldnames=['name', 'address', 'type', 'date', 'description'])
       writer.writeheader()
       writer.writerows(records) # ok now we write to csv
   
   with open(unclassified_txt, 'w', encoding='utf-8') as f:
       for text in unclassified:
           f.write(text + '\n\n' + '-'*80 + '\n\n') # junk text lol

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Process environmental report emails.')
   parser.add_argument('mbox_file', help='Input mbox file')
   parser.add_argument('output_csv', help='Output CSV file')
   parser.add_argument('unclassified_txt', help='File for unclassified text')
   parser.add_argument('--limit', type=int, default=float('inf'), 
                     help='Maximum number of emails to process (default: no limit)')
   
   args = parser.parse_args()
   
   process_mbox(args.mbox_file, args.output_csv, args.unclassified_txt, args.limit)