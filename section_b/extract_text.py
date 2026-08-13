import os
import re
import json

def parse_ocr_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"OCR text file not found at: {file_path}")
        
    # Using utf-8-sig to automatically handle Byte Order Mark (BOM) if present
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        
    # Split content by double newlines or blank lines (supporting both CRLF and LF)
    # This splits the file into blocks for each customer
    blocks = re.split(r'\n\s*\n', content.strip())
    
    parsed_people = []
    seen_emails = set()
    seen_people_keys = set()
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Parse block lines into key-value pairs
        data = {}
        for line in block.splitlines():
            if ':' in line:
                key, val = line.split(':', 1)
                data[key.strip().lower()] = val.strip()
                
        # Extract Name and split into components
        full_name = data.get('name', '')
        first_name = ""
        middle_name = ""
        last_name = ""
        
        if full_name:
            # Clean name
            cleaned_name = re.sub(r'\s+', ' ', full_name).strip()
            name_parts = cleaned_name.split(' ')
            if len(name_parts) == 1:
                first_name = name_parts[0]
            elif len(name_parts) == 2:
                first_name = name_parts[0]
                last_name = name_parts[1]
            elif len(name_parts) == 3:
                first_name = name_parts[0]
                middle_name = name_parts[1]
                last_name = name_parts[2]
            else:
                first_name = name_parts[0]
                middle_name = " ".join(name_parts[1:-1])
                last_name = name_parts[-1]
                
        # Extract fields
        email = data.get('email', '')
        phone = data.get('phone', '') or data.get('phone number', '')
        dob = data.get('dob', '') or data.get('date of birth', '')
        address = data.get('address', '')
        marital_status = data.get('marital status', '')
        
        # Normalize fields for matching and output
        email = email.lower().strip()
        
        # Deduplication check
        # Deduplicate based on non-empty email, or a combination of name and DOB
        person_key = (first_name.lower(), last_name.lower(), dob)
        
        if email and email in seen_emails:
            print(f"Skipping duplicate record based on Email: {email}")
            continue
        if not email and person_key in seen_people_keys:
            print(f"Skipping duplicate record based on Name & DOB: {first_name} {last_name}, {dob}")
            continue
            
        # Record keeping
        if email:
            seen_emails.add(email)
        seen_people_keys.add(person_key)
        
        # Build customer record
        # Note: missing fields are defaulted to "" or null (None) if not found.
        person_record = {
            "First Name": first_name or None,
            "Last Name": last_name or None,
            "Middle Name": middle_name or None,
            "Email": email or None,
            "Phone Number": phone or None,
            "Date of Birth": dob or None,
            "Address": address or None,
            "Marital Status": marital_status or None
        }
        parsed_people.append(person_record)
        
    return parsed_people

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ocr_file_path = os.path.join(base_dir, 'data', 'ocr_data.txt')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Section B: OCR Text Extraction Pipeline...")
    try:
        extracted_data = parse_ocr_text(ocr_file_path)
        
        output_file_path = os.path.join(output_dir, 'extracted_customers.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4)
            
        print(f"Extraction successful! Parsed {len(extracted_data)} customers.")
        print(f"JSON output saved to: output/extracted_customers.json")
        
        # Print first parsed item as confirmation
        if extracted_data:
            print("\nSample Extracted Record:")
            print(json.dumps(extracted_data[0], indent=2))
            
    except Exception as e:
        print(f"Error occurred during text extraction: {e}")

if __name__ == '__main__':
    main()
