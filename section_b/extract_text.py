import os
import re
import json
import io
import sys
import pypdf
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# EasyOCR singleton reader - kept None until lazily requested by Path 2 (Upload Image/Scan)
_easyocr_reader = None

def get_easyocr_reader():
    """
    Lazy singleton initializer for EasyOCR engine.
    Imports and loads PyTorch/EasyOCR models into memory ONLY when image/scan OCR is executed.
    This guarantees Path 1 (Paste OCR Text) never imports or touches EasyOCR at import or runtime.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader

def extract_text_from_file(contents: bytes, filename: str) -> str:
    """
    Extracts raw text from file contents (TXT, PDF, PNG, JPG, JPEG).
    Used by Path 2 (Upload Image/Scan) or file uploads.
    """
    filename_lower = filename.lower()
    
    # 1. Text File (.txt)
    if filename_lower.endswith('.txt'):
        return contents.decode('utf-8-sig', errors='ignore')
        
    # 2. PDF Document (.pdf)
    elif filename_lower.endswith('.pdf'):
        pdf_file = io.BytesIO(contents)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
        
    # 3. Image Document (.png, .jpg, .jpeg)
    elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
        ocr_reader = get_easyocr_reader()
        results = ocr_reader.readtext(contents, detail=0)
        return "\n".join(results)
    else:
        raise ValueError("Unsupported file format. Supported: PDF, TXT, PNG, JPG, JPEG")


# Comprehensive Field Alias Mapping
ALIAS_MAP = {
    # Full Name aliases
    "name": "full_name",
    "full name": "full_name",
    "customer": "full_name",
    "customer name": "full_name",
    "fullname": "full_name",
    "customername": "full_name",
    
    # First Name aliases
    "first name": "first_name",
    "fname": "first_name",
    "first_name": "first_name",
    "firstname": "first_name",
    
    # Middle Name aliases
    "middle name": "middle_name",
    "mname": "middle_name",
    "middle_name": "middle_name",
    "middlename": "middle_name",
    
    # Last Name aliases
    "last name": "last_name",
    "lname": "last_name",
    "surname": "last_name",
    "last_name": "last_name",
    "lastname": "last_name",
    
    # Email aliases
    "email": "email",
    "e mail": "email",
    "email address": "email",
    "e-mail": "email",
    "emailaddress": "email",
    "e-mail address": "email",
    
    # Phone aliases
    "phone": "phone_number",
    "phone number": "phone_number",
    "phone no": "phone_number",
    "mobile": "phone_number",
    "mobile no": "phone_number",
    "mobile number": "phone_number",
    "telephone": "phone_number",
    "contact": "phone_number",
    "contact number": "phone_number",
    "phone_number": "phone_number",
    "phonenumber": "phone_number",
    "phoneno": "phone_number",
    "mobileno": "phone_number",
    "mobilenumber": "phone_number",
    "contactnumber": "phone_number",
    
    # DOB aliases
    "dob": "date_of_birth",
    "date of birth": "date_of_birth",
    "birth date": "date_of_birth",
    "birth": "date_of_birth",
    "date_of_birth": "date_of_birth",
    "dateofbirth": "date_of_birth",
    "birthdate": "date_of_birth",
    
    # Address aliases
    "address": "address",
    "residence": "address",
    
    # Marital Status aliases
    "marital status": "marital_status",
    "marital": "marital_status",
    "status": "marital_status",
    "marital_status": "marital_status",
    "maritalstatus": "marital_status"
}


def normalize_key(raw_key: str) -> str | None:
    k_lower = raw_key.strip().lower()
    k_space = re.sub(r'[\s_-]+', ' ', k_lower).strip()
    k_nospace = re.sub(r'[\s_-]+', '', k_lower).strip()
    
    return ALIAS_MAP.get(k_space) or ALIAS_MAP.get(k_nospace) or ALIAS_MAP.get(k_lower)


def parse_key_value_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
        
    # Ignore explicit record markers and customer information headers
    if re.match(r'^\s*-{2,}\s*RECORD', stripped, re.IGNORECASE):
        return None
    if re.match(r'^\s*CUSTOMER\s+INFORMATION\s*$', stripped, re.IGNORECASE):
        return None

    # 1. Try ':' or '=' separator first
    for sep in (':', '='):
        if sep in stripped:
            parts = stripped.split(sep, 1)
            raw_key = parts[0].strip()
            val = parts[1].strip()
            if normalize_key(raw_key) or re.match(r'^[A-Za-z\s_-]+$', raw_key):
                return raw_key, val

    # 2. Try '-' separator (e.g. NAME - Arjun Thomas, Customer Name - Aditya Raj)
    if '-' in stripped:
        parts = stripped.split('-', 1)
        raw_key = parts[0].strip()
        val = parts[1].strip()
        if normalize_key(raw_key) or raw_key.lower() in ("name", "customer", "full name", "customer name"):
            return raw_key, val

    # 3. Try space separator if line starts with a recognized key alias (e.g. "E-MAIL arjun.thomas@example.com", "CONTACT +91-81234-56789")
    parts = stripped.split(maxsplit=1)
    if len(parts) == 2:
        raw_key = parts[0].strip()
        val = parts[1].strip()
        if normalize_key(raw_key):
            return raw_key, val

    return None


def segment_records(content: str) -> list[str]:
    """
    Robust record segmentation logic.
    Primary Strategy: Use explicit record markers matching r'(?mi)^\s*-{2,}\s*RECORD\s+\d+(?:\s*\([^)]*\))?\s*-{2,}\s*$'.
    When explicit markers exist, split ONLY on those markers and preserve all lines between them.

    Secondary Strategy: If 'CUSTOMER INFORMATION' headers exist, split on header lines.

    Fallback Strategy: If no explicit markers or headers exist, split on blank-line boundaries (\n\s*\n),
    which is standard for plain labeled-block OCR files.
    """
    content = content.strip()
    if not content:
        return []

    # 1. Explicit record markers check (e.g. --- RECORD 1 --- or --- RECORD 21 (DUPLICATE TEST) ---)
    marker_pattern = r'(?mi)^\s*-{2,}\s*RECORD\s+\d+(?:\s*\([^)]*\))?\s*-{2,}\s*$'
    markers = list(re.finditer(marker_pattern, content))
    
    if len(markers) >= 1:
        blocks = []
        for i in range(len(markers)):
            start_idx = markers[i].end()
            end_idx = markers[i+1].start() if i + 1 < len(markers) else len(content)
            block_text = content[start_idx:end_idx].strip()
            if block_text:
                blocks.append(block_text)
        if blocks:
            return blocks

    # 2. Customer Information headers check
    header_pattern = re.compile(r'^\s*(?:-{2,}|\={2,})\s*RECORD|\bCUSTOMER\s+INFORMATION\b', re.IGNORECASE)
    if header_pattern.search(content):
        lines = content.splitlines()
        blocks = []
        current_block = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            if header_pattern.search(stripped) and current_block:
                blocks.append("\n".join(current_block))
                current_block = [stripped]
            else:
                current_block.append(stripped)

        if current_block:
            blocks.append("\n".join(current_block))

        return [b for b in blocks if b.strip()]

    # 3. Fallback for plain labeled-block OCR text without markers/headers: split on blank lines (\n\s*\n)
    blocks = re.split(r'\n\s*\n', content)
    return [b.strip() for b in blocks if b.strip()]


def parse_record_block(block: str) -> dict:
    """
    Parses key-value pairs from a record block text.
    Handles multi-line field continuation (EasyOCR frequently wraps field values across multiple lines on skewed/scanned images).
    """
    raw_data = {}
    field_fragments = {}  # norm_field -> list of line strings
    open_field = None
    last_set_field = None

    lines = block.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        kv = parse_key_value_line(stripped)
        norm_field = None
        if kv:
            raw_key, val = kv
            norm_field = normalize_key(raw_key)

        if norm_field:
            # Line matched a recognized field label
            open_field = None
            clean_val = val.strip()
            
            # If initial value is empty or just stray separators (e.g. "Email:" -> val is "")
            if not clean_val or not clean_val.strip(' :=-'):
                open_field = norm_field
                if norm_field not in field_fragments:
                    field_fragments[norm_field] = []
                last_set_field = norm_field
            else:
                if norm_field not in field_fragments:
                    field_fragments[norm_field] = [clean_val]
                else:
                    if len(field_fragments[norm_field]) < 3:
                        field_fragments[norm_field].append(clean_val)
                last_set_field = norm_field
        else:
            # Line does NOT match any recognized label
            if open_field:
                # 1. An open field exists (e.g. "Email:" followed by "tanvi:", "desai98@gmail", "com")
                if len(field_fragments[open_field]) < 3:
                    field_fragments[open_field].append(stripped)
            elif last_set_field == "address":
                # 2. No open field, but PREVIOUS line set "address" (addresses frequently wrap onto unlabeled lines)
                if "address" in field_fragments and len(field_fragments["address"]) < 3:
                    field_fragments["address"].append(stripped)
            else:
                # 3. Otherwise ignore stray document title/footer noise line
                pass

    # Finalize and join fragments for each field
    for field, fragments in field_fragments.items():
        if not fragments:
            raw_data[field] = ""
            continue

        if field == "address":
            # Address: join fragments with ", " (keep human-readable)
            joined_lines = []
            for f in fragments:
                cleaned_f = f.strip().strip(',')
                if cleaned_f:
                    joined_lines.append(cleaned_f)
            raw_data[field] = ", ".join(joined_lines)
        elif field in ("email", "phone_number"):
            # Email / phone_number: join fragments with NO separator, then strip stray whitespace & colons
            joined = "".join([f.strip() for f in fragments if f.strip()])
            joined = re.sub(r'[\s:]+', '', joined)
            raw_data[field] = joined
        else:
            # Any other field: join fragments with a single space
            raw_data[field] = " ".join([f.strip() for f in fragments if f.strip()])

    return raw_data


def process_name_fields(raw_data: dict) -> tuple[str, str, str]:
    """
    Extracts (first_name, middle_name, last_name).
    Prioritizes explicit first/middle/last fields if provided.
    Merges partial explicit fields (e.g. Middle Name: Ananya in Record 20) with full_name tokens.
    """
    exp_first = raw_data.get('first_name', '').strip()
    exp_middle = raw_data.get('middle_name', '').strip()
    exp_last = raw_data.get('last_name', '').strip()
    full_name = raw_data.get('full_name', '').strip()

    if exp_first and exp_last:
        return exp_first, exp_middle, exp_last

    if full_name:
        cleaned_name = re.sub(r'\s+', ' ', full_name)
        tokens = cleaned_name.split(' ')
        
        first_name = exp_first or (tokens[0] if tokens else "")
        last_name = exp_last or (tokens[-1] if len(tokens) > 1 else "")
        
        if exp_middle:
            middle_name = exp_middle
        elif len(tokens) == 3:
            middle_name = tokens[1]
        elif len(tokens) > 3:
            middle_name = " ".join(tokens[1:-1])
        else:
            middle_name = ""
            
        return first_name, middle_name, last_name

    return exp_first, exp_middle, exp_last


def normalize_record(raw_data: dict) -> dict:
    """
    Builds standard 8-field dict with lowercase snake_case keys.
    Missing fields default to empty strings ("").
    """
    first_name, middle_name, last_name = process_name_fields(raw_data)
    
    return {
        "first_name": first_name or "",
        "last_name": last_name or "",
        "middle_name": middle_name or "",
        "email": raw_data.get("email", "").strip(),
        "phone_number": raw_data.get("phone_number", "").strip(),
        "date_of_birth": raw_data.get("date_of_birth", "").strip(),
        "address": raw_data.get("address", "").strip(),
        "marital_status": raw_data.get("marital_status", "").strip()
    }


def is_empty_record(rec: dict) -> bool:
    """
    True only when all 8 fields are empty/falsy - used to drop
    noise blocks (titles, footers) that produced no real fields.
    """
    return not any(rec.values())


def normalize_phone(phone: str) -> str:
    return re.sub(r'\D', '', phone)


def normalize_str(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def merge_records(primary: dict, secondary: dict) -> dict:
    """
    Merges non-empty fields from secondary into primary.
    """
    merged = dict(primary)
    for key, val in secondary.items():
        if val and not merged.get(key):
            merged[key] = val
        elif val and len(str(val)) > len(str(merged.get(key, ""))):
            merged[key] = val
    return merged


def records_match(r1: dict, r2: dict) -> bool:
    """
    Checks if r1 and r2 represent the same customer using multiple normalized field rules.
    """
    e1, e2 = r1["email"].lower().strip(), r2["email"].lower().strip()
    if e1 and e2 and e1 == e2:
        return True

    p1, p2 = normalize_phone(r1["phone_number"]), normalize_phone(r2["phone_number"])
    if p1 and p2 and p1 == p2:
        return True

    n1 = normalize_str(f'{r1["first_name"]} {r1["last_name"]}')
    n2 = normalize_str(f'{r2["first_name"]} {r2["last_name"]}')
    dob1, dob2 = normalize_str(r1["date_of_birth"]), normalize_str(r2["date_of_birth"])
    
    if n1 and n2 and n1 == n2:
        if dob1 and dob2 and dob1 == dob2:
            return True
        if p1 and p2 and p1 == p2:
            return True

    return False


def parse_ocr_data(content_or_filepath: str) -> dict:
    """
    Main extraction pipeline returning dict with total_records, unique_profiles count, and parsed_profiles.
    Shared by both Path 1 (Fast Text) and Path 2 (OCR Image/Scan).
    Filters out empty noise records (titles, footers) from total_records and parsed_profiles.
    """
    if os.path.exists(content_or_filepath) and os.path.isfile(content_or_filepath):
        with open(content_or_filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    else:
        content = content_or_filepath

    blocks = segment_records(content)
    
    unique_profiles = []
    valid_record_count = 0
    
    for block in blocks:
        raw_data = parse_record_block(block)
        rec = normalize_record(raw_data)
        
        # Skip noise blocks (titles, footers) that produced no real fields
        if is_empty_record(rec):
            continue
            
        valid_record_count += 1
        
        matched_idx = -1
        for idx, existing in enumerate(unique_profiles):
            if records_match(rec, existing):
                matched_idx = idx
                break
                
        if matched_idx >= 0:
            unique_profiles[matched_idx] = merge_records(unique_profiles[matched_idx], rec)
        else:
            unique_profiles.append(rec)

    return {
        "total_records": valid_record_count,
        "unique_profiles": len(unique_profiles),
        "parsed_profiles": unique_profiles
    }


def parse_ocr_text(file_path):
    """
    Backward-compatible function returning list of extracted customer profiles.
    """
    result = parse_ocr_data(file_path)
    return result["parsed_profiles"]


if __name__ == '__main__':
    import json
    # Run a simple extraction test on the default ocr_data.txt file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ocr_file_path = os.path.join(base_dir, 'data', 'ocr_data.txt')
    if os.path.exists(ocr_file_path):
        print(f"Running standalone extraction on: {ocr_file_path}")
        results = parse_ocr_data(ocr_file_path)
        print(f"Total valid candidate records segmented: {results['total_records']}")
        print(f"Unique profiles identified: {results['unique_profiles']}")
        print("\nFirst unique profile preview:")
        if results['parsed_profiles']:
            print(json.dumps(results['parsed_profiles'][0], indent=2))
    else:
        print(f"Default OCR text data file not found at: {ocr_file_path}")
