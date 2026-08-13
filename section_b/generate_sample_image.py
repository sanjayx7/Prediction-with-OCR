import os
import re
import shutil
from PIL import Image, ImageDraw, ImageFont

def generate_all_cards():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ocr_file = os.path.join(base_dir, 'data', 'ocr_data.txt')
    output_dir = os.path.join(base_dir, 'data', 'sample_cards')
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the OCR text records
    if not os.path.exists(ocr_file):
        raise FileNotFoundError(f"OCR source data not found at: {ocr_file}")
        
    with open(ocr_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        
    # Split text records by empty lines
    blocks = re.split(r'\n\s*\n', content.strip())
    
    first_card_path = None
    
    for i, block in enumerate(blocks):
        if not block.strip():
            continue
            
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        
        # Get customer name to use as file name
        name_slug = f"customer_{i+1}"
        for line in lines:
            if line.lower().startswith('name:'):
                raw_name = line.split(':', 1)[1].strip()
                name_slug = raw_name.replace(' ', '_').lower()
                break
                
        # Card specifications
        width, height = 700, 400
        image = Image.new('RGB', (width, height), color='#ffffff')
        draw = ImageDraw.Draw(image)
        
        # Accent and border styling
        draw.rectangle([10, 10, width-10, height-10], outline='#cbd5e1', width=2)
        draw.rectangle([10, 10, 30, height-10], fill='#3b82f6') # Left stripe
        
        # Font setup
        title = "InsuredCRM - Customer Profile Document"
        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_body = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
            
        # Draw header
        draw.text((60, 40), title, fill='#0f172a', font=font_title)
        draw.line([60, 70, width-50, 70], fill='#e2e8f0', width=1)
        
        # Draw text details dynamically from the record block
        y = 100
        for line in lines:
            draw.text((60, y), line, fill='#334155', font=font_body)
            y += 35
            
        # Draw footer
        draw.text((60, 350), f"Document ID: DOC-2026-{10000000 + i} | OCR Standard Document", fill='#94a3b8', font=font_body)
        
        # Save each card
        card_path = os.path.join(output_dir, f"{name_slug}.png")
        image.save(card_path, 'PNG')
        print(f"Generated card for {name_slug} at: {card_path}")
        
        if i == 0:
            first_card_path = card_path
            
    # Copy the first card to data/sample_customer_card.png for UI download compliance
    if first_card_path and os.path.exists(first_card_path):
        shutil.copy(first_card_path, os.path.join(base_dir, 'data', 'sample_customer_card.png'))

if __name__ == '__main__':
    generate_all_cards()
