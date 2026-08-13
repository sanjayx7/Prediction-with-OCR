import os
from PIL import Image, ImageDraw, ImageFont

def generate_card():
    # Create base image with premium light layout
    width, height = 700, 400
    image = Image.new('RGB', (width, height), color='#ffffff')
    draw = ImageDraw.Draw(image)
    
    # Draw border and accents
    draw.rectangle([10, 10, width-10, height-10], outline='#cbd5e1', width=2)
    draw.rectangle([10, 10, 30, height-10], fill='#3b82f6') # Blue left stripe
    
    # Text contents
    title = "InsuredCRM - Customer Profile Document"
    lines = [
        "Name: Ramesh Kumar",
        "DOB: 17-04-1985",
        "Email: ramesh.kumar85@gmail.com",
        "Phone: +91-9876543210",
        "Address: 123, MG Road, Bengaluru, Karnataka, India",
        "Marital Status: Married",
        "ID Number: 4789652310"
    ]
    
    # Draw Title
    try:
        # Try loading a standard system font
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_body = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Fall back to default
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
        
    draw.text((60, 40), title, fill='#0f172a', font=font_title)
    draw.line([60, 70, width-50, 70], fill='#e2e8f0', width=1)
    
    # Draw Details
    y = 100
    for line in lines:
        draw.text((60, y), line, fill='#334155', font=font_body)
        y += 35
        
    # Draw footer
    draw.text((60, 350), "Document ID: DOC-2026-89652310 | OCR Standard Document", fill='#94a3b8', font=font_body)
    
    # Save image to data/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'sample_customer_card.png')
    image.save(output_path, 'PNG')
    print(f"Sample customer card image generated at: {output_path}")

if __name__ == '__main__':
    generate_card()
