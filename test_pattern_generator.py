import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_smpte_bars(width=1024, height=768):
    """
    Generates a mathematically perfect SMPTE-style Color Bar test pattern.
    This serves as the "Authentic" control with zero noise and perfect edges.
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    # Standard SMPTE Colors (Normalized RGB approx)
    # White, Yellow, Cyan, Green, Magenta, Red, Blue
    colors_top = [
        (192, 192, 192), # White (75%)
        (192, 192, 0),   # Yellow
        (0, 192, 192),   # Cyan
        (0, 192, 0),     # Green
        (192, 0, 192),   # Magenta
        (192, 0, 0),     # Red
        (0, 0, 192)      # Blue
    ]
    
    # Top 2/3rds: The 7 standard bars
    bar_width = width // 7
    for i, color in enumerate(colors_top):
        x0 = i * bar_width
        x1 = (i + 1) * bar_width if i < 6 else width
        y0 = 0
        y1 = int(height * 0.67)
        draw.rectangle([x0, y0, x1, y1], fill=color)
        
    # Middle Band (Castellated) - Reverse Blue bars
    # Blue, Black, Magenta, Black, Cyan, Black, White
    colors_mid = [
        (0, 0, 192), (19, 19, 19), (192, 0, 192), (19, 19, 19),
        (0, 192, 192), (19, 19, 19), (192, 192, 192)
    ]
    
    y_mid_start = int(height * 0.67)
    y_mid_end = int(height * 0.75)
    
    for i, color in enumerate(colors_mid):
        x0 = i * bar_width
        x1 = (i + 1) * bar_width if i < 6 else width
        draw.rectangle([x0, y_mid_start, x1, y_mid_end], fill=color)
        
    # Bottom Section (PLUGE signals)
    # 50% is special: It has the PLUGE pulse (super black, black, light black)
    y_bot_start = int(height * 0.75)
    y_bot_end = height
    
    # Big block of I-white (100% white)
    draw.rectangle([0, y_bot_start, int(bar_width * 1.25), y_bot_end], fill=(0, 33, 76)) # I-signal approx
    
    # White 100%
    draw.rectangle([int(bar_width * 1.25), y_bot_start, int(bar_width * 2.5), y_bot_end], fill=(255, 255, 255))
    
    # Q-signal (Purple)
    draw.rectangle([int(bar_width * 2.5), y_bot_start, int(bar_width * 3.75), y_bot_end], fill=(50, 0, 106)) 
    
    # Black
    draw.rectangle([int(bar_width * 3.75), y_bot_start, width, y_bot_end], fill=(19, 19, 19))

    # Add PLUGE stripes in the black block (Sub-black, Super-black)
    pluge_center = int(bar_width * 5.5)
    pluge_width = bar_width // 3
    
    draw.rectangle([pluge_center - pluge_width, y_bot_start + 20, pluge_center - pluge_width//2, y_bot_end - 20], fill=(0, 0, 0)) # Super Black
    draw.rectangle([pluge_center + pluge_width//2, y_bot_start + 20, pluge_center + pluge_width, y_bot_end - 20], fill=(28, 28, 28)) # Light Black

    return img

if __name__ == "__main__":
    img = generate_smpte_bars()
    img.save("smpte_real.jpg")
    print("Generated 'smpte_real.png'. Now download the AI image as 'smpte_ai.png' and compare!")