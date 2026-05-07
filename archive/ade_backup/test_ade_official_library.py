#!/usr/bin/env python3
"""
Test ADE using the official agentic-doc library
"""
import base64
from pathlib import Path

# Read key from secrets.toml
with open('.streamlit/secrets.toml', 'r') as f:
    for line in f:
        if 'LANDINGAI_API_KEY' in line:
            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
            break

print(f"🔑 Testing with key from secrets.toml")
print(f"   Length: {len(api_key)}")
print()

# Try with official library
try:
    from agentic_doc import ADEClient
    
    print("✅ agentic-doc library found!")
    print()
    
    # Initialize client
    print("🧪 Initializing ADE client...")
    client = ADEClient(api_key=api_key)
    print("   ✅ Client initialized")
    print()
    
    # Test with a PDF
    pdf_path = Path('subset_gold/selected_pages_no_annotations.pdf')
    if not pdf_path.exists():
        print("❌ PDF not found")
        exit(1)
    
    print(f"📄 Parsing PDF: {pdf_path}")
    print("   (This may take a moment...)")
    
    try:
        # Parse the document
        result = client.parse(document=str(pdf_path))
        
        print()
        print("✅ SUCCESS! PDF parsed!")
        print(f"   Type: {type(result)}")
        
        if hasattr(result, 'pages'):
            print(f"   📄 Pages: {len(result.pages)}")
            if result.pages:
                page1 = result.pages[0]
                print(f"   📋 Page 1 elements: {len(page1.elements) if hasattr(page1, 'elements') else 'N/A'}")
        elif isinstance(result, dict):
            pages = result.get('pages', [])
            print(f"   📄 Pages: {len(pages)}")
            if pages:
                print(f"   📋 Page 1 keys: {list(pages[0].keys()) if pages else 'N/A'}")
        
        # Save result
        import json
        if hasattr(result, '__dict__'):
            result_dict = result.__dict__
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"result": str(result)}
        
        with open('test_ade_result_official.json', 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        print("   💾 Saved to test_ade_result_official.json")
        
    except Exception as e:
        print(f"   ❌ Error parsing: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("❌ agentic-doc library not installed")
    print("   Install with: pip install agentic-doc")
    print()
    print("💡 The key might work with the official library even if direct API calls fail")
    print("   This is because the library handles authentication differently")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()



