#!/usr/bin/env python3
"""
Test ADE API with a real PDF file to see if the key works
"""
import base64
import requests
import toml
import json
from pathlib import Path

# Load key from secrets.toml
secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets.get('LANDINGAI_API_KEY')

print(f"🔑 Testing with key from secrets.toml")
print(f"   Length: {len(api_key)}")
print(f"   Preview: {api_key[:15]}...{api_key[-5:]}")
print()

# Find a test PDF
pdf_path = None
for path in [
    Path('subset_gold/selected_pages_no_annotations.pdf'),
    Path('test.pdf'),
]:
    if path.exists():
        pdf_path = path
        break

if not pdf_path:
    print("❌ No PDF file found for testing")
    exit(1)

print(f"📄 Using PDF: {pdf_path}")
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
print(f"   PDF size: {len(pdf_bytes)} bytes")
print(f"   Base64 size: {len(pdf_base64)} chars")
print()

# Test with real PDF - try all authentication methods
methods = [
    ("Bearer token (original)", {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }),
]

# Try decoded key if it's base64
try:
    decoded_key = base64.b64decode(api_key).decode('utf-8')
    if ':' in decoded_key:
        username, password = decoded_key.split(':', 1)
        methods.append(("Basic Auth (decoded)", None, username, password))
    methods.append(("Bearer token (decoded)", {
        "Authorization": f"Bearer {decoded_key}",
        "Content-Type": "application/json"
    }))
except:
    pass

# Test each method
for method_name, headers, *auth_args in methods:
    print(f"🧪 Testing: {method_name}")
    
    payload = {
        "file": pdf_base64,
        "output_format": "json"
    }
    
    try:
        if auth_args and len(auth_args) == 2:
            # Basic Auth
            from requests.auth import HTTPBasicAuth
            response = requests.post(
                "https://api.landing.ai/v1/parse",
                json=payload,
                auth=HTTPBasicAuth(auth_args[0], auth_args[1]),
                headers={"Content-Type": "application/json"},
                timeout=120
            )
        else:
            # Bearer token
            response = requests.post(
                "https://api.landing.ai/v1/parse",
                json=payload,
                headers=headers,
                timeout=120
            )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ SUCCESS! API call worked!")
            result = response.json()
            
            # Check what we got back
            pages = result.get('pages', [])
            print(f"   📄 Pages returned: {len(pages)}")
            
            if pages:
                page1 = pages[0]
                print(f"   📋 Page 1 structure:")
                print(f"      - Elements: {len(page1.get('elements', []))}")
                print(f"      - Text chunks: {len(page1.get('text_chunks', []))}")
                print(f"      - Tables: {len(page1.get('tables', []))}")
                
                # Show first few elements
                elements = page1.get('elements', [])[:3]
                for i, elem in enumerate(elements):
                    print(f"      Element {i+1}: type={elem.get('type')}, text={elem.get('text', '')[:50]}...")
            
            print()
            print("   ✅ ADE API is working! The key is valid!")
            print(f"   💾 Saving result to test_ade_result.json...")
            with open('test_ade_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("   ✅ Result saved!")
            break
            
        elif response.status_code == 401:
            print(f"   ❌ 401 Unauthorized")
            error_data = response.json() if response.text else {}
            print(f"   Error: {error_data.get('message', response.text[:200])}")
        elif response.status_code == 402:
            print(f"   ❌ 402 Payment Required")
            print(f"   Response: {response.text[:200]}")
        elif response.status_code == 400:
            print(f"   ❌ 400 Bad Request")
            print(f"   Response: {response.text[:200]}")
        else:
            print(f"   ❌ Status {response.status_code}")
            print(f"   Response: {response.text[:300]}")
            
    except requests.exceptions.Timeout:
        print("   ⏱️ Request timed out (PDF might be too large)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()

print("=" * 60)
print("SUMMARY:")
print("=" * 60)
if Path('test_ade_result.json').exists():
    print("✅ ADE API is working! Check test_ade_result.json for full results.")
else:
    print("❌ All authentication methods failed.")
    print("   The key may be invalid, expired, or have insufficient permissions.")























