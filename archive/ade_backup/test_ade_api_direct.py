#!/usr/bin/env python3
"""
Test ADE API directly with the key from secrets.toml
"""
import base64
import requests
import toml
import json
from pathlib import Path

# Load key from secrets.toml
secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets.get('LANDINGAI_API_KEY')

print(f"🔑 Key from secrets.toml:")
print(f"   Length: {len(api_key)}")
print(f"   Preview: {api_key[:20]}...{api_key[-10:]}")
print()

# Test 1: Try as Bearer token (standard)
print("🧪 Test 1: Bearer token authentication")
headers1 = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Create a minimal test PDF
test_pdf_bytes = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/MediaBox [0 0 612 792]\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 <<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\n>>\n>>\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test) Tj\nET\nendstream\nendobj\nxref\n0 5\ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n385\n%%EOF'

pdf_base64 = base64.b64encode(test_pdf_bytes).decode('utf-8')
payload = {
    "file": pdf_base64,
    "output_format": "json"
}

try:
    response = requests.post(
        "https://api.landing.ai/v1/parse",
        json=payload,
        headers=headers1,
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ SUCCESS! Bearer token works!")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
    elif response.status_code == 401:
        print("   ❌ 401 Unauthorized")
        print(f"   Response: {response.text[:200]}")
    else:
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 2: Try decoding base64 and using as username:password (Basic Auth)
print("🧪 Test 2: Basic Auth (decoded username:password)")
try:
    decoded_key = base64.b64decode(api_key).decode('utf-8')
    print(f"   Decoded: {decoded_key[:30]}...")
    
    if ':' in decoded_key:
        username, password = decoded_key.split(':', 1)
        print(f"   Username: {username}")
        print(f"   Password: {password[:20]}...")
        
        from requests.auth import HTTPBasicAuth
        try:
            response = requests.post(
                "https://api.landing.ai/v1/parse",
                json=payload,
                auth=HTTPBasicAuth(username, password),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ SUCCESS! Basic Auth works!")
                result = response.json()
                print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
            elif response.status_code == 401:
                print("   ❌ 401 Unauthorized")
                print(f"   Response: {response.text[:200]}")
            else:
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ⚠️ Decoded key doesn't have ':' separator")
except Exception as e:
    print(f"   ❌ Error decoding: {e}")

print()

# Test 3: Try decoded key as Bearer token
print("🧪 Test 3: Decoded key as Bearer token")
try:
    decoded_key = base64.b64decode(api_key).decode('utf-8')
    headers3 = {
        "Authorization": f"Bearer {decoded_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(
            "https://api.landing.ai/v1/parse",
            json=payload,
            headers=headers3,
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ SUCCESS! Decoded Bearer token works!")
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
        elif response.status_code == 401:
            print("   ❌ 401 Unauthorized")
            print(f"   Response: {response.text[:200]}")
        else:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
except Exception as e:
    print(f"   ❌ Error decoding: {e}")

print()

# Test 4: Try X-API-Key header
print("🧪 Test 4: X-API-Key header")
headers4 = {
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}
try:
    response = requests.post(
        "https://api.landing.ai/v1/parse",
        json=payload,
        headers=headers4,
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ SUCCESS! X-API-Key header works!")
    elif response.status_code == 401:
        print("   ❌ 401 Unauthorized")
        print(f"   Response: {response.text[:200]}")
    else:
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("=" * 60)
print("SUMMARY:")
print("=" * 60)
print("If all tests failed with 401, the API key is invalid/expired.")
print("You need to get a new valid key from https://landing.ai/")
print("The key should start with 'land_sk_' and not be base64 encoded.")



