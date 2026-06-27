import httpx
import sys

def main():
    print("Testing GET /api/v1/papers/ with no API key...")
    try:
        response = httpx.get("http://localhost:8000/api/v1/papers/", timeout=10.0)
        print(f"Status: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Body: {response.text}")
    except httpx.TimeoutException:
        print("Request timed out!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
