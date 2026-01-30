"""
RunPod Client for DeepSeek-OCR-2 Endpoint

Usage:
    # With PDF file
    python client.py --pdf path/to/label.pdf

    # With PDF URL
    python client.py --url https://example.com/label.pdf

    # Async mode for large PDFs
    python client.py --pdf path/to/label.pdf --async
"""

import argparse
import base64
import json
import os
import time
import requests


def get_config():
    """Get configuration from environment or prompt user"""
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

    if not api_key:
        api_key = input("Enter your RunPod API Key: ").strip()

    if not endpoint_id:
        endpoint_id = input("Enter your Endpoint ID: ").strip()

    return api_key, endpoint_id


def send_sync_request(api_key, endpoint_id, payload, timeout=600):
    """Send synchronous request and wait for result"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=timeout
    )

    response.raise_for_status()
    return response.json()


def send_async_request(api_key, endpoint_id, payload):
    """Send async request and return job ID"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    response.raise_for_status()
    return response.json()["id"]


def poll_for_result(api_key, endpoint_id, job_id, interval=5, max_wait=1800):
    """Poll for async job result"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    start_time = time.time()

    while True:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        status = response.json()

        print(f"Status: {status['status']}", end="\r")

        if status["status"] == "COMPLETED":
            return status["output"]
        elif status["status"] == "FAILED":
            raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
        elif status["status"] == "CANCELLED":
            raise Exception("Job was cancelled")

        if time.time() - start_time > max_wait:
            raise TimeoutError(f"Job did not complete within {max_wait} seconds")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="RunPod DeepSeek-OCR-2 Client")
    parser.add_argument("--pdf", help="Path to local PDF file")
    parser.add_argument("--url", help="URL to PDF file")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                       help="Use async mode (poll for results)")
    parser.add_argument("--full-text", action="store_true",
                       help="Return full OCR text in addition to indications")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds (default: 600)")

    args = parser.parse_args()

    if not args.pdf and not args.url:
        parser.error("Must provide either --pdf or --url")

    api_key, endpoint_id = get_config()

    # Prepare payload
    payload = {
        "input": {
            "return_full_text": args.full_text
        }
    }

    if args.pdf:
        print(f"Reading PDF: {args.pdf}")
        with open(args.pdf, "rb") as f:
            pdf_bytes = f.read()
        payload["input"]["pdf_base64"] = base64.b64encode(pdf_bytes).decode()
        print(f"PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
    else:
        payload["input"]["pdf_url"] = args.url
        print(f"Using PDF URL: {args.url}")

    print("\nSending request to RunPod...")

    try:
        if args.async_mode:
            job_id = send_async_request(api_key, endpoint_id, payload)
            print(f"Job submitted: {job_id}")
            print("Polling for results...")
            result = poll_for_result(api_key, endpoint_id, job_id)
        else:
            response = send_sync_request(api_key, endpoint_id, payload, args.timeout)
            if "error" in response:
                raise Exception(response["error"])
            result = response.get("output", response)

        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"Pages processed: {result.get('page_count', 'N/A')}")
        print(f"\nINDICATIONS AND USAGE:\n")
        indications = result.get("indications_and_usage", "")
        print(indications[:2000] if len(indications) > 2000 else indications)

        if len(indications) > 2000:
            print(f"\n... (Total: {len(indications)} characters)")

        # Save output
        if args.output:
            output_path = args.output
        elif args.pdf:
            output_path = args.pdf.replace(".pdf", "_indications.txt")
        else:
            output_path = "indications_output.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(indications)
        print(f"\nFull output saved to: {output_path}")

        if args.full_text and "full_text" in result:
            full_text_path = output_path.replace(".txt", "_full_ocr.txt")
            with open(full_text_path, "w", encoding="utf-8") as f:
                f.write(result["full_text"])
            print(f"Full OCR text saved to: {full_text_path}")

    except requests.exceptions.Timeout:
        print(f"Request timed out after {args.timeout} seconds.")
        print("Try using --async mode for large PDFs.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
