import os
import requests
from pathlib import Path

UNSPLASH_ACCESS_KEY = os.environ["UNSPLASH_ACCESS_KEY"]
QUERY = "oil painting portrait"
NUM_IMAGES = 40
OUTPUT_DIR = Path("data/style")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_images():
    page = 1
    downloaded = 0

    while downloaded < NUM_IMAGES:
        response = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": QUERY,
                "per_page": 30,
                "page": page,
                "orientation": "squarish",
            },
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
        )
        results = response.json()["results"]

        for photo in results:
            if downloaded >= NUM_IMAGES:
                break

            img_url = photo["urls"]["regular"]
            img_id  = photo["id"]
            img_path = OUTPUT_DIR / f"{img_id}.jpg"
            caption_path = OUTPUT_DIR / f"{img_id}.txt"

            img_data = requests.get(img_url).content
            img_path.write_bytes(img_data)

            caption_path.write_text(
                f"an oil painting portrait, classical style, detailed brushwork, warm tones"
            )

            downloaded += 1
            print(f"Downloaded {downloaded}/{NUM_IMAGES}: {img_id}")

        page += 1

    print(f"Done. {downloaded} images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_images()