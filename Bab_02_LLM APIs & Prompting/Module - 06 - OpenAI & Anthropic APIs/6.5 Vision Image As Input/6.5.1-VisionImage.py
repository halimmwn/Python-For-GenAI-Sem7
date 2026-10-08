import anthropic, os, base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ.get("XKIRO_API_KEY"),
    base_url="https://api.xkiro.com"
)

# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    response = client.messages.create(
        model="qwen/qwen3-vl-plus:free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": url}},
                {"type": "text", "text": "Describe what you see in this image."}
            ]
        }]
    )
    return response.content[0].text

# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    media_type = "image/jpeg"
    response = client.messages.create(
        model="qwen/qwen3-vl-plus:free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64}
                },
                {"type": "text", "text": "What is in this image?"}
            ]
        }]
    )
    return response.content[0].text

# Usage:
image_path = Path(__file__).with_name("alhamdulillah.jpg")
text = describe_image_file(str(image_path))
print(text)