import base64
from openai import OpenAI

client = OpenAI(
    api_key="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"
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to image
image_path = "Wolf8.png"

# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is there a wolf on the image? If yes, tell me the location of the wolf on the image. Share the x_min, y_min, x_max and y_max in 0-1 normalized space. Only return the numbers, nothing else. If no, return: there is no wolf ",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)


print(response.choices[0])
