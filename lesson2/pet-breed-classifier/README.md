# Pet Breed Classifier on HuggingFace Space using Gradio

Check the amazing pet breed space in my [HuggingFace Space](https://huggingface.co/spaces/saicmsaicm/pet-breed)

[HuggingFace Space API endpoint](https://saicmsaicm-pet-breed.hf.space/)

``` python
import requests
import json
from gradio_client import Client
from IPython.display import Image

image_url = "https://petkeen.com/wp-content/uploads/2021/05/grey-cat.jpeg"

client = Client("https://saicmsaicm-pet-breed.hf.space/")
result = client.predict(
    image_url,
    api_name="/predict"
)

with open(result, "r") as f:
    result_json = json.load(f)

print(
    f"The breed of this pet is a {(' '.join(result_json['label'].split('_')))}"
)

display(Image(url=image_url, width=475))

print("Original JSON returned from the request: ", json.dumps(result_json, indent=2))
```

[Simple Client Github Repo](https://github.com/mchen50/pet-breed-ui)
[Simple Client Guthub Page](https://mchen50.github.io/pet-breed-ui/)
