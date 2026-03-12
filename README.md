Local test:

```bash
docker compose up --build
```

Text emdbeddings:
```bash
MODEL="Qwen3-VL-Embedding-8B"
curl -s http://localhost:9000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'${MODEL}'",
    "input": ["The food was delicious and the waiter..."],
    "dimensions": 10,
    "encoding_format": "float"
  }' | jq
```

Image emdbeddings:
```bash
MODEL="Qwen3-VL-Embedding-8B"
curl -s http://localhost:9000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'${MODEL}'",
    "input": ["https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "https://example.com/123.jpg"],
    "dimensions": 5,
    "encoding_format": "float",
    "modality": "image"
  }' | jq
```

Output:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        1.078125,
        4.28125,
        1.7734375,
        3.09375,
        -8.4375
      ],
      "index": 0
    },
    {
      "object": "error",
      "message": "Failed to fetch image at index 1 (https://example.com/123.jpg): 404, message='Not Found', url='https://example.com/123.jpg'",
      "index": 1
    }
  ],
  "model": "Qwen3-VL-Embedding-8B",
  "usage": {
    "prompt_tokens": 1259,
    "total_tokens": 1259
  }
}
```

Test after deploy:
```bash
MODEL="Qwen3-VL-Embedding-8B"
RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
curl -s https://<ENDPOINT_ID>.api.runpod.ai/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "model": "'${MODEL}'",
    "input": ["The food was delicious and the waiter..."],
    "dimensions": 10,
    "encoding_format": "float"
  }' | jq
```