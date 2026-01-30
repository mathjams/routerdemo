# Upload Limits Guide

## Current Default Limits

- **Max request size:** 16 MB (FastAPI default)
- **Max files:** No hard limit (limited by request size)
- **Practical limit:** ~30-100 images depending on size

## Recommended Limits by Use Case

### Testing/Demo (Default Settings)
- **Max images:** 50-100 images
- **Max size per image:** 5 MB
- **Total batch:** Under 16 MB
- **No changes needed**

### Production/Large Batches
- **Max images:** 500+ images
- **Max size per image:** 10 MB
- **Total batch:** 100 MB+
- **Requires configuration changes**

## Increase Upload Limits

### Method 1: Using Uvicorn Config (Recommended)

Edit `backend/run.py`:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        # Increase limits
        limit_max_requests=10000,        # Max concurrent requests
        timeout_keep_alive=120,          # Keep-alive timeout (seconds)
    )
```

### Method 2: Nginx Configuration (Production)

If using Nginx as reverse proxy:

```nginx
# In your nginx.conf
http {
    client_max_body_size 100M;  # Max upload size
    client_body_timeout 300s;    # Upload timeout
}
```

### Method 3: Environment Variable

Set before starting:

```bash
# Linux/Mac
export UVICORN_LIMIT_MAX_REQUESTS=10000
python3 run.py

# Windows
set UVICORN_LIMIT_MAX_REQUESTS=10000
python run.py
```

## Memory Limits

Processing batch size is determined by:

1. **Model size:** Larger models need more memory
2. **Number of images:** Each image uses ~3KB after resize
3. **Available RAM:** Check with `top` or Activity Monitor

### Memory Usage Estimates

| Batch Size | Model Memory | Image Memory | Total (approx) |
|------------|--------------|--------------|----------------|
| 50 images  | ~30-100 MB   | 0.15 MB      | ~30-100 MB     |
| 100 images | ~30-100 MB   | 0.3 MB       | ~30-100 MB     |
| 500 images | ~30-100 MB   | 1.5 MB       | ~32-102 MB     |
| 1000 images| ~30-100 MB   | 3 MB         | ~33-103 MB     |

**Memory is not usually the bottleneck** - request size limit is.

## Batch Processing for Large Datasets

For very large datasets (1000+ images), split into batches:

### Frontend Batching (Option 1)

Modify `frontend/src/App.jsx`:

```javascript
const handleSubmit = async () => {
  const BATCH_SIZE = 100;
  const batches = [];

  // Split files into batches
  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    batches.push(files.slice(i, i + BATCH_SIZE));
  }

  // Process each batch
  for (const batch of batches) {
    const formData = new FormData();
    batch.forEach(file => formData.append('files', file));

    const response = await axios.post(`${API_BASE_URL}/predict`, formData);
    // Accumulate results
  }
};
```

### Script-Based Upload (Option 2)

For programmatic use:

```python
import requests
import glob

API_URL = "http://localhost:8000/predict"
images = glob.glob("images/*.jpg")
BATCH_SIZE = 100

for i in range(0, len(images), BATCH_SIZE):
    batch = images[i:i + BATCH_SIZE]
    files = [('files', open(img, 'rb')) for img in batch]

    response = requests.post(API_URL, files=files)
    results = response.json()
    # Process results
```

## Performance Considerations

### Upload Speed

- **10 images (20 MB):** ~1-2 seconds
- **50 images (100 MB):** ~5-10 seconds
- **100 images (200 MB):** ~10-20 seconds

Depends on:
- Network speed
- Image sizes
- Server processing power

### Processing Speed

- **CPU:** ~50-200 ms per image
- **GPU:** ~5-20 ms per image

For 100 images:
- **CPU:** ~5-20 seconds
- **GPU:** ~0.5-2 seconds

## Troubleshooting

### "Request Entity Too Large" (413)

**Cause:** Batch exceeds request size limit

**Fix:**
1. Reduce number of images per upload
2. Compress images before upload
3. Increase server limits (see above)

### "Request Timeout" (504)

**Cause:** Processing takes too long

**Fix:**
1. Reduce batch size
2. Use GPU for faster inference
3. Increase timeout settings

### "Out of Memory" (500)

**Cause:** Not enough RAM for batch

**Fix:**
1. Reduce batch size
2. Close other applications
3. Use smaller models
4. Upgrade server RAM

## Recommended Settings

### Development
```python
# backend/run.py
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    timeout_keep_alive=60,  # 1 minute
)
```

Max: ~100 images

### Production
```python
# backend/run.py
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=False,
    workers=4,  # Multiple workers
    timeout_keep_alive=300,  # 5 minutes
)
```

Max: 500+ images (with batching)

## Summary

| Setting | Default | Recommended | Maximum |
|---------|---------|-------------|---------|
| **Files per upload** | Unlimited | 100 | 1000+ (batched) |
| **Total size** | 16 MB | 100 MB | 1 GB (with config) |
| **Processing time** | 120s | 300s | 600s |
| **Concurrent requests** | 100 | 1000 | 10000 |

**For most demos:** Default settings are fine (50-100 images)

**For production:** Increase limits based on your needs
