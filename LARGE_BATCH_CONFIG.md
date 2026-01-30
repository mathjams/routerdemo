# Large Batch Configuration (5000+ Images)

This guide explains how to configure the system to handle 5000+ images in a single batch.

## Backend Configuration

### 1. Update Environment Variables

Create or edit `backend/.env`:

```bash
# Allow large request bodies (500MB for ~5000 images)
MAX_REQUEST_SIZE=524288000

# Increase timeout (10 minutes for 5000 images)
REQUEST_TIMEOUT=600
```

### 2. Uvicorn Configuration

The [backend/run.py](backend/run.py) is already configured for large batches:
- `timeout_keep_alive=600` - 10 minute timeout
- `limit_concurrency=1000` - Handle many requests
- `limit_max_requests=10000` - Worker restart limit

### 3. For Production (Render/Railway/Fly.io)

#### Render.com
Add environment variables in dashboard:
```
UVICORN_TIMEOUT_KEEP_ALIVE=600
```

Update start command:
```bash
cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 600
```

#### Railway
Add to `railway.toml`:
```toml
[deploy]
startCommand = "cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 600"
```

#### With Nginx (if using)
Add to nginx.conf:
```nginx
http {
    client_max_body_size 500M;      # Allow 500MB uploads
    client_body_timeout 600s;        # 10 minute timeout
    proxy_read_timeout 600s;         # 10 minute proxy timeout
    proxy_send_timeout 600s;
}
```

## Memory Requirements

### Estimated Memory Usage

| Batch Size | Model Memory | Image Memory | Total (approx) |
|------------|--------------|--------------|----------------|
| 100 images | ~30-100 MB   | 0.3 MB       | ~30-100 MB     |
| 1000 images| ~30-100 MB   | 3 MB         | ~33-103 MB     |
| 5000 images| ~30-100 MB   | 15 MB        | ~45-115 MB     |

**Note:** The largest model (Branch 4) determines base memory. Image tensors are small (32x32x3).

### System Requirements

For 5000 images:
- **Minimum RAM:** 2GB
- **Recommended RAM:** 4GB+
- **CPU:** 4+ cores recommended
- **GPU:** Optional (speeds up by 10-20x)

## Performance Estimates

### CPU Inference

| Batch Size | Time (CPU) | Time (GPU) |
|------------|------------|------------|
| 100 images | 10-20s     | 1-2s       |
| 1000 images| 100-200s   | 10-20s     |
| 5000 images| 500-1000s  | 50-100s    |

**Note:** Times vary based on routing distribution and hardware.

## Frontend Configuration

### Upload Progress (Optional)

For large batches, consider adding upload progress tracking. Update `frontend/src/App.jsx`:

```javascript
const handleSubmit = async () => {
  setIsLoading(true);
  setError(null);

  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600000, // 10 minute timeout
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        console.log(`Upload progress: ${percentCompleted}%`);
        // You can show this in the UI
      }
    });

    setResults(response.data.results);
    // ... rest of code
  } catch (error) {
    // ... error handling
  }
};
```

## Browser Limitations

### Memory Constraints

Browsers have memory limits for rendering large tables:
- **Chrome/Edge:** ~2GB per tab
- **Firefox:** ~1.5GB per tab
- **Safari:** ~1GB per tab

### Table Performance

For 5000 rows, the results table may be slow. Consider:

1. **Virtual Scrolling** (render only visible rows)
2. **Pagination** (show 100 rows per page)
3. **Server-side Filtering** (filter on backend)

### Recommended: Add Pagination

Update `frontend/src/components/ResultsTable.jsx` to show 100 results at a time:

```javascript
const [currentPage, setCurrentPage] = useState(0);
const ITEMS_PER_PAGE = 100;

const paginatedResults = sortedResults.slice(
  currentPage * ITEMS_PER_PAGE,
  (currentPage + 1) * ITEMS_PER_PAGE
);

// Render paginatedResults instead of sortedResults
// Add pagination controls at bottom
```

## Testing Large Batches

### Generate Test Images

```python
# generate_test_images.py
from PIL import Image
import numpy as np

for i in range(5000):
    # Create random 32x32 RGB image
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'test_images/img_{i:04d}.png')
```

### Upload via Script

```python
import requests
import glob

API_URL = "http://localhost:8000/predict"
images = glob.glob("test_images/*.png")

files = [('files', open(img, 'rb')) for img in images]

print(f"Uploading {len(files)} images...")
response = requests.post(API_URL, files=files, timeout=600)

if response.status_code == 200:
    results = response.json()
    print(f"✓ Success! Processed {len(results['results'])} images")
    print(f"Params saved: {results['metrics']['params_savings']['savings_percent']:.1f}%")
else:
    print(f"✗ Error: {response.status_code}")
```

## Troubleshooting

### "Request Entity Too Large" (413)
- Increase `client_max_body_size` in nginx
- Or split into smaller batches (e.g., 1000 images each)

### "Gateway Timeout" (504)
- Increase timeouts in uvicorn and nginx
- Or use async processing with a job queue

### "Out of Memory" (500)
- Reduce batch size
- Close other applications
- Upgrade server RAM

### Browser Freezes
- Add pagination to results table
- Use virtual scrolling
- Limit displayed results to 1000 at a time

## Recommendations

### Development
```bash
# backend/run.py already configured
python3 run.py
```
- Works for up to 5000 images
- May be slow on CPU

### Production
```bash
# Use production server with workers
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 4 \
  --timeout-keep-alive 600
```
- Multiple workers for concurrent requests
- Better for multiple users

### Best Practice
For very large datasets (10,000+ images):
- Use batch processing (1000 images per request)
- Implement async job queue (Celery/RQ)
- Store results in database
- Provide download link for results

## Summary

**For 5000 images:**
1. Backend is already configured ✓
2. Increase timeouts to 600s ✓
3. Ensure 4GB+ RAM available
4. Consider adding pagination to frontend
5. Test with smaller batches first (100, 500, 1000)
6. Monitor memory usage during processing

**Current limits:**
- Single request: 5000+ images (tested)
- Memory: ~50-200MB per batch
- Time: 10-20 minutes (CPU), 1-2 minutes (GPU)
