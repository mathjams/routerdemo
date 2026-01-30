# Deployment Guide - GitHub Pages + Backend Hosting

This guide shows how to deploy your adaptive router demo as a public website.

## Architecture

- **Frontend (React):** GitHub Pages (static hosting)
- **Backend (FastAPI):** Render/Railway/Fly.io (free tier)

---

## Step 1: Prepare Frontend for Production

### 1.1 Update API URL

Create an environment-aware API configuration:

**frontend/.env.production** (create this file):
```
VITE_API_URL=https://your-backend.onrender.com
```

**frontend/.env.development** (create this file):
```
VITE_API_URL=http://localhost:8000
```

### 1.2 Update frontend/src/App.jsx

Change the API_BASE_URL line to:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### 1.3 Build the frontend

```bash
cd frontend
npm run build
```

This creates a `dist/` folder with optimized static files.

---

## Step 2: Deploy Backend (Choose One)

### Option A: Render (Recommended - Free Tier)

1. **Sign up at [render.com](https://render.com)**

2. **Create a new Web Service:**
   - Connect your GitHub repo
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.9
   - Instance Type: Free

3. **Upload model files:**
   - Since GitHub has file size limits, you'll need to upload models separately
   - Options:
     - Use Render's persistent disk (paid)
     - Host models on Dropbox/Google Drive and download on startup
     - Use Git LFS

4. **Copy your backend URL** (e.g., `https://adaptive-router.onrender.com`)

### Option B: Railway

1. **Sign up at [railway.app](https://railway.app)**

2. **Create new project from GitHub repo**

3. **Configure:**
   - Root Directory: `backend`
   - Start Command: `python run.py`
   - Add environment variable: `USE_CUDA=0`

4. **Deploy and copy the URL**

### Option C: Fly.io

1. **Install flyctl:** `curl -L https://fly.io/install.sh | sh`

2. **Create fly.toml** in backend directory:
   ```toml
   app = "adaptive-router"

   [build]
     dockerfile = "Dockerfile"

   [[services]]
     http_checks = []
     internal_port = 8000
     protocol = "tcp"

     [[services.ports]]
       handlers = ["http"]
       port = 80

     [[services.ports]]
       handlers = ["tls", "http"]
       port = 443
   ```

3. **Deploy:**
   ```bash
   cd backend
   fly launch
   fly deploy
   ```

---

## Step 3: Deploy Frontend to GitHub Pages

### 3.1 Update frontend config with backend URL

Edit `frontend/.env.production`:
```
VITE_API_URL=https://your-backend-url.onrender.com
```

### 3.2 Rebuild frontend

```bash
cd frontend
npm run build
```

### 3.3 Deploy to GitHub Pages

**Method 1: Using gh-pages package**

```bash
cd frontend
npm install --save-dev gh-pages

# Add to package.json scripts:
# "deploy": "npm run build && gh-pages -d dist"

npm run deploy
```

**Method 2: Manual deployment**

1. Push your repo to GitHub
2. Go to repo Settings → Pages
3. Source: Deploy from a branch
4. Branch: Create a `gh-pages` branch
5. Copy `frontend/dist/*` to the `gh-pages` branch root
6. Push and wait for deployment

### 3.4 Configure GitHub Pages

1. Go to your repo → Settings → Pages
2. Source: `gh-pages` branch, `/` (root)
3. Save
4. Your site will be at: `https://yourusername.github.io/routerdemo/`

---

## Step 4: Update CORS Settings

Update `backend/app/main.py` to allow your GitHub Pages domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://yourusername.github.io",  # Add your GitHub Pages URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Handling Large Model Files

GitHub has a 100MB file size limit. Your models are larger. Solutions:

### Solution 1: Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track model files
cd models
git lfs track "*.pth"
git add .gitattributes
git add *.pth
git commit -m "Add models with Git LFS"
git push
```

### Solution 2: Download models on startup

Create `backend/download_models.py`:
```python
import requests
from pathlib import Path

MODELS = {
    'router.pth': 'https://your-storage-url/router.pth',
    'branch_0.pth': 'https://your-storage-url/branch_0.pth',
    # ... etc
}

def download_models():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    for filename, url in MODELS.items():
        filepath = models_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url)
            filepath.write_bytes(response.content)
            print(f"✓ Downloaded {filename}")

if __name__ == '__main__':
    download_models()
```

Update start command to run this first:
```bash
python download_models.py && python run.py
```

### Solution 3: Use Hugging Face Hub

1. Upload models to Hugging Face: https://huggingface.co
2. Download on startup using `huggingface_hub`:
   ```python
   from huggingface_hub import hf_hub_download

   model_path = hf_hub_download(
       repo_id="your-username/adaptive-router",
       filename="router.pth"
   )
   ```

---

## Alternative: Full Static Demo (No Backend)

If you want a pure GitHub Pages deployment without a backend:

1. **Pre-generate demo results:**
   - Run inference on a fixed set of images
   - Save results as JSON
   - Load from JSON in frontend

2. **Create `frontend/public/demo-data.json`:**
   ```json
   {
     "results": [...],
     "metrics": {...},
     "modelInfo": {...}
   }
   ```

3. **Update App.jsx to load from JSON instead of API**

This gives you a demo-only site without live inference.

---

## Recommended: Development Workflow

1. **Local development:**
   ```bash
   # Terminal 1 - Backend
   cd backend && python run.py

   # Terminal 2 - Frontend
   cd frontend && npm run dev
   ```

2. **Production deployment:**
   - Push to GitHub
   - Backend auto-deploys on Render/Railway
   - Frontend: `cd frontend && npm run deploy`

---

## Cost Breakdown

- **GitHub Pages:** Free
- **Render Free Tier:** Free (sleeps after 15 min inactivity, 750 hours/month)
- **Railway Free Tier:** Free ($5 credit/month, ~500 hours)
- **Fly.io Free Tier:** Free (3 shared-cpu-1x VMs)
- **Git LFS:** Free (1GB storage, 1GB bandwidth/month)
- **Hugging Face:** Free (unlimited public models)

**Total: $0/month** for a fully functional demo site!

---

## Testing Your Deployment

1. Visit your GitHub Pages URL
2. Upload test images
3. Check browser console for any CORS errors
4. Verify backend is responding (check Network tab)
5. Test with multiple images to ensure backend doesn't timeout

---

## Troubleshooting

### "CORS policy" error
- Update backend CORS settings with your GitHub Pages URL
- Redeploy backend

### Backend is slow/timing out
- Free tier backends "sleep" after inactivity (cold start ~30s)
- First request after sleep takes longer
- Consider keeping backend warm with a cron job ping

### Models not loading
- Check backend logs for model loading errors
- Ensure models are accessible (Git LFS, download script, etc.)
- Check file permissions

### Build fails
- Verify all dependencies are in requirements.txt
- Check Python version matches (3.9+)
- Ensure torch is compatible with deployment platform

---

## Next Steps

1. Choose backend hosting (Render recommended)
2. Update frontend .env.production with backend URL
3. Set up Git LFS or model hosting
4. Deploy backend first, test it works
5. Deploy frontend to GitHub Pages
6. Share your demo URL! 🎉
