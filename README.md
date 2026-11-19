# Adaptive Router Demo

A small **router** network looks at each image and sends it to one of five CIFAR-100
classifiers ("branches") of increasing size, so that an image only costs as much compute as
the branch it is sent to. This repo is a web demo of that idea: upload images and see where
each one was routed, what that branch predicted, and how the cost compares with always
using the largest branch.

| Model    | Parameters | Architecture                       |
| -------- | ---------- | ---------------------------------- |
| Router   | 1.26M      | 4 conv layers, one output per branch |
| Branch 0 | 1.61M      | `SimpleCNN`, 64 channels × 4 layers |
| Branch 1 | 3.01M      | `SimpleCNN`, 88 channels × 4 layers |
| Branch 2 | 6.38M      | `SimpleCNN`, 64 channels × 5 layers |
| Branch 3 | 9.94M      | `SimpleCNN`, 80 channels × 5 layers |
| Branch 4 | 14.28M     | `SimpleCNN`, 96 channels × 5 layers |

## Run it

You need Python 3.9+ and Node 18+. The trained weights are already in the repo.

```bash
# Terminal 1: API on http://localhost:8000
pip install -r backend/requirements.txt
python backend/run.py
```

```bash
# Terminal 2: web app on http://localhost:5173
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, drop in some images and press **Run Inference**. Any image
format works, but the models are CIFAR-100 classifiers: they see every upload shrunk to
32×32 and only know [those 100 classes](backend/cifar100.py).

## What the page shows

- **Prediction results** – per image: the 32×32 input, the branch the router chose, that
  branch's prediction, and what the largest branch would have said for the same image.
  Click a row to highlight its branch in the graph.
- **Router architecture** – the router and its branches. Node size follows parameter
  count, line width follows how many images took that route.
- **Routing distribution** and **top labels by branch** – which branch handles what.
- **Parameters saved** – the average size of the branch each image ran through, against
  the largest branch. The router's own parameters are not counted.
- **Time saved** – each model's single-image latency is measured once when the server
  starts. "Routed" is router + chosen branch, averaged over the batch; the baseline is
  router + largest branch. Both sides include the router pass, so against the largest
  branch running on its own (no router) the saving is smaller.

## Layout

```
backend/
  run.py          starts the server
  app.py          FastAPI endpoints: /health, /model-info, /predict
  inference.py    preprocessing, routing, metrics
  models.py       the two architectures and the weight loader
  cifar100.py     class names and normalisation constants
  weights/        router.pth and branch_0.pth ... branch_4.pth (plain state dicts)
frontend/         React + Vite + Tailwind; src/App.jsx plus four components
training/         how the weights were produced (optional, see below)
render.yaml       backend deployment on Render
```

## API

```bash
curl -F "files=@cat.png" -F "files=@dog.jpg" http://localhost:8000/predict
```

returns one entry per image plus metrics for the batch (up to 5,000 files per request):

```json
{
  "results": [{"filename": "cat.png", "predicted_label": "tiger", "confidence": 0.71,
               "route_chosen": 2, "route_confidence": 0.22,
               "baseline_label": "tiger", "baseline_confidence": 0.84, "...": "..."}],
  "metrics": {"total_images": 2, "avg_confidence": 0.64,
              "routing": [{"count": 0, "top_labels": []}, "... one per branch"],
              "params_savings": {"routed": 6384740, "baseline": 14277220, "savings_percent": 55.28},
              "time_savings": {"routed": 1.1, "baseline": 1.7, "savings_percent": 35.3}},
  "skipped": []
}
```

The full schema, with a form to try it, is at <http://localhost:8000/docs>.

## Using your own models

Replace the files in `backend/weights/` with your own state dicts: `router.pth` (a
`RouterModel`) and one `branch_<i>.pth` (a `SimpleCNN`) per router output. The number of
branches and their sizes are read from the weights, so nothing else needs changing. For a
dataset other than CIFAR-100, also edit `backend/cifar100.py`.

## Training (optional)

`training/` contains the code that produced the weights. It needs `torchvision`, downloads
CIFAR-100 (~170 MB) into `./data`, and realistically needs a GPU.

```bash
pip install torchvision
python -m training.train                                         # writes runs/<timestamp>/
python -m training.evaluate --weights runs/<timestamp>/weights   # accuracy on the test set
python -m training.evaluate                                      # same, for backend/weights
```

Training alternates between two phases: the branches train on the samples the router sends
them, then the router (with branches frozen) learns to pick the branch with the lowest
loss + cost for each sample while keeping every branch in use. The hyperparameters and
branch sizes are at the top of [`training/train.py`](training/train.py). A full run is
4,000 outer steps and keeps a ~450 MB checkpoint every 10 steps, so check your disk first.
To serve a trained model, copy `runs/<timestamp>/weights/*.pth` into `backend/weights/`.

## Deploying

- **Backend** – `render.yaml` deploys it on [Render](https://render.com). On any other
  host: `pip install -r backend/requirements.txt`, then `python backend/run.py`, which
  listens on `$PORT`.
- **Frontend** – put the backend's URL in `frontend/.env.production`, then run
  `npm run deploy` in `frontend/` to publish the build to the repo's `gh-pages` branch
  (GitHub Pages).

## License

MIT
