import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle, Image, Loader2 } from 'lucide-react';

import FileUpload from './components/FileUpload';
import MetricsCards from './components/MetricsCards';
import ResultsTable from './components/ResultsTable';
import RouterGraph from './components/RouterGraph';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function api(path, options) {
  let response;
  try {
    response = await fetch(API_URL + path, options);
  } catch {
    throw new Error(`Can't reach the backend at ${API_URL}. Is it running?`);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (HTTP ${response.status})`);
  return body;
}

function Alert({ tone, children }) {
  const colors = tone === 'red' ? 'bg-red-50 border-red-200 text-red-800' : 'bg-yellow-50 border-yellow-200 text-yellow-800';
  return (
    <div className={`border rounded-lg p-4 flex items-start gap-3 text-sm ${colors}`}>
      <AlertCircle className="h-5 w-5 flex-shrink-0" />
      <p>{children}</p>
    </div>
  );
}

function BackendStatus({ modelInfo, offline }) {
  const [Icon, color, text] = modelInfo
    ? [CheckCircle, 'text-green-500', `Models loaded (${modelInfo.branches.length} branches)`]
    : offline
      ? [AlertCircle, 'text-yellow-500', 'Backend not reachable']
      : [Loader2, 'text-gray-400 animate-spin', 'Connecting to backend…'];
  return (
    <div className="flex items-center gap-2 text-sm text-gray-700">
      <Icon className={`h-5 w-5 ${color}`} />
      {text}
    </div>
  );
}

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [offline, setOffline] = useState(false);
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [selected, setSelected] = useState(null); // index of the clicked result
  const [error, setError] = useState(null);

  const loadModelInfo = () =>
    api('/model-info').then((info) => {
      setModelInfo(info);
      setOffline(false);
    });

  useEffect(() => {
    loadModelInfo().catch(() => setOffline(true));
  }, []);

  const runInference = async () => {
    setIsLoading(true);
    setError(null);
    setResponse(null);
    setSelected(null);
    try {
      if (!modelInfo) await loadModelInfo();
      const form = new FormData();
      files.forEach((file) => form.append('files', file));
      setResponse(await api('/predict', { method: 'POST', body: form }));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Adaptive Router Demo</h1>
            <p className="mt-1 text-sm text-gray-600">
              A router network sends each image to one of several classifiers of different sizes
            </p>
          </div>
          <BackendStatus modelInfo={modelInfo} offline={offline} />
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {error && <Alert tone="red">{error}</Alert>}

        <FileUpload files={files} setFiles={setFiles} onSubmit={runInference} isLoading={isLoading} />

        {response?.skipped.length > 0 && (
          <Alert tone="yellow">
            Skipped {response.skipped.length} file(s) that could not be read as images:{' '}
            {response.skipped.slice(0, 5).join(', ')}
            {response.skipped.length > 5 && ` and ${response.skipped.length - 5} more`}
          </Alert>
        )}

        {response && (
          <>
            <MetricsCards metrics={response.metrics} baselineBranch={modelInfo.baseline_branch} />
            <RouterGraph
              modelInfo={modelInfo}
              routing={response.metrics.routing}
              selectedRoute={response.results[selected]?.route_chosen}
            />
            <ResultsTable
              results={response.results}
              baselineBranch={modelInfo.baseline_branch}
              selected={selected}
              onSelect={setSelected}
            />
          </>
        )}

        {!response && !isLoading && (
          <div className="card text-center py-12">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Image className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No results yet</h3>
            <p className="text-gray-600">Upload some images and run inference to see results here</p>
          </div>
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <p className="py-6 text-center text-sm text-gray-500">Adaptive Router Demo · FastAPI + PyTorch + React</p>
      </footer>
    </div>
  );
}
