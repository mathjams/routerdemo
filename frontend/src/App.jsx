import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AlertCircle, CheckCircle } from 'lucide-react';

import FileUpload from './components/FileUpload';
import RouterGraph from './components/RouterGraph';
import ResultsTable from './components/ResultsTable';
import MetricsCards from './components/MetricsCards';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [selectedRowIndex, setSelectedRowIndex] = useState(null);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  // Check health on mount
  useEffect(() => {
    checkHealth();
    fetchModelInfo();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setHealthStatus(response.data);
    } catch (err) {
      console.error('Health check failed:', err);
      setHealthStatus({ status: 'error', models_loaded: false });
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model-info`);
      setModelInfo(response.data);
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  };

  const handleSubmit = async () => {
    if (files.length === 0) return;

    setIsLoading(true);
    setError(null);
    setResults(null);
    setSelectedRoute(null);
    setSelectedRowIndex(null);

    try {
      // Prepare form data
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      // Optional: Add ground truth labels here if you have them
      // const labels = [0, 1, 2, ...]; // Example
      // formData.append('labels', JSON.stringify(labels));

      // Make request with extended timeout for large batches
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 600000, // 10 minutes for large batches (5000+ images)
      });

      setResults(response.data);
      console.log('Prediction results:', response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      const errorMessage =
        err.response?.data?.detail || err.message || 'Unknown error occurred';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRowClick = (index) => {
    setSelectedRowIndex(index);
    if (results?.results?.[index]) {
      setSelectedRoute(results.results[index].route_chosen);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Adaptive Router Demo
              </h1>
              <p className="mt-1 text-sm text-gray-600">
                Visualize dynamic model routing with PyTorch
              </p>
            </div>

            {/* Health Status */}
            {healthStatus && (
              <div className="flex items-center gap-2">
                {healthStatus.models_loaded ? (
                  <>
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm text-gray-700">
                      Models Loaded ({healthStatus.num_branches} branches)
                    </span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="h-5 w-5 text-yellow-500" />
                    <span className="text-sm text-gray-700">
                      Models Not Loaded
                    </span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Large Batch Warning */}
        {files.length > 1000 && (
          <div className="mb-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-yellow-800">Large Batch ({files.length} images)</h3>
              <p className="text-sm text-yellow-700 mt-1">
                Processing may take several minutes. The page may appear unresponsive during inference.
              </p>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="mb-8">
          <FileUpload
            files={files}
            setFiles={setFiles}
            onSubmit={handleSubmit}
            isLoading={isLoading}
          />
        </div>

        {/* Results Section */}
        {results && (
          <div className="space-y-8">
            {/* Metrics Cards */}
            <MetricsCards metrics={results.metrics} />

            {/* Router Graph */}
            <RouterGraph
              modelInfo={modelInfo}
              selectedRoute={selectedRoute}
              routingDistribution={results.metrics.routing_distribution}
            />

            {/* Results Table */}
            <ResultsTable
              results={results.results}
              onRowClick={handleRowClick}
              selectedIndex={selectedRowIndex}
            />
          </div>
        )}

        {/* Empty State */}
        {!results && !isLoading && (
          <div className="card text-center py-12">
            <div className="max-w-md mx-auto">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-8 h-8 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No results yet
              </h3>
              <p className="text-gray-600">
                Upload some images and run inference to see results here
              </p>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            Adaptive Router Demo - Built with FastAPI & React
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
