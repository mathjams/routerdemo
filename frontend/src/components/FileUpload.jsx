import { useRef } from 'react';
import { Loader2, Upload, X } from 'lucide-react';

const MAX_FILES = 5000; // per request; the backend enforces the same limit

export default function FileUpload({ files, setFiles, onSubmit, isLoading }) {
  const input = useRef(null);
  const isFull = files.length >= MAX_FILES;

  const addFiles = (fileList) => {
    const added = Array.from(fileList); // copy now: the browser empties the list once the event is over
    setFiles((current) => [...current, ...added].slice(0, MAX_FILES));
  };
  const removeFile = (index) => setFiles((current) => current.filter((_, i) => i !== index));

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4">Upload Images</h2>

      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isFull
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-60'
            : 'border-gray-300 hover:border-primary-500 cursor-pointer'
        }`}
        onClick={() => !isFull && input.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          addFiles(e.dataTransfer.files);
        }}
      >
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-lg font-medium text-gray-700 mb-2">
          {isFull ? 'Maximum file limit reached' : 'Drop images here or click to browse'}
        </p>
        <p className="text-sm text-gray-500">
          Any image format works; everything is resized to 32×32. Up to {MAX_FILES.toLocaleString()} files per batch.
        </p>
        <input
          ref={input}
          type="file"
          multiple
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = ''; // so picking the same file again still fires onChange
          }}
        />
      </div>

      {files.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold">
              Selected Files ({files.length.toLocaleString()} / {MAX_FILES.toLocaleString()})
            </h3>
            <button
              onClick={() => setFiles([])}
              disabled={isLoading}
              className="text-sm text-gray-500 hover:text-gray-800"
            >
              Clear all
            </button>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-50 rounded-lg p-3">
                <span className="text-sm text-gray-700 truncate">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  disabled={isLoading}
                  aria-label={`Remove ${file.name}`}
                  className="ml-2 text-red-500 hover:text-red-700 transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>

          <button onClick={onSubmit} disabled={isLoading} className="btn btn-primary w-full mt-4">
            {isLoading ? (
              <span className="flex items-center justify-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin" />
                Processing...
              </span>
            ) : (
              'Run Inference'
            )}
          </button>
        </div>
      )}
    </div>
  );
}
