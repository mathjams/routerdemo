import React, { useRef, useState } from 'react';
import { Upload, X, AlertCircle } from 'lucide-react';

const MAX_FILES = 5000;

const FileUpload = ({ files, setFiles, onSubmit, isLoading }) => {
  const fileInputRef = useRef(null);
  const [warning, setWarning] = useState(null);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    const totalFiles = files.length + selectedFiles.length;

    if (totalFiles > MAX_FILES) {
      setWarning(`Maximum ${MAX_FILES} files allowed. Only adding first ${MAX_FILES - files.length} files.`);
      const allowedFiles = selectedFiles.slice(0, MAX_FILES - files.length);
      setFiles((prev) => [...prev, ...allowedFiles]);
      setTimeout(() => setWarning(null), 5000);
    } else {
      setFiles((prev) => [...prev, ...selectedFiles]);
      setWarning(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    const totalFiles = files.length + droppedFiles.length;

    if (totalFiles > MAX_FILES) {
      setWarning(`Maximum ${MAX_FILES} files allowed. Only adding first ${MAX_FILES - files.length} files.`);
      const allowedFiles = droppedFiles.slice(0, MAX_FILES - files.length);
      setFiles((prev) => [...prev, ...allowedFiles]);
      setTimeout(() => setWarning(null), 5000);
    } else {
      setFiles((prev) => [...prev, ...droppedFiles]);
      setWarning(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (files.length > 0 && !isLoading) {
      onSubmit();
    }
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4">Upload Images</h2>

      {/* Warning Message */}
      {warning && (
        <div className="mb-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3 flex items-start gap-2">
          <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-yellow-700">{warning}</p>
        </div>
      )}

      {/* Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          files.length >= MAX_FILES
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-60'
            : 'border-gray-300 hover:border-primary-500 cursor-pointer'
        }`}
        onClick={() => files.length < MAX_FILES && fileInputRef.current?.click()}
        onDrop={files.length < MAX_FILES ? handleDrop : (e) => e.preventDefault()}
        onDragOver={handleDragOver}
      >
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-lg font-medium text-gray-700 mb-2">
          {files.length >= MAX_FILES
            ? 'Maximum file limit reached'
            : 'Drop images here or click to browse'}
        </p>
        <p className="text-sm text-gray-500">
          {files.length >= MAX_FILES
            ? `Remove some files to upload more (${files.length.toLocaleString()} / ${MAX_FILES.toLocaleString()})`
            : 'Supports JPG, PNG, BMP, GIF, TIFF, WEBP'}
        </p>
        {files.length < MAX_FILES && (
          <p className="text-xs text-gray-400 mt-2">
            Maximum {MAX_FILES.toLocaleString()} files per batch
          </p>
        )}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="mt-6">
          <h3 className={`text-lg font-semibold mb-3 ${files.length >= MAX_FILES ? 'text-red-600' : files.length > MAX_FILES * 0.8 ? 'text-yellow-600' : ''}`}>
            Selected Files ({files.length.toLocaleString()} / {MAX_FILES.toLocaleString()})
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between bg-gray-50 rounded-lg p-3"
              >
                <span className="text-sm text-gray-700 truncate flex-1">
                  {file.name}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}
                  className="ml-2 text-red-500 hover:text-red-700 transition-colors"
                  disabled={isLoading}
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={isLoading || files.length === 0}
            className="btn btn-primary w-full mt-4"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
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
};

export default FileUpload;
