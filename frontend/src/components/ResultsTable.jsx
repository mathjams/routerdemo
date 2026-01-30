import React, { useState } from 'react';
import { CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';

const ResultsTable = ({ results, onRowClick, selectedIndex }) => {
  const [sortField, setSortField] = useState('predicted_label');
  const [sortDirection, setSortDirection] = useState('asc');
  const [expandedRows, setExpandedRows] = useState(new Set());

  if (!results || results.length === 0) {
    return null;
  }

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const sortedResults = [...results].sort((a, b) => {
    let aVal = a[sortField];
    let bVal = b[sortField];

    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }

    if (sortDirection === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  const toggleRowExpand = (index) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  const SortableHeader = ({ field, children }) => (
    <th
      className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors"
      onClick={() => handleSort(field)}
    >
      <div className="flex items-center gap-2">
        {children}
        {sortField === field && (
          sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
        )}
      </div>
    </th>
  );

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4">Prediction Results</h2>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Preview
              </th>
              <SortableHeader field="filename">Filename</SortableHeader>
              <SortableHeader field="predicted_label">Routed Prediction</SortableHeader>
              <SortableHeader field="confidence">Confidence</SortableHeader>
              <SortableHeader field="route_chosen">Route</SortableHeader>
              <SortableHeader field="branch4_predicted_label">Branch 4 Prediction</SortableHeader>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Match?
              </th>
              {results[0].ground_truth !== null && (
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Correct
                </th>
              )}
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedResults.map((result, index) => {
              const isExpanded = expandedRows.has(index);
              const isSelected = selectedIndex === index;

              return (
                <React.Fragment key={index}>
                  <tr
                    className={`hover:bg-gray-50 cursor-pointer transition-colors ${
                      isSelected ? 'bg-primary-50' : ''
                    }`}
                    onClick={() => {
                      onRowClick?.(index);
                      toggleRowExpand(index);
                    }}
                  >
                    {/* Preview */}
                    <td className="px-4 py-4 whitespace-nowrap">
                      <img
                        src={result.image_preview}
                        alt={result.filename}
                        className="w-16 h-16 object-cover rounded border border-gray-200"
                      />
                    </td>

                    {/* Filename */}
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900 max-w-xs truncate">
                      {result.filename}
                    </td>

                    {/* Routed Prediction */}
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {result.predicted_label}
                      </div>
                      <div className="text-xs text-gray-500">
                        Class {result.predicted_class}
                      </div>
                    </td>

                    {/* Confidence */}
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${result.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>

                    {/* Route */}
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        Branch {result.route_chosen}
                      </span>
                    </td>

                    {/* Branch 4 Prediction */}
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-700">
                        {result.branch4_predicted_label}
                      </div>
                      <div className="text-xs text-gray-500">
                        {(result.branch4_confidence * 100).toFixed(1)}%
                      </div>
                    </td>

                    {/* Match between Routed and Branch 4 */}
                    <td className="px-4 py-4 whitespace-nowrap">
                      {result.predicted_class === result.branch4_predicted_class ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <XCircle className="h-5 w-5 text-orange-500" />
                      )}
                    </td>

                    {/* Correctness */}
                    {result.ground_truth !== null && (
                      <td className="px-4 py-4 whitespace-nowrap">
                        {result.is_correct ? (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        ) : (
                          <XCircle className="h-5 w-5 text-red-500" />
                        )}
                      </td>
                    )}

                    {/* Expand */}
                    <td className="px-4 py-4 whitespace-nowrap text-right text-sm">
                      {isExpanded ? (
                        <ChevronUp className="h-5 w-5 text-gray-400" />
                      ) : (
                        <ChevronDown className="h-5 w-5 text-gray-400" />
                      )}
                    </td>
                  </tr>

                  {/* Expanded Row */}
                  {isExpanded && (
                    <tr className="bg-gray-50">
                      <td colSpan="9" className="px-4 py-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div className="col-span-2 font-semibold text-gray-700 border-b pb-2">
                            Routed Prediction (Branch {result.route_chosen})
                          </div>
                          <div>
                            <span className="font-semibold">Predicted Class:</span> {result.predicted_class}
                          </div>
                          <div>
                            <span className="font-semibold">Predicted Label:</span> {result.predicted_label}
                          </div>
                          <div>
                            <span className="font-semibold">Confidence:</span> {result.confidence.toFixed(4)}
                          </div>
                          <div>
                            <span className="font-semibold">Route Confidence:</span> {result.route_confidence.toFixed(4)}
                          </div>

                          <div className="col-span-2 font-semibold text-gray-700 border-b pb-2 mt-4">
                            Branch 4 Comparison (Largest Model)
                          </div>
                          <div>
                            <span className="font-semibold">Branch 4 Class:</span> {result.branch4_predicted_class}
                          </div>
                          <div>
                            <span className="font-semibold">Branch 4 Label:</span> {result.branch4_predicted_label}
                          </div>
                          <div>
                            <span className="font-semibold">Branch 4 Confidence:</span> {result.branch4_confidence.toFixed(4)}
                          </div>
                          <div>
                            <span className="font-semibold">Predictions Match:</span>{' '}
                            {result.predicted_class === result.branch4_predicted_class ? (
                              <span className="text-green-600">Yes ✓</span>
                            ) : (
                              <span className="text-orange-600">No ✗</span>
                            )}
                          </div>

                          {result.ground_truth !== null && (
                            <>
                              <div className="col-span-2 font-semibold text-gray-700 border-b pb-2 mt-4">
                                Ground Truth
                              </div>
                              <div>
                                <span className="font-semibold">Ground Truth:</span> {result.ground_truth}
                              </div>
                              <div>
                                <span className="font-semibold">Routed Correct:</span>{' '}
                                {result.is_correct ? 'Yes' : 'No'}
                              </div>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ResultsTable;
