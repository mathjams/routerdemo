import { useMemo, useState } from 'react';
import { CheckCircle, ChevronDown, ChevronUp, XCircle } from 'lucide-react';

const PAGE_SIZE = 200; // rows rendered at a time, so huge batches stay responsive

const compare = (a, b) =>
  typeof a === 'string' ? a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }) : a - b;

const headerClass = 'px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';

export default function ResultsTable({ results, baselineBranch, selected, onSelect }) {
  const [sort, setSort] = useState({ key: 'predicted_label', ascending: true });
  const [visible, setVisible] = useState(PAGE_SIZE);

  // Rows remember their original index, so selection survives re-sorting.
  const rows = useMemo(() => {
    const direction = sort.ascending ? 1 : -1;
    return results
      .map((result, index) => ({ ...result, index }))
      .sort(
        (a, b) =>
          direction *
          (compare(a[sort.key], b[sort.key]) ||
            compare(a.predicted_label, b.predicted_label) ||
            a.route_chosen - b.route_chosen),
      );
  }, [results, sort]);

  const SortableHeader = ({ field, children }) => (
    <th
      className={`${headerClass} cursor-pointer hover:bg-gray-100 transition-colors`}
      onClick={() => setSort({ key: field, ascending: sort.key === field ? !sort.ascending : true })}
    >
      <div className="flex items-center gap-2">
        {children}
        {sort.key === field && (sort.ascending ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />)}
      </div>
    </th>
  );

  return (
    <div className="card">
      <h2 className="text-2xl font-bold">Prediction Results</h2>
      <p className="text-sm text-gray-600 mt-1 mb-4">
        Click a row to highlight its branch in the graph. The last columns show what the largest branch
        would have predicted for the same image.
      </p>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className={headerClass}>Preview</th>
              <SortableHeader field="filename">Filename</SortableHeader>
              <SortableHeader field="predicted_label">Routed Prediction</SortableHeader>
              <SortableHeader field="confidence">Confidence</SortableHeader>
              <SortableHeader field="route_chosen">Route</SortableHeader>
              <SortableHeader field="baseline_label">Branch {baselineBranch} Prediction</SortableHeader>
              <th className={headerClass}>Match?</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {rows.slice(0, visible).map((result) => (
              <tr
                key={result.index}
                onClick={() => onSelect(result.index)}
                className={`cursor-pointer transition-colors ${
                  selected === result.index ? 'bg-primary-50' : 'hover:bg-gray-50'
                }`}
              >
                <td className="px-4 py-4">
                  <img
                    src={result.image_preview}
                    alt={result.filename}
                    className="w-16 h-16 rounded border border-gray-200"
                  />
                </td>
                <td className="px-4 py-4 text-sm text-gray-900 max-w-xs truncate">{result.filename}</td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{result.predicted_label}</div>
                  <div className="text-xs text-gray-500">Class {result.predicted_class}</div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                      <div className="bg-primary-600 h-2 rounded-full" style={{ width: `${result.confidence * 100}%` }} />
                    </div>
                    <span className="text-sm text-gray-900">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <span className="px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                    Branch {result.route_chosen}
                  </span>
                  <div className="text-xs text-gray-500 mt-1">
                    router {(result.route_confidence * 100).toFixed(1)}% sure
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-700">{result.baseline_label}</div>
                  <div className="text-xs text-gray-500">{(result.baseline_confidence * 100).toFixed(1)}%</div>
                </td>
                <td className="px-4 py-4">
                  {result.predicted_class === result.baseline_class ? (
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  ) : (
                    <XCircle className="h-5 w-5 text-orange-500" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {rows.length > visible && (
        <button onClick={() => setVisible(visible + PAGE_SIZE)} className="btn btn-secondary w-full mt-4">
          Show more ({visible.toLocaleString()} of {rows.length.toLocaleString()} shown)
        </button>
      )}
    </div>
  );
}
