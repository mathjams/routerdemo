import React from 'react';
import { Activity, Zap, Cpu, TrendingDown, Target } from 'lucide-react';

const MetricsCards = ({ metrics }) => {
  if (!metrics) return null;

  const {
    total_images,
    avg_confidence,
    accuracy,
    params_savings,
    time_savings,
    routing_distribution,
    branch_label_distribution,
  } = metrics;

  return (
    <div className="space-y-6">
      {/* Overall Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Total Images */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-primary-700">Total Images</p>
              <p className="text-3xl font-bold text-primary-900 mt-1">
                {total_images}
              </p>
            </div>
            <Activity className="h-10 w-10 text-primary-500 opacity-50" />
          </div>
        </div>

        {/* Average Confidence */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-primary-700">
                Avg Confidence
              </p>
              <p className="text-3xl font-bold text-primary-900 mt-1">
                {(avg_confidence * 100).toFixed(1)}%
              </p>
            </div>
            <Target className="h-10 w-10 text-primary-500 opacity-50" />
          </div>
        </div>

        {/* Accuracy */}
        {accuracy !== null && accuracy !== undefined && (
          <div className="metric-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-primary-700">Accuracy</p>
                <p className="text-3xl font-bold text-primary-900 mt-1">
                  {(accuracy * 100).toFixed(1)}%
                </p>
              </div>
              <Target className="h-10 w-10 text-primary-500 opacity-50" />
            </div>
          </div>
        )}
      </div>

      {/* Savings Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Parameters Saved */}
        <div className="card border-2 border-green-200 bg-green-50">
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="h-6 w-6 text-green-600" />
            <h3 className="text-lg font-semibold text-green-900">
              Parameters Saved
            </h3>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-green-700">Routed (avg)</span>
              <span className="text-sm font-semibold text-green-900">
                {(params_savings.avg_routed / 1e6).toFixed(2)}M
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-green-700">Baseline (largest)</span>
              <span className="text-sm font-semibold text-green-900">
                {(params_savings.baseline_value / 1e6).toFixed(2)}M
              </span>
            </div>
            <div className="pt-3 border-t border-green-200">
              <div className="flex justify-between items-center">
                <span className="text-base font-medium text-green-700">
                  Savings
                </span>
                <span className="text-2xl font-bold text-green-600">
                  {params_savings.savings_percent.toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 w-full bg-green-200 rounded-full h-3">
                <div
                  className="bg-green-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${params_savings.savings_percent}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Time Saved */}
        {time_savings && (
          <div className="card border-2 border-blue-200 bg-blue-50">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="h-6 w-6 text-blue-600" />
              <h3 className="text-lg font-semibold text-blue-900">
                Time Saved
              </h3>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-700">Routed (avg)</span>
                <span className="text-sm font-semibold text-blue-900">
                  {(time_savings.avg_routed / 1000).toFixed(2)} ms
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-700">Baseline (largest)</span>
                <span className="text-sm font-semibold text-blue-900">
                  {(time_savings.baseline_value / 1000).toFixed(2)} ms
                </span>
              </div>
              <div className="pt-3 border-t border-blue-200">
                <div className="flex justify-between items-center">
                  <span className="text-base font-medium text-blue-700">
                    Savings
                  </span>
                  <span className="text-2xl font-bold text-blue-600">
                    {time_savings.savings_percent.toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2 w-full bg-blue-200 rounded-full h-3">
                  <div
                    className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${time_savings.savings_percent}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Routing Distribution */}
      <div className="card">
        <div className="flex items-center gap-3 mb-4">
          <TrendingDown className="h-6 w-6 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            Routing Distribution
          </h3>
        </div>

        <div className="space-y-3">
          {Object.entries(routing_distribution)
            .sort(([a], [b]) => parseInt(a) - parseInt(b))
            .map(([route, count]) => {
              const percentage = (count / total_images) * 100;
              return (
                <div key={route}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      Branch {route}
                    </span>
                    <span className="text-sm text-gray-600">
                      {count} ({percentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Branch Label Distribution */}
      {branch_label_distribution && Object.keys(branch_label_distribution).length > 0 && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-6 w-6 text-primary-600" />
            <h3 className="text-lg font-semibold text-gray-900">
              Top Labels by Branch
            </h3>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Most common predicted labels routed to each branch
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(branch_label_distribution)
              .sort(([a], [b]) => {
                const branchA = parseInt(a.replace('Branch ', ''));
                const branchB = parseInt(b.replace('Branch ', ''));
                return branchA - branchB;
              })
              .map(([branch, labels]) => (
                <div key={branch} className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-primary-700 mb-2">{branch}</h4>
                  <ul className="space-y-1">
                    {labels.map((label, idx) => (
                      <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                        <span className="text-primary-500">•</span>
                        <span className="capitalize">{label}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricsCards;
