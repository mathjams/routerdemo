import { Activity, Cpu, Target, TrendingDown, Zap } from 'lucide-react';

const TONES = {
  green: { card: 'border-green-200 bg-green-50 text-green-800', accent: 'text-green-600', track: 'bg-green-200', fill: 'bg-green-600' },
  blue: { card: 'border-blue-200 bg-blue-50 text-blue-800', accent: 'text-blue-600', track: 'bg-blue-200', fill: 'bg-blue-600' },
};

function Stat({ icon: Icon, label, value }) {
  return (
    <div className="metric-card flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-primary-700">{label}</p>
        <p className="text-3xl font-bold text-primary-900 mt-1">{value}</p>
      </div>
      <Icon className="h-10 w-10 text-primary-500 opacity-50" />
    </div>
  );
}

function Bar({ percent, track, fill, height = 'h-2' }) {
  const width = Math.min(100, Math.max(0, percent));
  return (
    <div className={`w-full rounded-full ${height} ${track}`}>
      <div className={`rounded-full transition-all duration-500 ${height} ${fill}`} style={{ width: `${width}%` }} />
    </div>
  );
}

function SavingsCard({ tone, icon: Icon, title, note, savings, format }) {
  const { card, accent, track, fill } = TONES[tone];
  return (
    <div className={`card border-2 ${card}`}>
      <div className="flex items-center gap-3">
        <Icon className={`h-6 w-6 ${accent}`} />
        <h3 className="text-lg font-semibold">{title}</h3>
      </div>
      <p className="text-xs mt-1 mb-4">{note}</p>

      <div className="space-y-3 text-sm">
        <div className="flex justify-between">
          <span>Routed (avg)</span>
          <span className="font-semibold">{format(savings.routed)}</span>
        </div>
        <div className="flex justify-between">
          <span>Baseline (largest)</span>
          <span className="font-semibold">{format(savings.baseline)}</span>
        </div>
        <div className="flex justify-between items-center pt-3">
          <span className="text-base font-medium">Savings</span>
          <span className={`text-2xl font-bold ${accent}`}>{savings.savings_percent.toFixed(1)}%</span>
        </div>
        <Bar percent={savings.savings_percent} track={track} fill={fill} height="h-3" />
      </div>
    </div>
  );
}

function Heading({ icon: Icon, children }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <Icon className="h-6 w-6 text-primary-600" />
      <h3 className="text-lg font-semibold text-gray-900">{children}</h3>
    </div>
  );
}

export default function MetricsCards({ metrics, baselineBranch }) {
  const { total_images, avg_confidence, routing, params_savings, time_savings } = metrics;
  const largest = `Branch ${baselineBranch}`;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Stat icon={Activity} label="Total Images" value={total_images.toLocaleString()} />
        <Stat icon={Target} label="Avg Confidence" value={`${(avg_confidence * 100).toFixed(1)}%`} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SavingsCard
          tone="green"
          icon={Cpu}
          title="Parameters Saved"
          note={`Size of the branch each image ran through vs. always using ${largest} (router not counted)`}
          savings={params_savings}
          format={(params) => `${(params / 1e6).toFixed(2)}M`}
        />
        <SavingsCard
          tone="blue"
          icon={Zap}
          title="Time Saved"
          note={`Per-image latency: router + routed branch vs. router + ${largest}`}
          savings={time_savings}
          format={(ms) => `${ms.toFixed(2)} ms`}
        />
      </div>

      <div className="card">
        <Heading icon={TrendingDown}>Routing Distribution</Heading>
        <div className="space-y-3">
          {routing.map(({ count }, branch) => {
            const percent = (100 * count) / total_images;
            return (
              <div key={branch}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-gray-700">Branch {branch}</span>
                  <span className="text-gray-600">
                    {count.toLocaleString()} ({percent.toFixed(1)}%)
                  </span>
                </div>
                <Bar percent={percent} track="bg-gray-200" fill="bg-primary-600" />
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <Heading icon={Target}>Top Labels by Branch</Heading>
        <p className="text-sm text-gray-600 mb-4">Most common predicted labels among the images routed to each branch</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {routing.map(
            ({ top_labels }, branch) =>
              top_labels.length > 0 && (
                <div key={branch} className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-primary-700 mb-2">Branch {branch}</h4>
                  <ul className="list-disc list-inside marker:text-primary-500 space-y-1 text-sm text-gray-700">
                    {top_labels.map((label) => (
                      <li key={label}>{label}</li>
                    ))}
                  </ul>
                </div>
              ),
          )}
        </div>
      </div>
    </div>
  );
}
