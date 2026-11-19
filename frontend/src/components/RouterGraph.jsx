const WIDTH = 640;
const HEIGHT = 420;
const CENTER = { x: WIDTH / 2, y: HEIGHT / 2 };
const ORBIT = 150; // distance from the router to each branch
const ROUTER_RADIUS = 46;

const millions = (params) => `${(params / 1e6).toFixed(2)}M params`;

function Node({ x, y, radius, className, name, model }) {
  return (
    <g className="fill-white" textAnchor="middle">
      <circle cx={x} cy={y} r={radius} className={`drop-shadow-lg transition-all duration-300 ${className}`} />
      <text x={x} y={y - 9} className="text-sm font-bold">{name}</text>
      <text x={x} y={y + 6} className="text-[11px]">{millions(model.params)}</text>
      <text x={x} y={y + 20} className="text-[11px]">{model.latency_ms.toFixed(2)} ms</text>
    </g>
  );
}

export default function RouterGraph({ modelInfo, routing, selectedRoute }) {
  const { router, branches } = modelInfo;
  const maxParams = Math.max(...branches.map((branch) => branch.params));
  const total = routing.reduce((sum, { count }) => sum + count, 0);

  // Branches sit on a circle around the router, starting at the top; bigger models get bigger nodes.
  const nodes = branches.map((branch, index) => {
    const angle = (2 * Math.PI * index) / branches.length - Math.PI / 2;
    const direction = { x: Math.cos(angle), y: Math.sin(angle) };
    return {
      branch,
      direction,
      x: CENTER.x + ORBIT * direction.x,
      y: CENTER.y + ORBIT * direction.y,
      radius: 40 + 22 * (branch.params / maxParams),
      count: routing[index].count,
      isSelected: selectedRoute === index,
    };
  });

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4">Router Architecture</h2>

      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="mx-auto w-full max-w-2xl">
        {nodes.map(({ x, y, radius, direction, count, isSelected }, index) => {
          // Label the visible part of the link, halfway between the two circles.
          const middle = (ROUTER_RADIUS + ORBIT - radius) / 2;
          const label = { x: CENTER.x + middle * direction.x, y: CENTER.y + middle * direction.y };
          return (
            <g key={index}>
              <line
                x1={CENTER.x}
                y1={CENTER.y}
                x2={x}
                y2={y}
                strokeWidth={2 + (10 * count) / total}
                strokeDasharray={count > 0 ? undefined : '5 5'}
                className={`transition-all duration-300 ${isSelected ? 'stroke-primary-500' : 'stroke-gray-300'}`}
              />
              {count > 0 && (
                <text
                  x={label.x}
                  y={label.y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="text-xs font-semibold fill-gray-600 stroke-white [paint-order:stroke] [stroke-width:4px]"
                >
                  {count.toLocaleString()}
                </text>
              )}
            </g>
          );
        })}

        <Node {...CENTER} radius={ROUTER_RADIUS} className="fill-violet-500" name="Router" model={router} />
        {nodes.map(({ branch, x, y, radius, isSelected }, index) => (
          <Node
            key={index}
            x={x}
            y={y}
            radius={radius}
            className={isSelected ? 'fill-primary-500' : 'fill-emerald-500 opacity-80'}
            name={`Branch ${index}`}
            model={branch}
          />
        ))}
      </svg>

      <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 justify-center text-sm">
        <span className="flex items-center gap-2">
          <span className="w-4 h-4 rounded-full bg-violet-500" /> Router
        </span>
        <span className="flex items-center gap-2">
          <span className="w-4 h-4 rounded-full bg-emerald-500" /> Branch
        </span>
        <span className="flex items-center gap-2">
          <span className="w-4 h-4 rounded-full bg-primary-500" /> Branch of the selected image
        </span>
        <span className="text-gray-500">Node size ~ parameters · line width ~ images routed · ms = single-image latency</span>
      </div>
    </div>
  );
}
