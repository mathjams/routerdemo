import React from 'react';
import { Network, Cpu, Zap } from 'lucide-react';

const RouterGraph = ({ modelInfo, selectedRoute, routingDistribution }) => {
  if (!modelInfo) {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4">Router Architecture</h2>
        <p className="text-gray-500">Loading model information...</p>
      </div>
    );
  }

  const { router, branches } = modelInfo;

  // Find largest branch for normalization
  const maxParams = Math.max(...branches.map((b) => b.params));

  // Calculate node sizes (proportional to params)
  const getNodeSize = (params) => {
    const minSize = 40;
    const maxSize = 80;
    const ratio = params / maxParams;
    return minSize + (maxSize - minSize) * ratio;
  };

  // SVG dimensions
  const width = 800;
  const height = 400;
  const routerX = width / 2;
  const routerY = height / 2;
  const routerSize = 60;

  // Calculate branch positions (circular layout)
  const branchPositions = branches.map((_, index) => {
    const angle = (index * 2 * Math.PI) / branches.length - Math.PI / 2;
    const radius = 140;
    return {
      x: routerX + radius * Math.cos(angle),
      y: routerY + radius * Math.sin(angle),
    };
  });

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4">Router Architecture</h2>

      <svg
        width={width}
        height={height}
        className="mx-auto"
        style={{ maxWidth: '100%', height: 'auto' }}
      >
        {/* Connections */}
        {branches.map((branch, index) => {
          const pos = branchPositions[index];
          const isSelected = selectedRoute === index;
          const routeCount = routingDistribution?.[index] || 0;

          return (
            <g key={`connection-${index}`}>
              {/* Line */}
              <line
                x1={routerX}
                y1={routerY}
                x2={pos.x}
                y2={pos.y}
                stroke={isSelected ? '#0ea5e9' : '#d1d5db'}
                strokeWidth={isSelected ? 3 : 2}
                strokeDasharray={isSelected ? '' : '5,5'}
                className="transition-all duration-300"
              />

              {/* Arrow */}
              <polygon
                points={`${pos.x},${pos.y} ${pos.x - 5},${pos.y - 5} ${pos.x + 5},${pos.y - 5}`}
                fill={isSelected ? '#0ea5e9' : '#d1d5db'}
                transform={`rotate(${(index * 360) / branches.length - 90}, ${pos.x}, ${pos.y})`}
              />

              {/* Route count label */}
              {routeCount > 0 && (
                <text
                  x={(routerX + pos.x) / 2}
                  y={(routerY + pos.y) / 2}
                  className="text-xs font-semibold"
                  fill="#6b7280"
                  textAnchor="middle"
                >
                  {routeCount}
                </text>
              )}
            </g>
          );
        })}

        {/* Router Node */}
        <g>
          <circle
            cx={routerX}
            cy={routerY}
            r={routerSize}
            fill="#8b5cf6"
            className="drop-shadow-lg"
          />
          <text
            x={routerX}
            y={routerY - 5}
            className="text-sm font-bold"
            fill="white"
            textAnchor="middle"
          >
            Router
          </text>
          <text
            x={routerX}
            y={routerY + 10}
            className="text-xs"
            fill="white"
            textAnchor="middle"
          >
            {router.params_m}M params
          </text>
        </g>

        {/* Branch Nodes */}
        {branches.map((branch, index) => {
          const pos = branchPositions[index];
          const size = getNodeSize(branch.params);
          const isSelected = selectedRoute === index;

          return (
            <g key={`branch-${index}`}>
              {/* Node circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={size}
                fill={isSelected ? '#0ea5e9' : '#10b981'}
                className="drop-shadow-lg transition-all duration-300"
                opacity={isSelected ? 1 : 0.8}
              />

              {/* Branch label */}
              <text
                x={pos.x}
                y={pos.y - 10}
                className="text-sm font-bold"
                fill="white"
                textAnchor="middle"
              >
                Branch {index}
              </text>

              {/* Params */}
              <text
                x={pos.x}
                y={pos.y + 5}
                className="text-xs"
                fill="white"
                textAnchor="middle"
              >
                {branch.params_m}M
              </text>

              {/* FLOPs (if available) */}
              {branch.flops_m && (
                <text
                  x={pos.x}
                  y={pos.y + 18}
                  className="text-xs"
                  fill="white"
                  textAnchor="middle"
                >
                  {branch.flops_m}M FLOPs
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="mt-6 flex flex-wrap gap-6 justify-center text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-purple-500"></div>
          <span>Router</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-green-500"></div>
          <span>Branch (idle)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-primary-500"></div>
          <span>Branch (selected)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-1 bg-gray-300"></div>
          <span>Node size ~ parameters</span>
        </div>
      </div>
    </div>
  );
};

export default RouterGraph;
