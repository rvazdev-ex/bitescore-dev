import type { FeatureEntry } from "../types";

interface Props {
  features: FeatureEntry[];
}

export default function FeatureTable({ features }: Props) {
  if (features.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-sm text-brand-400 italic">
          No data available for this feature category.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-brand-100">
            <th className="text-left py-2.5 px-3 text-xs font-semibold text-brand-400 uppercase tracking-wider">
              Metric
            </th>
            <th className="text-right py-2.5 px-3 text-xs font-semibold text-brand-400 uppercase tracking-wider">
              Value
            </th>
          </tr>
        </thead>
        <tbody>
          {features.map(({ metric, value }, i) => (
            <tr
              key={metric}
              className="border-b border-brand-50 last:border-0 hover:bg-brand-50/40 transition-colors"
            >
              <td className="py-2.5 px-3 text-brand-700 font-medium">
                <code className="text-xs bg-brand-50 px-1.5 py-0.5 rounded">
                  {metric}
                </code>
              </td>
              <td className="py-2.5 px-3 text-right text-brand-600 font-mono text-xs">
                {formatValue(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatValue(v: string): string {
  if (v === "—" || v === "None" || !v) return "\u2014";
  // Highlight booleans
  if (v === "Yes" || v === "True") return "\u2705 Yes";
  if (v === "No" || v === "False") return "No";
  return v;
}
