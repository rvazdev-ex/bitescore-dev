import clsx from "clsx";

interface Props {
  percent: number;
  label?: string;
  className?: string;
}

export default function ProgressBar({ percent, label, className }: Props) {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));

  return (
    <div className={clsx("w-full", className)}>
      {label && (
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-brand-600">{label}</span>
          <span className="text-sm font-semibold text-brand-500">
            {clamped}%
          </span>
        </div>
      )}
      <div className="w-full h-3 rounded-full bg-brand-100 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-500 transition-all duration-500 ease-out"
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
