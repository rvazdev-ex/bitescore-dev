interface Props {
  score: number;
  label: string;
}

export default function ScoreGauge({ score, label }: Props) {
  // Determine color based on score magnitude
  const getColor = (s: number) => {
    if (s >= 0.7) return { ring: "text-emerald-400", bg: "bg-emerald-50", text: "text-emerald-700" };
    if (s >= 0.4) return { ring: "text-brand-400", bg: "bg-brand-50", text: "text-brand-700" };
    if (s >= 0.2) return { ring: "text-amber-400", bg: "bg-amber-50", text: "text-amber-700" };
    return { ring: "text-red-400", bg: "bg-red-50", text: "text-red-700" };
  };

  const colors = getColor(score);
  const circumference = 2 * Math.PI * 54;
  const progress = Math.min(1, Math.max(0, score)) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          {/* Background circle */}
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-brand-100"
          />
          {/* Progress circle */}
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            strokeLinecap="round"
            className={colors.ring}
            strokeDasharray={`${progress} ${circumference}`}
            style={{ transition: "stroke-dasharray 0.8s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-2xl font-bold ${colors.text}`}>
            {score.toFixed(3)}
          </span>
        </div>
      </div>
      <span className="text-xs font-medium text-brand-400 mt-2">{label}</span>
    </div>
  );
}
