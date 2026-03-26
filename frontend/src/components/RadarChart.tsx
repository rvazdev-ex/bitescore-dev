import { useRef, useEffect } from "react";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Radar } from "react-chartjs-2";

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

const DISPLAY_LABELS: Record<string, string> = {
  aa_essential_frac: "Essential AA",
  protease_total_sites: "Protease Sites",
  trypsin_K_sites: "Trypsin K",
  trypsin_R_sites: "Trypsin R",
  cleavage_site_accessible_fraction: "Accessibility",
  disorder_propensity_mean: "Disorder",
  plddt_mean: "pLDDT",
  digestibility_score: "Digestibility",
};

const SKIP_KEYS = new Set(["red_flag", "green_flag"]);

interface Props {
  metrics: Record<string, number | boolean>;
}

export default function RadarChart_({ metrics }: Props) {
  const entries = Object.entries(metrics).filter(
    ([k, v]) => !SKIP_KEYS.has(k) && typeof v === "number"
  ) as [string, number][];

  if (entries.length < 3) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-brand-400">
        Not enough metrics for radar chart
      </div>
    );
  }

  const labels = entries.map(([k]) => DISPLAY_LABELS[k] ?? k);
  const values = entries.map(([, v]) => v);

  const data = {
    labels,
    datasets: [
      {
        data: values,
        fill: true,
        backgroundColor: "rgba(74, 144, 226, 0.12)",
        borderColor: "rgba(74, 144, 226, 0.8)",
        pointBackgroundColor: "rgba(74, 144, 226, 1)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(74, 144, 226, 1)",
        borderWidth: 2,
        pointRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(29, 53, 87, 0.9)",
        titleFont: { family: "Inter", size: 12 },
        bodyFont: { family: "Inter", size: 12 },
        padding: 10,
        cornerRadius: 8,
        callbacks: {
          label: (ctx: any) => {
            const val = ctx.raw;
            if (typeof val !== "number") return "";
            return `${ctx.label}: ${Number.isInteger(val) ? val : val.toFixed(4)}`;
          },
        },
      },
    },
    scales: {
      r: {
        beginAtZero: true,
        ticks: {
          display: false,
        },
        grid: {
          color: "rgba(74, 144, 226, 0.1)",
        },
        angleLines: {
          color: "rgba(74, 144, 226, 0.1)",
        },
        pointLabels: {
          font: { family: "Inter", size: 11, weight: "bold" as const },
          color: "#4a6b94",
        },
      },
    },
  };

  return (
    <div className="w-full h-full min-h-[260px]">
      <Radar data={data} options={options} />
    </div>
  );
}
