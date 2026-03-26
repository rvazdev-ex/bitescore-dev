import { useRef, useEffect, useId } from "react";

declare global {
  interface Window {
    $3Dmol: any;
  }
}

interface Props {
  pdbText: string;
  seqId: string;
}

export default function StructureViewer({ pdbText, seqId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const uniqueId = useId();

  useEffect(() => {
    if (!containerRef.current || !pdbText) return;

    const render = () => {
      if (!window.$3Dmol || !containerRef.current) return;

      // Clean up previous viewer
      if (viewerRef.current) {
        try {
          viewerRef.current.clear();
        } catch {}
      }

      try {
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          backgroundColor: "white",
        });
        viewer.addModel(pdbText, "pdb");
        viewer.setStyle({}, { cartoon: { color: "spectrum" } });
        viewer.zoomTo();
        viewer.render();
        viewerRef.current = viewer;
      } catch (err) {
        console.error("Failed to render structure:", err);
      }
    };

    if (window.$3Dmol) {
      render();
    } else {
      // Wait for 3Dmol to load
      const check = setInterval(() => {
        if (window.$3Dmol) {
          clearInterval(check);
          render();
        }
      }, 200);
      return () => clearInterval(check);
    }

    return () => {
      if (viewerRef.current) {
        try {
          viewerRef.current.clear();
        } catch {}
        viewerRef.current = null;
      }
    };
  }, [pdbText, uniqueId]);

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="w-full h-[420px] rounded-2xl border border-brand-100 bg-gradient-to-br from-brand-50/30 to-white overflow-hidden"
        style={{ position: "relative" }}
      />
      <div className="absolute bottom-3 right-3">
        <span className="text-[10px] text-brand-300 bg-white/80 backdrop-blur px-2 py-1 rounded-full">
          Powered by 3Dmol.js
        </span>
      </div>
    </div>
  );
}
