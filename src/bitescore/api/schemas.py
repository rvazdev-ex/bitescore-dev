from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InputType(str, Enum):
    proteome = "proteome"
    genome = "genome"
    metagenome = "metagenome"
    sequences = "sequences"


class OrganismType(str, Enum):
    prok = "prok"
    euk = "euk"


class AnalysisRequest(BaseModel):
    input_type: InputType = InputType.proteome
    organism: Optional[OrganismType] = None
    organisms: List[str] = Field(default_factory=list)
    sequences: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class AnalysisStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ProgressUpdate(BaseModel):
    job_id: str
    status: AnalysisStatus
    percent: float = 0.0
    description: str = ""


class SequenceSummary(BaseModel):
    id: str
    rank: int
    length: Optional[int] = None
    digestibility_score: Optional[float] = None
    aa_essential_frac: Optional[float] = None


class SequenceDetail(BaseModel):
    id: str
    rank: int
    sequence: Optional[str] = None
    digestibility_score: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    features: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
    structure_available: bool = False
    blastp_url: Optional[str] = None


class AnalysisResult(BaseModel):
    job_id: str
    status: AnalysisStatus
    sequence_count: int = 0
    input_type: str = ""
    organisms: List[str] = Field(default_factory=list)
    ranked: List[SequenceSummary] = Field(default_factory=list)
    download_url: Optional[str] = None


class ExampleInfo(BaseModel):
    name: str
    description: str
    sequences: Optional[str] = None
    file_path: Optional[str] = None
    input_type: str = "proteome"
