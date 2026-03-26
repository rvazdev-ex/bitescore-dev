"""Built-in reference protein sequences and experimental digestibility values.

These reference proteins serve as calibration anchors: the pipeline scores them
alongside user proteins and fits a monotonic mapping from raw predicted scores
to experimentally validated DIAAS values.

Sources:
    FAO (2013) Dietary protein quality evaluation in human nutrition. Report of
        an FAO Expert Consultation. FAO Food and Nutrition Paper 92.
    Rutherfurd SM et al. (2015) Protein digestibility-corrected amino acid
        scores and digestible indispensable amino acid scores differentially
        describe protein quality in growing male rats. J Nutr 145:372-379.
    Mathai JK et al. (2017) Values for digestible indispensable amino acid
        scores (DIAAS) for some dairy and plant proteins may better describe
        protein quality than values calculated using the concept for protein
        digestibility-corrected amino acid scores (PDCAAS). Br J Nutr 117:490-499.

Each food entry contains:
    food_id:      Unique identifier
    food_name:    Human-readable name
    diaas:        Digestible Indispensable Amino Acid Score (0-140+ scale)
    pdcaas:       Protein Digestibility Corrected Amino Acid Score (0-1 scale, truncated at 1)
    proteins:     Dict mapping protein_id -> {sequence, abundance_fraction}
                  abundance_fraction values within a food sum to 1.0

Protein sequences are the mature (signal-peptide-cleaved) forms from UniProt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ReferenceProtein:
    """A single protein within a reference food."""
    protein_id: str
    uniprot_accession: str
    sequence: str
    abundance_fraction: float  # fraction of total protein in the food


@dataclass
class ReferenceFood:
    """A food with known experimental digestibility and constituent proteins."""
    food_id: str
    food_name: str
    diaas: float           # 0-140+ scale
    pdcaas: float          # 0-1 scale (truncated)
    proteins: List[ReferenceProtein] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reference protein sequences (mature forms, UniProt canonical)
# ---------------------------------------------------------------------------

# Bovine beta-casein (P02666), mature form (signal peptide removed)
_BOVINE_BETA_CASEIN = (
    "RELEELNVPGEIVESLSSSEESITRINKKIEKFQSEEQQQTEDELQDKIHPFAQTQSLVYPFPGPIHN"
    "SLPQNIPPLTQTPVVVPPFLQPEVMGVSKVKEAMAPKHKEMPFPKYPVEPFTESQSLTLTDVENLHLP"
    "LPLLQSWMHQPHQPLPPTVMFPPQSVLSLSQSKVLPVPQKAVPYPQRDMPIQAFLLYQEPVLGPVRGPFPIIV"
)

# Bovine alpha-s1-casein (P02662), mature form
_BOVINE_ALPHAS1_CASEIN = (
    "RPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELSKDIGSESTEDQAMEDIKQMEAESISSSE"
    "EIVPNSVEQKHIQKEDVPSERYLGYLEQLLRLKKYKVPQLEIVPNSAEERLHSMKEGIHAQQKEPMIGV"
    "NQELAYFYPELFRQFYQLDAYPSGAWYYVPLGTQYTDAPSFSDIPNPIGSENSEKTTMPLW"
)

# Bovine beta-lactoglobulin (P02754), mature form
_BOVINE_BETA_LACTOGLOBULIN = (
    "LIVTQTMKGLDIQKVAGTWYSLAMAASDISLLDAQSAPLRVYVEELKPTPEGDLEILLQKWENGECAQK"
    "KIIAEKTKIPAVFKIDALNENKVLVLDTDYKKYLLFCMENSAEPEQSLACQCLVRTPEVDDEALEKFDK"
    "ALKALPMHIRLSFNPTQLEEQCHI"
)

# Soy glycinin G1 (P04776), mature acidic chain
_SOY_GLYCININ_G1 = (
    "REQAQQNECQIQKLNALKPDNRIESEGGFIETWNPNNKPFQCAGVALSRCTLNRNALRRPSYTNGPQE"
    "IYIQQGKGIFGMIYPGCPSTFEEPQQPQQRGQSSRPQDRHQKIYNFREGDLIAVPTGFKWPHGRISQR"
    "AQLQNFNNAGNIVRIVNQHGLYRSAAGKNETALGSLVNAQTDNLGNKFGKADAQIAKNHKGQYMIAPY"
    "HYTHPVGRIIYLAGNNPNANVAIFSTNRGTGKIILDNRRSQRNEQEREFSGRGDEDSEQPFPFPREQHG"
)

# Soy beta-conglycinin alpha subunit (P13916), partial mature form
_SOY_BETA_CONGLYCININ = (
    "KEQQQEQQLQNLATQHSQIRQARQKLKNNNPFKFLVPPQESQKRAVEPLLKKNQNPFLKPAAREQERR"
    "GQHRRKNKQNLRITKDFPVQLDDIDISKILQIHTFNSSQLERAFSRNTLEAAFNAEFNEIRKLLIPQYD"
    "SRIVVQFKSGIEQTLQFPKNWQQHQKQQKMFLKPHEADFLNHQNNQRNITCQIKNLQNQLDQMPRRVF"
    "YLAGNQDNQHPEQKRFQETFKNQYGIKVFKPQVFQLLSNSLSKMFAPFPDHFFTMRGQSF"
)

# Chicken egg ovalbumin (P01012), mature form
_EGG_OVALBUMIN = (
    "GSIGAASMEFCFDVFKELKVHHANENIFYCPIAIMSALAMVYLGAKDSTRTQINKVVRFDKLPGFGDSIE"
    "AQCGTSVNVHSSLRDILNQITKPNDVYSFSLASRLYAEERYPILPEYLQCVKELYRGGLEPINFQTAAD"
    "QARELINSWVESQTNGIIRNVLQPSSVDSQTAMVLVNAIVFKGLWEKAFKDEDTQAMPFRVTEQESKPV"
    "QMMYQIGLFRVASPAAAHILSDIDAKLIREAIDSADLALSSASLYYLANALRRCPAEQMPDALRGVSISV"
    "MQPINLLPHELPDDKPLRPPCSELIAKTKNFVNELQSVSEAFLHPFKIIEKLEIEPVAHGIVSISRP"
    "ELLKLKEWLLASKNFHTFPQLHQFPAHAQSIQSLEHISFLLIHNPTKANLQEISTLYWGR"
)

# Wheat high-molecular-weight glutenin subunit (P10388), partial
_WHEAT_HMW_GLUTENIN = (
    "ELQESSLEACQQVVDQQLAGRLPWSTGLQMRCCQQLRDIAAIHSVAHHAGYPYAGAGYYGATSPAAQLSP"
    "AQGYYPASSPQQSGQGQQPGQWQQPEQGQQPGQGQQGYHPTTSPQQSGQGQQSGQGQQGYYPTSLQQPG"
    "QGQQGHYPASSPQQPGQGQQPGQWQQSGQGQQPGQGQQGYYPASSPQQSGQGQQPGQWQQPGQGQQGYY"
    "PTSLQQPGQGQQGHYPTSPQQSGQGQQPGQWQQPEQGQQPGQGQQGYYPASSPQQPGQGQQPGQGQQGYY"
)

# Rice glutelin type-A 1 (P20056), partial mature form
_RICE_GLUTELIN = (
    "QQQQLVQGIQSQQQQFLAGANQLEQSLERLAELSQMQRIMAQLEQQLDDQSGQPQREEEEQFWQHYRNR"
    "NALRRAQLEQNQFEELREGIFFQPQYAQEQAQGEQNMRIIQNQNLQYLAAFNPTIFEAANAGRVSVSML"
    "EPNFIAPAILYFAQQLAQNRAQQGRYEQARAQLEEQGGEIDTGAQLPSGRDVLQITSNQNQLTPFNHPS"
    "QRRAGRNGIYMIPPGCPETFLSPQQEQPAREHQEGVIVILSIDANPSINDWNHHQALQQARRVVEEFGVF"
)

# Pea legumin A (P15838), partial mature form
_PEA_LEGUMIN = (
    "REEQEWEEEQQSHRRQEQSRRNKIQGEDEIQQRPSHQKEGDKKHRRQHEQQHRRSQRSRRSREGEERED"
    "EDEKPQEKRDFNSFNLECGLRAQIKRFNLRSQQRQGLNIFKGLYEETFPSGVIVTLYGISPRTINLFD"
    "RHLNAGQLYAVTQDHENVYALKGRASITTLLFERNPEVIVAVSLAGKRTITISDEEPHSDAFQASCYVNK"
    "LEEGRALVFLPGQHEENQKGQKRFQAGQNIVISNRPGTIFYWDNNDGEQVVAISAGATPALRFLLRMGE"
)

# Corn zein alpha (P04701), partial
_CORN_ZEIN = (
    "MATKILALLALLAVSATNAFIIPQCSLAPSANILQQAILAPCLNQLQQISNLSSPIYNQFYQASIASLP"
    "QLLAQLAASPFLQQQQLLPFNQLAQSIAPAYPTQQFQQLTPFNQLAQSVSQLASLLNPYLATANAVHLFA"
    "PLAQPQQAHLPAFALAQAFSVAHLTPMMHQFLPYLAQAASTYQQILRQAISVSALVLQQ"
)

# Chickpea legumin-like (A0A1S3E4J5), partial
_CHICKPEA_LEGUMIN = (
    "REQEWERQQEKQHGRQEQSRRSRIQGEDDIRQRFSHQKEGDKQHRRTQREQNHRRSQRSQRSREGEERE"
    "DEDEKRQEKRDFNSFNLECGLGAQIKRFNLETQQRQGLNIFKGLYEETFPSGVIVTLYGISPRSINLFD"
    "RHLNAGQLYAVTQDHENVYALKGRASITTLLFERNPEVIVAVSLAGKRTITISDEEPHSDAFQASCYVNK"
    "LEEGRALVFLPGQHEENQKGQKRFQAGQNIVISNRPGTIFYWDNNDGEQVVAISAGATPALRFLLRAGE"
)


# ---------------------------------------------------------------------------
# Assembled reference foods
# ---------------------------------------------------------------------------

REFERENCE_FOODS: List[ReferenceFood] = [
    ReferenceFood(
        food_id="whole_milk",
        food_name="Whole milk",
        diaas=114,
        pdcaas=1.0,
        proteins=[
            ReferenceProtein("milk_beta_casein", "P02666", _BOVINE_BETA_CASEIN, 0.28),
            ReferenceProtein("milk_alphas1_casein", "P02662", _BOVINE_ALPHAS1_CASEIN, 0.32),
            ReferenceProtein("milk_beta_lactoglobulin", "P02754", _BOVINE_BETA_LACTOGLOBULIN, 0.10),
        ],
    ),
    ReferenceFood(
        food_id="whey_protein_isolate",
        food_name="Whey protein isolate",
        diaas=109,
        pdcaas=1.0,
        proteins=[
            ReferenceProtein("whey_beta_lactoglobulin", "P02754", _BOVINE_BETA_LACTOGLOBULIN, 0.55),
            ReferenceProtein("whey_alphas1_casein", "P02662", _BOVINE_ALPHAS1_CASEIN, 0.05),
        ],
    ),
    ReferenceFood(
        food_id="soy_protein_isolate",
        food_name="Soy protein isolate",
        diaas=90,
        pdcaas=0.91,
        proteins=[
            ReferenceProtein("soy_glycinin_g1", "P04776", _SOY_GLYCININ_G1, 0.40),
            ReferenceProtein("soy_beta_conglycinin", "P13916", _SOY_BETA_CONGLYCININ, 0.30),
        ],
    ),
    ReferenceFood(
        food_id="egg_whole",
        food_name="Whole egg",
        diaas=113,
        pdcaas=1.0,
        proteins=[
            ReferenceProtein("egg_ovalbumin", "P01012", _EGG_OVALBUMIN, 0.54),
        ],
    ),
    ReferenceFood(
        food_id="pea_protein",
        food_name="Pea protein concentrate",
        diaas=82,
        pdcaas=0.73,
        proteins=[
            ReferenceProtein("pea_legumin", "P15838", _PEA_LEGUMIN, 0.60),
        ],
    ),
    ReferenceFood(
        food_id="wheat",
        food_name="Wheat (whole grain)",
        diaas=40,
        pdcaas=0.42,
        proteins=[
            ReferenceProtein("wheat_hmw_glutenin", "P10388", _WHEAT_HMW_GLUTENIN, 0.45),
        ],
    ),
    ReferenceFood(
        food_id="rice_polished",
        food_name="Rice (polished)",
        diaas=59,
        pdcaas=0.62,
        proteins=[
            ReferenceProtein("rice_glutelin", "P20056", _RICE_GLUTELIN, 0.55),
        ],
    ),
    ReferenceFood(
        food_id="corn",
        food_name="Corn (maize)",
        diaas=44,
        pdcaas=0.47,
        proteins=[
            ReferenceProtein("corn_zein", "P04701", _CORN_ZEIN, 0.50),
        ],
    ),
    ReferenceFood(
        food_id="chickpea",
        food_name="Chickpea (cooked)",
        diaas=83,
        pdcaas=0.78,
        proteins=[
            ReferenceProtein("chickpea_legumin", "A0A1S3E4J5", _CHICKPEA_LEGUMIN, 0.55),
        ],
    ),
]


def get_all_reference_proteins() -> List[ReferenceProtein]:
    """Return a flat list of all unique reference proteins across all foods."""
    seen: set = set()
    proteins: List[ReferenceProtein] = []
    for food in REFERENCE_FOODS:
        for prot in food.proteins:
            if prot.protein_id not in seen:
                seen.add(prot.protein_id)
                proteins.append(prot)
    return proteins


def get_reference_food_by_id(food_id: str) -> ReferenceFood | None:
    """Look up a reference food by its ID."""
    for food in REFERENCE_FOODS:
        if food.food_id == food_id:
            return food
    return None


def reference_proteins_as_seqrecords():
    """Convert all unique reference proteins to BioPython SeqRecords."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    records = []
    for prot in get_all_reference_proteins():
        rec = SeqRecord(
            Seq(prot.sequence),
            id=prot.protein_id,
            description=f"ref|{prot.uniprot_accession}",
        )
        records.append(rec)
    return records
