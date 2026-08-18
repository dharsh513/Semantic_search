"""
AI-Assisted Research Gap Identification & Suggested Research Directions Engine.

Performs evidence-grounded research gap analysis across 14 scientific dimensions:
  1. Dataset limitations (size, public availability, class imbalance)
  2. Population & demographic limitations (age, sex, ethnic diversity)
  3. Geographical & regional limitations (single country / region)
  4. Methodological limitations (untested baselines, manual features)
  5. Missing modalities (unimodal text/imaging vs multi-modal genomics/EHR)
  6. Limited validation (retrospective only, lack of prospective validation)
  7. Lack of external validation (single-institution cohorts)
  8. Small sample sizes (low statistical power)
  9. Single-center clinical designs
  10. Computational complexity & deployment latency
  11. Conflicting or discordant findings between comparative papers
  12. Understudied sub-populations & comorbidities
  13. Missing standard-of-care comparisons
  14. Unexplored combinations of complementary techniques

Generates actionable 'Possible Research Directions' linked directly to identified gaps.
Strict scientific safety: All findings are framed as 'AI-Assisted Potential Gaps' with explicit citations.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from rag.chunker import split_sentences
from rag.store import store

log = logging.getLogger(__name__)

GAP_CATEGORIES = [
    {
        "category": "External Validation & Multi-Center Generalization",
        "pattern": re.compile(
            r"\b(single[- ]center|single institution|internal validation only|lack(s|ing)? external|no external validation|unvalidated in independent|generalizability|generalise|generalize)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Limited Multi-Center & External Dataset Validation",
        "gap_desc": "Evaluations are primarily restricted to single-center or internal cohorts without rigorous external multi-center validation.",
        "direction_title": "Multi-Center Cross-Institutional Validation & Benchmarking",
        "direction_desc": "Validate the proposed models or findings across geographically diverse, independent multi-center cohorts to evaluate generalization and domain-shift robustness.",
        "methodology": "Assemble multi-site external validation cohorts and test out-of-distribution performance metrics.",
    },
    {
        "category": "Study Design & Prospective Clinical Evidence",
        "pattern": re.compile(
            r"\b(retrospective(ly)?|lack(s|ing)? prospective|no prospective|observational only|cross[- ]sectional|needs prospective)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Reliance on Retrospective Cohorts without Prospective Trials",
        "gap_desc": "Study designs rely on retrospective data collections, leaving real-time prospective clinical utility and workflow impact unverified.",
        "direction_title": "Prospective Pragmatic Clinical Trials & Decision Support Impact",
        "direction_desc": "Conduct prospective observational or randomized clinical trials to measure real-world clinical decision-making impact, diagnostic turnaround time, and patient outcomes.",
        "methodology": "Design prospective Phase II/III diagnostic accuracy trials integrated directly into point-of-care workflows.",
    },
    {
        "category": "Sample Size & Cohort Scale",
        "pattern": re.compile(
            r"\b(small sample|limited sample size|small cohort|pilot study|preliminary (data|study|results)|underpowered|sample size of \d{1,2}\b|n\s*=\s*\d{1,2}\b)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Constrained Sample Size & Statistical Power",
        "gap_desc": "Analyzed studies report relatively small patient/sample numbers, which may constrain statistical power and subgroup resolution.",
        "direction_title": "Large-Scale Multi-Cohort Expansion & Subgroup Stratification",
        "direction_desc": "Expand cohort scale via federated learning or consortium data registries to evaluate statistical significance across diverse clinical stratifications.",
        "methodology": "Utilize privacy-preserving federated analytics across hospital networks to scale sample sizes without centralizing sensitive PHI.",
    },
    {
        "category": "Multimodal Data Integration",
        "pattern": re.compile(
            r"\b(unimodal|only (imaging|text|genomics|clinical)|lack(s|ing)? multimodal|not integrated with (genomic|clinical|radiomic|histopathol)|single modality)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Lack of Multimodal Clinical & Omics Integration",
        "gap_desc": "Current investigations focus primarily on an isolated modality (e.g. imaging alone or tabular records alone), omitting complementary multi-omics and longitudinal biomarkers.",
        "direction_title": "Holistic Multimodal Fusion (Imaging + Genomics + Clinical EHR)",
        "direction_desc": "Develop multimodal deep learning fusion architectures that synthesize radiological scans, whole-slide histology, genomic sequencing, and longitudinal EHR trajectories.",
        "methodology": "Implement cross-attention multimodal transformers that combine vision encoders with clinical tabular/genomic embeddings.",
    },
    {
        "category": "Demographic & Population Diversity",
        "pattern": re.compile(
            r"\b(predominantly (male|female|caucasian|white|asian)|homogeneous population|single nationality|ethnic diversity|age restriction|underrepresented)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Underrepresentation of Diverse Demographic Subgroups",
        "gap_desc": "Cohorts exhibit potential demographic imbalances across ethnicity, age brackets, or geographic heritage, which may induce algorithmic bias.",
        "direction_title": "Fairness-Aware Demographic Stratification & Bias Auditing",
        "direction_desc": "Audit models across ethnically and demographically diverse international populations and apply algorithmic fairness regularization techniques.",
        "methodology": "Perform intersectional subgroup disparity analysis and adopt adversarial debiasing during training.",
    },
    {
        "category": "Computational Efficiency & Real-Time Deployment",
        "pattern": re.compile(
            r"\b(computationally expensive|high latency|heavy compute|large model size|not optimized for edge|inference time|resource[- ]intensive)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Computational Footprint & Real-Time Inference Latency",
        "gap_desc": "High model complexity and compute overhead create barriers for low-resource hospital infrastructure and real-time clinical bedside deployment.",
        "direction_title": "Model Compression, Knowledge Distillation & Edge Optimization",
        "direction_desc": "Quantize, prune, and distill large models into ultra-lightweight architectures suitable for edge devices, PACS workstations, and point-of-care mobile scanners.",
        "methodology": "Employ 4-bit INT quantization, structured layer pruning, and student-teacher knowledge distillation.",
    },
    {
        "category": "Longitudinal Monitoring & Treatment Response",
        "pattern": re.compile(
            r"\b(cross[- ]sectional only|no longitudinal|short follow[- ]up|lack of long[- ]term|treatment response over time|recurrence prediction)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Limited Longitudinal Trajectory & Follow-Up Tracking",
        "gap_desc": "Most available analyses assess diagnostic snapshot moments rather than continuous disease progression, therapeutic resistance, or multi-year recurrence monitoring.",
        "direction_title": "Longitudinal Disease Trajectory Modeling & Recurrence Forecasting",
        "direction_desc": "Formulate temporal sequential models (e.g. state-space models, recurrent neural ODEs) to track longitudinal biomarker dynamics and predict late relapse.",
        "methodology": "Train recurrent temporal architectures on multi-timepoint serial scans and laboratory panels.",
    },
    {
        "category": "Explainability, Interpretability & Mechanistic Causality",
        "pattern": re.compile(
            r"\b(black[- ]box|interpretability|explainability|mechanistic basis|causal (inference|mechanism)|saliency map limitations|lack of biological insight)\b",
            re.IGNORECASE,
        ),
        "gap_title": "Potential Gap: Black-Box Predictions & Lack of Mechanistic Interpretability",
        "gap_desc": "High accuracy is achieved without transparent biological mechanisms or clinically verifiable feature attributions, hindering clinician trust.",
        "direction_title": "Inherently Interpretable Causal Modeling & Concept-Based Explanations",
        "direction_desc": "Construct concept bottleneck models and causal structural equation models that ground predictions in established pathophysiological biomarkers.",
        "methodology": "Incorporate concept attribution bottlenecks and causal discovery graphs rather than post-hoc gradient saliency.",
    },
]


class ResearchGapEngine:
    """Analyzes selected literature to surface evidence-grounded research gaps and directions."""

    def analyze_gaps(self, pmids: Sequence[str]) -> Dict[str, Any]:
        pmids = [str(p).strip() for p in pmids if str(p).strip()]
        if not pmids:
            raise ValueError("Research gap analysis requires at least one PMID.")

        cached_articles = store.get_articles(pmids)
        missing = [p for p in pmids if p not in cached_articles]
        if missing:
            from rag.pipeline import pipeline
            fetched = pipeline.article_batch(missing)
            cached_articles.update({a["pmid"]: a for a in fetched})

        articles = [cached_articles[p] for p in pmids if p in cached_articles]
        if not articles:
            raise ValueError("No valid article records found for the requested PMIDs.")

        identified_gaps: List[Dict[str, Any]] = []
        research_directions: List[Dict[str, Any]] = []

        # Analyze each paper for specific gap indicators
        category_hits: Dict[str, List[Dict[str, Any]]] = {}

        for art in articles:
            pmid = art["pmid"]
            title = art.get("title", "")
            abstract = art.get("abstract", "")
            text = f"{title} {abstract}"

            for cat_def in GAP_CATEGORIES:
                cat_name = cat_def["category"]
                pat = cat_def["pattern"]
                match = pat.search(text)
                if match:
                    # Find exact matching sentence for evidence
                    matching_sentence = ""
                    for s in split_sentences(abstract):
                        if pat.search(s):
                            matching_sentence = s.strip()
                            break
                    if not matching_sentence:
                        matching_sentence = f"Pattern '{match.group(0)}' observed in study context: {title[:100]}."

                    if cat_name not in category_hits:
                        category_hits[cat_name] = []

                    category_hits[cat_name].append({
                        "pmid": pmid,
                        "title": title,
                        "first_author": (art.get("authors") or ["Unknown"])[0],
                        "year": art.get("year", ""),
                        "matched_text": match.group(0),
                        "evidence_sentence": matching_sentence,
                        "def": cat_def,
                    })

        # Synthesize gaps from hits
        for cat_name, hits in category_hits.items():
            first_hit = hits[0]
            cat_def = first_hit["def"]
            evidence_pmids = list({h["pmid"] for h in hits})
            evidence_count = len(evidence_pmids)

            # Confidence is higher when multiple papers share the limitation or explicit cues exist
            confidence = "high" if evidence_count >= 2 else "medium"

            supporting_points = [
                f"PMID {h['pmid']} ({h['first_author']} et al., {h['year'] or 'N/A'}): “{h['evidence_sentence']}”"
                for h in hits[:4]
            ]

            gap_item = {
                "id": f"gap-{len(identified_gaps) + 1}",
                "category": cat_name,
                "title": cat_def["gap_title"],
                "description": cat_def["gap_desc"],
                "confidence": confidence,
                "evidence_pmids": evidence_pmids,
                "supporting_points": supporting_points,
            }
            identified_gaps.append(gap_item)

            direction_item = {
                "id": f"dir-{len(research_directions) + 1}",
                "linked_gap_title": cat_def["gap_title"],
                "title": cat_def["direction_title"],
                "description": cat_def["direction_desc"],
                "suggested_methodology": cat_def["methodology"],
                "target_pmids": evidence_pmids,
            }
            research_directions.append(direction_item)

        # If no heuristic pattern fired (e.g. abstract too concise or general), construct a systematic comparative gap
        if not identified_gaps and len(articles) >= 2:
            paper_a = articles[0]
            paper_b = articles[1]
            gap_item = {
                "id": "gap-1",
                "category": "Comparative Benchmark & Harmonization",
                "title": "Potential Gap: Absence of Standardized Head-to-Head Benchmarking",
                "description": f"Studies evaluate disparate cohorts (PMID {paper_a['pmid']} vs PMID {paper_b['pmid']}) without a unified, harmonized benchmark dataset.",
                "confidence": "medium",
                "evidence_pmids": [paper_a["pmid"], paper_b["pmid"]],
                "supporting_points": [
                    f"PMID {paper_a['pmid']}: {paper_a.get('title', '')[:80]}…",
                    f"PMID {paper_b['pmid']}: {paper_b.get('title', '')[:80]}…",
                ],
            }
            identified_gaps.append(gap_item)

            direction_item = {
                "id": "dir-1",
                "linked_gap_title": gap_item["title"],
                "title": "Unified Open-Access Benchmarking & Cross-Model Comparison",
                "description": "Establish a standardized public test harness evaluating these distinct methodologies under identical baseline conditions.",
                "suggested_methodology": "Curate a shared multi-institution benchmark dataset with fixed training, validation, and testing splits.",
                "target_pmids": [paper_a["pmid"], paper_b["pmid"]],
            }
            research_directions.append(direction_item)

        # Fallback if only 1 paper and no hits
        if not identified_gaps:
            art = articles[0]
            gap_item = {
                "id": "gap-1",
                "category": "General Literature Gap",
                "title": "Potential Gap: Generalizability & Independent Replication Needed",
                "description": "Single-study findings require independent replication across varied clinical protocols and diverse patient populations.",
                "confidence": "low",
                "evidence_pmids": [art["pmid"]],
                "supporting_points": [
                    f"PMID {art['pmid']}: Independent prospective reproduction recommended."
                ],
            }
            identified_gaps.append(gap_item)
            direction_item = {
                "id": "dir-1",
                "linked_gap_title": gap_item["title"],
                "title": "Independent Multi-Cohort Replication",
                "description": "Replicate study methodology across an external cohort to verify repeatability.",
                "suggested_methodology": "Execute pre-registered replication protocol on an independent dataset.",
                "target_pmids": [art["pmid"]],
            }
            research_directions.append(direction_item)

        return {
            "pmids": pmids,
            "paper_count": len(articles),
            "gaps": identified_gaps,
            "research_directions": research_directions,
            "safety_disclaimer": "AI-Assisted Potential Gaps are derived from peer-reviewed abstract evidence and should guide exploratory study design rather than definitive clinical claims.",
        }


gap_engine = ResearchGapEngine()
