"""
Matcher Module
===============
TF-IDF vectorization and cosine similarity scoring
between a resume and a job description.
"""

from typing import Dict, List, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import preprocess, extract_skills, keyword_frequency
from skills_database import (
    ALL_SKILLS, TECHNICAL_SKILLS, SOFT_SKILLS, CERTIFICATIONS,
    categorize_skill,
)


def compute_similarity(resume_text: str, job_text: str) -> float:
    """
    Compute cosine similarity between the resume and job description
    using TF-IDF vectors.  Returns a float in [0, 1].
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0])


def analyze_match(
    resume_text: str,
    job_text: str,
) -> Dict:
    """
    Full matching pipeline.

    Returns a dict with:
      - match_score        : float  (0–100 %)
      - matched_skills     : set of skills found in both texts
      - missing_skills     : set of skills in job but not in resume
      - resume_only_skills : set of skills in resume but not in job
      - resume_skills      : set of all skills found in the resume
      - job_skills         : set of all skills found in the job desc
      - job_keywords       : top keywords from the job description
      - resume_keywords    : top keywords from the resume
      - skill_categories   : dict grouping matched/missing by category
    """
    # --- Skill extraction ---
    resume_skills = extract_skills(resume_text, ALL_SKILLS)
    job_skills = extract_skills(job_text, ALL_SKILLS)

    matched_skills = resume_skills & job_skills
    missing_skills = job_skills - resume_skills
    resume_only_skills = resume_skills - job_skills

    # --- TF-IDF cosine similarity ---
    tfidf_score = compute_similarity(resume_text, job_text)

    # --- Skill overlap ratio ---
    if job_skills:
        skill_ratio = len(matched_skills) / len(job_skills)
    else:
        skill_ratio = 0.0

    # Weighted blend: 60 % skill overlap + 40 % TF-IDF similarity
    match_score = round((0.6 * skill_ratio + 0.4 * tfidf_score) * 100, 1)

    # --- Keyword frequency ---
    job_keywords = keyword_frequency(job_text, top_n=15)
    resume_keywords = keyword_frequency(resume_text, top_n=15)

    # --- Categorise skills ---
    def _categorise(skills_set):
        cats = {"Technical Skills": [], "Soft Skills": [], "Certifications": [], "Other": []}
        for s in sorted(skills_set):
            cats[categorize_skill(s)].append(s)
        return {k: v for k, v in cats.items() if v}

    return {
        "match_score": match_score,
        "tfidf_score": round(tfidf_score * 100, 1),
        "skill_score": round(skill_ratio * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_only_skills": resume_only_skills,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "job_keywords": job_keywords,
        "resume_keywords": resume_keywords,
        "matched_categories": _categorise(matched_skills),
        "missing_categories": _categorise(missing_skills),
        "resume_only_categories": _categorise(resume_only_skills),
    }


def generate_suggestions(result: Dict) -> List[str]:
    """Generate actionable improvement suggestions based on the match result."""
    suggestions = []
    missing = result["missing_skills"]
    match_score = result["match_score"]

    if match_score >= 80:
        suggestions.append(
            "✅ Excellent match! Your resume aligns well with this job description."
        )
    elif match_score >= 60:
        suggestions.append(
            "👍 Good match overall. A few targeted additions could push you higher."
        )
    elif match_score >= 40:
        suggestions.append(
            "⚠️ Moderate match. Consider adding more relevant skills and experience keywords."
        )
    else:
        suggestions.append(
            "🔴 Low match. This role may require skills significantly different from what's on your resume."
        )

    # Missing technical skills
    missing_tech = [s for s in missing if s in TECHNICAL_SKILLS]
    if missing_tech:
        top_missing = ", ".join(sorted(missing_tech)[:5])
        suggestions.append(
            f"📌 Add these missing technical skills if you have them: **{top_missing}**"
        )

    # Missing soft skills
    missing_soft = [s for s in missing if s in SOFT_SKILLS]
    if missing_soft:
        top_soft = ", ".join(sorted(missing_soft)[:3])
        suggestions.append(
            f"💡 Mention these soft skills in your experience descriptions: **{top_soft}**"
        )

    # Missing certifications
    missing_certs = [s for s in missing if s in CERTIFICATIONS]
    if missing_certs:
        suggestions.append(
            f"🎓 The job mentions certifications you haven't listed: **{', '.join(sorted(missing_certs)[:3])}**"
        )

    # General advice
    if len(result["resume_skills"]) < 5:
        suggestions.append(
            "📝 Your resume has very few recognizable skills. "
            "Consider adding a dedicated 'Skills' section with relevant technologies."
        )

    if result["tfidf_score"] < 30:
        suggestions.append(
            "📄 The language in your resume differs a lot from the job description. "
            "Try mirroring key phrases and terminology from the posting."
        )

    return suggestions
