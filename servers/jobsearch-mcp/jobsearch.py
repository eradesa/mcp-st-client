import os
import re
import json
import tempfile
from typing import Optional
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Job Search")

# ---------------------------------------------------------------------------
# Configuration — edit these or override via environment variables
# ---------------------------------------------------------------------------

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY", "")

# Default country for Adzuna searches (2-letter code)
DEFAULT_COUNTRY = os.getenv("JOBSEARCH_DEFAULT_COUNTRY", "us")

# Supported Adzuna countries
ADZUNA_COUNTRIES = {
    "gb": "https://api.adzuna.com/v1/api/jobs/gb/search/1",
    "us": "https://api.adzuna.com/v1/api/jobs/us/search/1",
    "in": "https://api.adzuna.com/v1/api/jobs/in/search/1",
    "au": "https://api.adzuna.com/v1/api/jobs/au/search/1",
    "nz": "https://api.adzuna.com/v1/api/jobs/nz/search/1",
    "ca": "https://api.adzuna.com/v1/api/jobs/ca/search/1",
    "za": "https://api.adzuna.com/v1/api/jobs/za/search/1",
    "fr": "https://api.adzuna.com/v1/api/jobs/fr/search/1",
    "de": "https://api.adzuna.com/v1/api/jobs/de/search/1",
    "pl": "https://api.adzuna.com/v1/api/jobs/pl/search/1",
    "br": "https://api.adzuna.com/v1/api/jobs/br/search/1",
    "at": "https://api.adzuna.com/v1/api/jobs/at/search/1",
}

# ---------------------------------------------------------------------------
# Skills dictionary for CV matching — edit to add/remove skills
# ---------------------------------------------------------------------------

SKILLS = {
    # Programming Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Haskell", "Perl", "R",
    "MATLAB", "Dart", "Lua", "Shell Scripting", "Bash",

    # Frontend Frameworks & Libraries
    "React", "Vue.js", "Angular", "Svelte", "Next.js", "Nuxt.js", "jQuery",
    "HTML", "CSS", "SASS", "Tailwind CSS", "Bootstrap", "Redux", "GraphQL",

    # Backend Frameworks
    "Django", "Flask", "FastAPI", "Express.js", "Node.js", "Spring Boot",
    "ASP.NET", "Laravel", "Ruby on Rails", "Gin", "Fiber",

    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "Prometheus", "Grafana",
    "Linux", "Nginx", "Apache",

    # Databases & Big Data
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "DynamoDB", "BigQuery", "Snowflake", "Apache Spark", "Kafka", "Airflow",

    # Data Science & ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow",
    "PyTorch", "scikit-learn", "Pandas", "NumPy", "LangChain", "LLM", "RAG",

    # Design & Creative
    "Figma", "Sketch", "Adobe XD", "Photoshop", "Illustrator", "InDesign",
    "Premiere Pro", "After Effects", "Blender", "Canva",

    # Marketing & SEO
    "SEO", "SEM", "Google Analytics", "Google Ads", "Meta Ads", "Content Marketing",
    "Email Marketing", "Social Media Marketing", "CRM", "HubSpot", "Salesforce",

    # Business & PM
    "Agile", "Scrum", "Jira", "Confluence", "Notion", "Slack", "Asana",
    "Trello", "Microsoft Project", "Tableau", "Power BI",

    # Certifications
    "PMP", "AWS Certified", "CISSP", "CEH", "CompTIA", "ITIL",
    "Google Cloud Certified", "Azure Certified", "Scrum Master", "TOGAF",

    # Soft Skills
    "Leadership", "Communication", "Problem Solving", "Critical Thinking",
    "Team Management", "Project Management", "Stakeholder Management",
    "Mentoring", "Cross-functional Collaboration", "Agile Methodologies",

    # Industry & Domain
    "FinTech", "HealthTech", "EdTech", "E-commerce", "SaaS", "IoT",
    "Blockchain", "Cybersecurity", "Game Development", "Embedded Systems",

    # Emerging & AI
    "Generative AI", "GPT", "Computer Vision", "Reinforcement Learning",
    "MLOps", "Data Engineering", "Prompt Engineering", "AI Safety",
}

# ---------------------------------------------------------------------------
# Query building helpers
# ---------------------------------------------------------------------------

JOB_BOARDS = {
    "remote": "site:linkedin.com OR site:indeed.com OR site:glassdoor.com OR site:ziprecruiter.com OR site:monster.com",
    "freelance": "site:upwork.com OR site:freelancer.com OR site:toptal.com OR site:linkedin.com OR site:peopleperhour.com",
    "both": "site:linkedin.com OR site:indeed.com OR site:glassdoor.com OR site:upwork.com OR site:freelancer.com",
}

LOCAL_JOB_BOARDS = {
    "remote": "site:linkedin.com OR site:indeed.com OR site:glassdoor.com OR site:monster.com",
    "freelance": "site:linkedin.com OR site:indeed.com OR site:glassdoor.com OR site:upwork.com",
    "both": "site:linkedin.com OR site:indeed.com OR site:glassdoor.com OR site:monster.com OR site:upwork.com",
}

REMOTE_KEYWORDS = 'remote work from home remote job'
PERMANENT_KEYWORDS = 'permanent full-time permanent position'
FREELANCE_KEYWORDS = 'freelance contract project based gig consulting'


def _build_web_query(query: str, location: Optional[str], remote: bool,
                     employment_type: str) -> str:
    terms = [query]
    if remote:
        terms.append(REMOTE_KEYWORDS)
    if location:
        terms.append(location)
    if employment_type == "permanent":
        terms.append(PERMANENT_KEYWORDS)
    elif employment_type == "freelance":
        terms.append(FREELANCE_KEYWORDS)
    return " ".join(terms)


# ---------------------------------------------------------------------------
# Backend: DuckDuckGo web search
# ---------------------------------------------------------------------------

def _search_web(query: str, location: Optional[str], remote: bool,
                employment_type: str, limit: int) -> str:
    import requests
    from urllib.parse import unquote, parse_qs, urlparse
    from bs4 import BeautifulSoup

    search_query = _build_web_query(query, location, remote, employment_type)

    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": search_query},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Web search error: {e}"

    def extract_url(ddg_href: str) -> str:
        if "uddg=" in ddg_href:
            parsed = urlparse(ddg_href)
            qs = parse_qs(parsed.query)
            return unquote(qs.get("uddg", [""])[0])
        return ddg_href

    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    for result in soup.select(".result"):
        title_el = result.select_one(".result__title a")
        snippet_el = result.select_one(".result__snippet")
        if not title_el:
            continue
        href = extract_url(title_el.get("href", ""))
        title = title_el.get_text(strip=True)
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        if not title and not href:
            continue
        results.append({"title": title, "body": snippet, "href": href})
        if len(results) >= limit:
            break

    if not results:
        return "No job listings found."

    lines = [f"Found {len(results)} job listings:\n"]
    for i, job in enumerate(results, 1):
        lines.append(f"{i}. {job['title']}")
        lines.append(f"   Source: {job['href']}")
        snippet = job['body'][:200] + ("..." if len(job['body']) > 200 else "")
        lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend: Adzuna API
# ---------------------------------------------------------------------------

def _search_adzuna(query: str, location: Optional[str], remote: bool,
                   employment_type: str, limit: int, country: str) -> str:
    if not ADZUNA_APP_ID or not ADZUNA_API_KEY:
        return _search_web(query, location, remote, employment_type, limit)

    country = country.lower()
    base_url = ADZUNA_COUNTRIES.get(country)
    if not base_url:
        return (f"Unsupported country '{country}'. Supported: "
                f"{', '.join(sorted(ADZUNA_COUNTRIES.keys()))}. "
                f"Falling back to web search.\n\n"
                f"{_search_web(query, location, remote, employment_type, limit)}")

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "what": query,
        "results_per_page": min(limit, 50),
        "content_type": "application/json",
    }

    if location:
        params["where"] = location
    if remote:
        params["remote"] = 1
    if employment_type == "permanent":
        params["contract_type"] = "permanent"
        params["full_time"] = 1
    elif employment_type == "freelance":
        params["contract_type"] = "contract"

    import requests
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Adzuna API error: {e}. Falling back to web search.\n\n{_search_web(query, location, remote, employment_type, limit)}"

    jobs = data.get("results", [])
    if not jobs:
        return "No job listings found via Adzuna. Falling back to web search.\n\n" + \
               _search_web(query, location, remote, employment_type, limit)

    lines = [f"Found {len(jobs)} job listings via Adzuna:\n"]
    for i, job in enumerate(jobs[:limit], 1):
        title = job.get("title", "Unknown Title")
        company = job.get("company", {}).get("display_name", "Unknown Company")
        loc = job.get("location", {}).get("display_name", "Unknown Location")
        salary_min = job.get("salary_min")
        salary_max = job.get("salary_max")
        salary_str = ""
        if salary_min and salary_max:
            salary_str = f" ${salary_min:,.0f} - ${salary_max:,.0f}"
        elif salary_min:
            salary_str = f" From ${salary_min:,.0f}"

        desc = job.get("description", "")[:200].replace("\n", " ") + "..."
        url = job.get("redirect_url", "")

        lines.append(f"{i}. {title} at {company}")
        lines.append(f"   Location: {loc}{salary_str}")
        if employment_type != "both":
            lines.append(f"   Type: {employment_type}")
        lines.append(f"   {desc}")
        lines.append(f"   Apply: {url}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CV text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _extract_text_from_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return _extract_text_from_pdf(path)
    elif path_lower.endswith(".docx"):
        return _extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}. Only PDF and DOCX are supported.")


# ---------------------------------------------------------------------------
# CV parsing — lightweight extraction of job titles, skills, experience, locations
# ---------------------------------------------------------------------------

JOB_TITLE_PATTERNS = [
    r"(?:^|\n)\s*([A-Z][A-Za-z\s]{2,50}(?:Engineer|Developer|Manager|Analyst|"
    r"Designer|Architect|Consultant|Specialist|Coordinator|Director|"
    r"Lead|Head|Officer|Administrator|Representative|Associate|Intern))",
    r"(?:^|\n)\s*(Software Engineer|Data Scientist|Product Manager|DevOps Engineer|"
    r"Machine Learning Engineer|Full Stack Developer|Frontend Developer|"
    r"Backend Developer|UX Designer|UI Designer|Project Manager|"
    r"Business Analyst|QA Engineer|Solutions Architect|Scrum Master)",
]

LOCATION_PATTERNS = [
    r"(?:📍|Location|Based in|Located in)[:\s]*([A-Za-z\s,]+?)(?:\n|$)",
    r"(?:^|\n)\s*([A-Z][a-z]+(?:\s*,\s*[A-Z]{2})?)\s*$",
]

EXPERIENCE_PATTERN = r"(\d+)\+?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)"


def _parse_cv(text: str) -> dict:
    extracted = {
        "titles": [],
        "skills": [],
        "experience_years": None,
        "locations": [],
        "education": [],
    }

    for pattern in JOB_TITLE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for m in matches:
            m = m.strip()
            if m and len(m) > 3 and m not in extracted["titles"]:
                extracted["titles"].append(m)

    exp_match = re.search(EXPERIENCE_PATTERN, text, re.IGNORECASE)
    if exp_match:
        extracted["experience_years"] = int(exp_match.group(1))

    for pattern in LOCATION_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        for m in matches:
            m = m.strip()
            if m and len(m) > 2 and m not in extracted["locations"]:
                extracted["locations"].append(m)

    words = set(re.findall(r"[A-Za-z#+]+(?:\.[A-Za-z]+)*", text))
    extracted["skills"] = sorted(words & SKILLS)

    edu_keywords = {"bachelor", "master", "phd", "b.s.", "m.s.", "ph.d.",
                    "bachelor's", "master's", "b.tech", "m.tech", "mba",
                    "degree", "diploma", "certification", "ba", "ma", "bsc", "msc"}
    for line in text.split("\n"):
        lower = line.lower()
        if any(kw in lower for kw in edu_keywords):
            extracted["education"].append(line.strip())

    return extracted


# ---------------------------------------------------------------------------
# Job relevance ranking
# ---------------------------------------------------------------------------

def _rank_jobs(jobs: list, cv_skills: set, cv_titles: list) -> list:
    scored = []
    normalized_titles = set(t.lower() for t in cv_titles)

    for job in jobs:
        score = 0
        job_text = (job.get("title", "") + " " + job.get("body", "")).lower()

        for skill in cv_skills:
            if skill.lower() in job_text:
                score += 2

        for title in normalized_titles:
            if title in job_text:
                score += 3

        scored.append((score, job))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [job for score, job in scored]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_jobs(
    query: str,
    location: Optional[str] = None,
    remote: bool = True,
    employment_type: str = "both",
    limit: int = 10,
    source: str = "web",
    salary_min: Optional[int] = None,
    country: str = DEFAULT_COUNTRY,
) -> str:
    """Search for job listings by query, location, employment type, and remote preference.

    Use this to find jobs, freelance projects, contracts, or permanent positions.
    For category-based search, include the category in the query (e.g., "Python developer freelance").

    Args:
        query: Job title, keywords, skills, or free-form search (e.g., "React developer", "data scientist freelance", "Python AWS remote")
        location: Optional city, state, or country to search in (e.g., "San Francisco", "London", "New York")
        remote: Whether to search for remote/work-from-home positions (default: True)
        employment_type: Type of employment - "permanent" for full-time permanent, "freelance" for contracts/projects, "both" for all types (default: "both")
        limit: Maximum number of results to return (1-25, default: 10)
        source: Search backend - "web" for DuckDuckGo (free, no key needed, default), "adzuna" for Adzuna API (requires ADZUNA_APP_ID and ADZUNA_API_KEY env vars)
        salary_min: Minimum salary filter (optional, only supported by Adzuna backend)
        country: 2-letter country code for Adzuna searches (default: "us"). Supported: gb, us, in, au, nz, ca, za, fr, de, pl, br, at
    """
    if limit < 1:
        limit = 1
    elif limit > 25:
        limit = 25

    if employment_type not in ("permanent", "freelance", "both"):
        employment_type = "both"

    if source == "adzuna":
        return _search_adzuna(query, location, remote, employment_type, limit, country)

    return _search_web(query, location, remote, employment_type, limit)


@mcp.tool()
def list_categories() -> str:
    """List suggested job search categories and their keywords.

    These are suggestions only — you can use any query with search_jobs().
    Each category shows keywords for permanent and freelance searches.
    """
    categories = {
        "software-engineering": "software engineer, developer, programmer, full stack",
        "data-science": "data scientist, data analyst, machine learning, AI engineer",
        "cybersecurity": "security engineer, cybersecurity analyst, penetration tester",
        "devops-cloud": "DevOps engineer, SRE, cloud architect, platform engineer",
        "web-mobile-dev": "React developer, Flutter, iOS developer, Android developer",
        "ai-ml": "machine learning engineer, AI engineer, NLP engineer, computer vision",
        "design-ux": "UX designer, UI designer, product designer, graphic designer",
        "writing-content": "technical writer, copywriter, content writer, editor",
        "marketing": "marketing manager, digital marketing, SEO, growth marketing",
        "sales": "account executive, sales representative, business development",
        "finance-accounting": "financial analyst, accountant, controller, auditor",
        "hr-recruiting": "HR manager, recruiter, talent acquisition, people operations",
        "project-management": "project manager, program manager, Scrum Master, PMP",
        "customer-support": "customer success, support specialist, account manager",
        "healthcare": "nurse, physician, medical assistant, healthcare administrator",
        "legal": "lawyer, paralegal, legal assistant, compliance officer",
        "education-training": "teacher, professor, instructor, corporate trainer",
        "engineering": "mechanical engineer, civil engineer, electrical engineer",
        "video-photo": "video editor, photographer, motion designer, animator",
        "virtual-assistant": "virtual assistant, executive assistant, admin support",
        "entry-level-internship": "entry level, junior, graduate, intern, new grad",
    }

    lines = [
        "Suggested Job Search Categories:\n",
        "Use any of these as your query in search_jobs(). "
        "Combine with 'remote', 'freelance', etc.\n",
    ]
    for cat, keywords in categories.items():
        lines.append(f"  • {cat}: {keywords}")
    lines.append("")
    lines.append(f"Supported countries for Adzuna: {', '.join(sorted(ADZUNA_COUNTRIES.keys()))}")
    lines.append("")
    lines.append("Example queries:")
    lines.append('  search_jobs(query="Python developer", remote=True, employment_type="freelance")')
    lines.append('  search_jobs(query="data scientist", location="London", employment_type="permanent")')
    return "\n".join(lines)


@mcp.tool()
def get_job_details(url: str) -> str:
    """Fetch full details of a job listing from its URL.

    Args:
        url: The full URL of the job listing to fetch

    Returns:
        The full text content of the job listing page
    """
    import requests
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
        })
        resp.raise_for_status()
    except Exception as e:
        return f"Failed to fetch job details: {e}"

    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:5000] + ("\n\n...(truncated)" if len(text) > 5000 else "")
    except ImportError:
        return resp.text[:5000]
    except Exception as e:
        return f"Error parsing page: {e}"


@mcp.tool()
def match_cv_to_jobs(
    file_path: str,
    remote: bool = True,
    employment_type: str = "both",
    limit: int = 10,
    source: str = "web",
) -> str:
    """Upload a CV (PDF or DOCX) and find matching jobs based on your skills and experience.

    The system extracts your job titles, skills, experience level, and location from the CV,
    then searches for the most relevant job listings.

    Args:
        file_path: Absolute path to your CV file (.pdf or .docx)
        remote: Whether to search for remote/work-from-home positions (default: True)
        employment_type: "permanent", "freelance", or "both" (default: "both")
        limit: Maximum number of job matches to return (1-25, default: 10)
        source: Search backend - "web" (default) or "adzuna"

    Returns:
        Formatted job matches with relevance indicators based on your CV
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    try:
        text = _extract_text(str(path))
    except Exception as e:
        return f"Error reading file: {e}"

    if not text.strip():
        return "Could not extract text from the file. The file may be empty or corrupted."

    cv_data = _parse_cv(text)

    if not cv_data["titles"] and not cv_data["skills"]:
        return ("Could not identify job titles or skills from your CV. "
                "Please make sure your CV contains clear job titles and skill keywords.")

    query_parts = []
    if cv_data["titles"]:
        query_parts.append(" ".join(cv_data["titles"][:3]))
    if cv_data["skills"]:
        query_parts.append(" ".join(cv_data["skills"][:10]))
    query = " ".join(query_parts)

    location = cv_data["locations"][0] if cv_data["locations"] else None

    result = _search_web(query, location, remote, employment_type, limit * 2)
    if not result or result.startswith("No job"):
        return ("No matching jobs found based on your CV. "
                f"Try adjusting the search parameters. "
                f"Extracted from your CV: titles={cv_data['titles'][:3]}, "
                f"skills={cv_data['skills'][:5]}, "
                f"experience={cv_data['experience_years']}yrs")

    lines = [
        "📄 *CV Analysis Complete*\n",
        f"**Extracted from your CV:**",
        f"  • Job Titles: {', '.join(cv_data['titles'][:5]) or 'Not detected'}",
        f"  • Skills: {', '.join(cv_data['skills'][:10]) or 'Not detected'}",
        f"  • Experience: {cv_data['experience_years'] or 'Not specified'} years",
        f"  • Locations: {', '.join(cv_data['locations'][:3]) or 'Not specified'}",
        f"  • Education: {cv_data['education'][0] if cv_data['education'] else 'Not specified'}",
        "",
        "🔍 *Matching Jobs:*",
        "",
    ]

    if source == "adzuna":
        adzuna_result = _search_adzuna(query, location, remote, employment_type, limit,
                                        os.getenv("JOBSEARCH_DEFAULT_COUNTRY", "us"))
        if not adzuna_result.startswith("No job") and not adzuna_result.startswith("Adzuna"):
            lines.append(f"📊 Adzuna Results:\n{adzuna_result}\n")

    lines.append(f"🌐 Web Search Results:\n{result}")
    return "\n".join(lines)


@mcp.tool()
def search_by_skills(
    skills: str,
    remote: bool = True,
    employment_type: str = "both",
    limit: int = 10,
    source: str = "web",
) -> str:
    """Search for jobs requiring specific skills or technologies.

    Args:
        skills: Comma-separated list of skills (e.g., "Python, React, AWS, Docker")
        remote: Whether to search for remote/work-from-home positions (default: True)
        employment_type: "permanent", "freelance", or "both" (default: "both")
        limit: Maximum number of results (1-25, default: 10)
        source: Search backend - "web" (default) or "adzuna"

    Returns:
        Job listings matching the specified skills
    """
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    if not skill_list:
        return "Please provide at least one skill."
    query = " ".join(skill_list[:5])
    return search_jobs(query=query, remote=remote, employment_type=employment_type,
                       limit=limit, source=source)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
