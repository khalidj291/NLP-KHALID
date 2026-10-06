"""
Skills Database
================
Curated list of technical and soft skills for matching against resumes
and job descriptions. Organized by category for structured output.
"""

TECHNICAL_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "swift", "kotlin", "scala", "r", "matlab", "php", "perl",
    "objective-c", "dart", "lua", "haskell", "elixir", "clojure",
    "assembly", "fortran", "cobol", "visual basic", "shell scripting",
    "bash", "powershell", "groovy",

    # Web Development
    "html", "css", "react", "angular", "vue", "vue.js", "next.js", "nuxt.js",
    "svelte", "jquery", "bootstrap", "tailwind", "tailwindcss", "sass", "less",
    "webpack", "vite", "node.js", "express", "express.js", "django", "flask",
    "fastapi", "spring boot", "asp.net", "ruby on rails", "laravel",
    "graphql", "rest api", "restful api", "websocket", "ajax",
    "responsive design", "web development", "frontend", "backend",
    "full stack", "fullstack",

    # Data Science & ML
    "machine learning", "deep learning", "artificial intelligence",
    "neural networks", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "transfer learning",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "opencv", "spacy", "nltk", "hugging face", "transformers",
    "bert", "gpt", "lstm", "cnn", "rnn", "gan",
    "random forest", "decision tree", "svm", "support vector machine",
    "logistic regression", "linear regression", "gradient boosting",
    "xgboost", "lightgbm", "catboost", "ensemble methods",
    "feature engineering", "feature selection", "hyperparameter tuning",
    "model deployment", "mlops", "data preprocessing",
    "data visualization", "statistical analysis", "statistics",
    "probability", "bayesian", "a/b testing", "hypothesis testing",
    "time series", "forecasting", "anomaly detection",
    "recommendation system", "sentiment analysis", "text mining",
    "text classification", "named entity recognition", "ner",
    "image classification", "object detection", "image segmentation",
    "generative ai", "large language models", "llm", "prompt engineering",
    "rag", "retrieval augmented generation", "fine tuning",
    "data science", "data analysis", "data analytics",
    "big data", "data engineering", "etl", "data pipeline",
    "data warehouse", "data lake", "data modeling",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra",
    "elasticsearch", "dynamodb", "firebase", "sqlite", "oracle",
    "sql server", "mariadb", "neo4j", "couchdb", "influxdb",
    "nosql", "database design", "database administration",

    # Cloud & DevOps
    "aws", "amazon web services", "azure", "google cloud", "gcp",
    "docker", "kubernetes", "jenkins", "ci/cd", "terraform",
    "ansible", "puppet", "chef", "vagrant", "nginx", "apache",
    "linux", "unix", "windows server",
    "microservices", "serverless", "lambda", "cloud computing",
    "devops", "site reliability engineering", "sre",
    "monitoring", "logging", "prometheus", "grafana", "datadog",
    "load balancing", "auto scaling", "infrastructure as code",

    # Version Control & Tools
    "git", "github", "gitlab", "bitbucket", "svn",
    "jira", "confluence", "trello", "asana",
    "vs code", "intellij", "eclipse", "vim",
    "postman", "swagger", "figma", "sketch", "adobe xd",

    # Mobile Development
    "android", "ios", "react native", "flutter", "xamarin",
    "mobile development", "swift ui", "jetpack compose",

    # Cybersecurity
    "cybersecurity", "network security", "penetration testing",
    "ethical hacking", "vulnerability assessment", "encryption",
    "firewall", "intrusion detection", "siem", "oauth", "jwt",
    "ssl", "tls", "authentication", "authorization",

    # Data Formats & APIs
    "json", "xml", "yaml", "csv", "parquet", "avro",
    "api development", "api integration", "soap", "grpc",

    # Testing
    "unit testing", "integration testing", "selenium", "cypress",
    "jest", "mocha", "pytest", "junit", "test automation",
    "tdd", "bdd", "qa", "quality assurance",

    # Other Technical
    "blockchain", "iot", "internet of things", "embedded systems",
    "robotics", "arduino", "raspberry pi", "3d printing",
    "ar", "vr", "augmented reality", "virtual reality",
    "game development", "unity", "unreal engine",
    "agile", "scrum", "kanban", "waterfall", "sdlc",
    "design patterns", "solid principles", "oop",
    "object oriented programming", "functional programming",
    "data structures", "algorithms", "system design",
    "distributed systems", "parallel computing", "concurrency",
    "caching", "message queue", "rabbitmq", "kafka", "apache kafka",
    "apache spark", "hadoop", "hive", "pig", "airflow",
    "tableau", "power bi", "looker", "excel", "google sheets",
    "erp", "sap", "salesforce", "crm",
}

SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "creativity", "adaptability", "time management",
    "project management", "collaboration", "presentation",
    "public speaking", "negotiation", "conflict resolution",
    "decision making", "analytical thinking", "attention to detail",
    "organizational skills", "interpersonal skills", "mentoring",
    "coaching", "strategic thinking", "innovation",
    "customer service", "client management", "stakeholder management",
    "cross-functional", "self-motivated", "proactive",
    "multitasking", "flexibility", "emotional intelligence",
    "work ethic", "accountability", "reliability",
    "written communication", "verbal communication",
    "team building", "team management", "people management",
    "risk management", "change management", "process improvement",
    "continuous improvement", "research", "documentation",
    "technical writing", "report writing", "budget management",
}

CERTIFICATIONS = {
    "aws certified", "azure certified", "google certified",
    "pmp", "project management professional",
    "scrum master", "csm", "certified scrum master",
    "cissp", "cisa", "ceh", "comptia",
    "ccna", "ccnp", "ccie",
    "itil", "togaf", "six sigma",
    "cfa", "cpa", "cma",
    "tensorflow developer certificate",
    "google data analytics",
    "ibm data science",
    "meta frontend developer",
    "oracle certified",
    "microsoft certified",
    "kubernetes certified",
    "ckad", "cka",
}

# Combined set for general matching
ALL_SKILLS = TECHNICAL_SKILLS | SOFT_SKILLS | CERTIFICATIONS

# Skill categories for display grouping
SKILL_CATEGORIES = {
    "Technical Skills": TECHNICAL_SKILLS,
    "Soft Skills": SOFT_SKILLS,
    "Certifications": CERTIFICATIONS,
}


def categorize_skill(skill: str) -> str:
    """Return the category a skill belongs to."""
    skill_lower = skill.lower()
    if skill_lower in TECHNICAL_SKILLS:
        return "Technical Skills"
    elif skill_lower in SOFT_SKILLS:
        return "Soft Skills"
    elif skill_lower in CERTIFICATIONS:
        return "Certifications"
    return "Other"
