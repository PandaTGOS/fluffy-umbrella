import random
import string
import json
import os
from typing import List, Dict

# Configuration
NUM_DOCS = 1000
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")
os.makedirs(DATA_DIR, exist_ok=True)

ROLES = ["admin", "manager", "member", "guest"]
TOPICS = ["finance", "engineering", "hr", "sales", "marketing", "legal", "compliance", "security"]

# Topic-specific content templates
CONTENT_TEMPLATES = {
    "finance": [
        "The Q3 revenue reports show a significant increase in margins.",
        "Cost optimization strategies have reduced overhead by 15%.",
        "Budget allocation for the upcoming fiscal year is finalized.",
        "Investment in new market sectors is yielding positive returns.",
        "Audit results indicate full compliance with financial regulations.",
        "Market volatility has been factored into the risk models.",
        "Cash flow projections remain strong for the next quarter."
    ],
    "engineering": [
        "The new microservices architecture improved system scalability.",
        "Latency issues in the database layer have been resolved.",
        "The deployment pipeline now fully supports blue-green deployments.",
        "Code review turnaround time has decreased by 20%.",
        "Technical debt reduction is a priority for this sprint.",
        "We are evaluating Rust for performance-critical components.",
        "The AI model training infrastructure is now live."
    ],
    "hr": [
        "Employee engagement surveys show high satisfaction rates.",
        "The new remote work policy is now in effect.",
        "Performance review cycles are scheduled for next month.",
        "We are launching a new mentorship program for junior staff.",
        "Diversity and inclusion initiatives are being expanded.",
        "Health benefits have been upgraded for all full-time employees.",
        "The annual team building offsite is confirmed for September."
    ],
    "sales": [
        "The sales team exceeded the quarterly targets by 10%.",
        "New client acquisition strategies are showing promising results.",
        "Customer retention rates have stabilized this quarter.",
        "The enterprise deal with Acme Corp has been closed.",
        "Sales training workshops will be held next week.",
        "The CRM migration is completed successfully.",
        "Regional sales performance in APAC is topping the charts."
    ],
    "marketing": [
        "The social media campaign generated 50% more leads.",
        "Brand awareness metrics are trending upwards.",
        "We are sponsoring the upcoming tech conference.",
        "The new product launch event was a huge success.",
        "Content marketing strategy focuses on thought leadership.",
        "SEO optimization has doubled organic traffic.",
        "The email newsletter open rate has hit a record high."
    ],
    "legal": [
        "The new privacy policy has been updated for GDPR compliance.",
        "Contract negotiations are ongoing with key vendors.",
        "Intellectual property filings have been submitted.",
        "Legal review of the new partnership agreement is complete.",
        "Compliance training is mandatory for all employees.",
        "We are monitoring changes in international trade laws.",
        "The patent portfolio has expanded by 5 patents this year."
    ],
    "compliance": [
        "We successfully passed the ISO 27001 audit.",
        "Internal controls have been strengthened across departments.",
        "Regulatory reporting requirements have been met on time.",
        "Risk assessment modules are being updated.",
        "Data protection policies are strictly enforced.",
        "Whistleblower anonymity protocols have been verified.",
        "Ethics committee meets quarterly to review incidents."
    ],
    "security": [
        "The vulnerability scan detected no critical issues.",
        "Access control policies have been tightened for admin roles.",
        "Security awareness training is mandatory for all staff.",
        "The incident response plan was tested successfully.",
        "Encryption standards have been upgraded to AES-256.",
        "Multi-factor authentication is now enforced globally.",
        "Third-party vendor security assessments are ongoing."
    ]
}

def generate_long_text(topic: str) -> str:
    """Generates a longer document with multiple paragraphs."""
    templates = CONTENT_TEMPLATES.get(topic, CONTENT_TEMPLATES["engineering"])
    paragraphs = []
    
    # Generate 5-10 paragraphs
    for _ in range(random.randint(5, 10)):
        # Each paragraph has 3-6 sentences
        sentences = random.choices(templates, k=random.randint(3, 6))
        paragraphs.append(" ".join(sentences))
        
    return "\n\n".join(paragraphs)

def generate_data():
    print(f"Generating {NUM_DOCS} individual documents in {DATA_DIR}...")
    
    # Clear directory first
    for f in os.listdir(DATA_DIR):
        os.remove(os.path.join(DATA_DIR, f))

    for i in range(NUM_DOCS):
        topic = random.choice(TOPICS)
        role = random.choice(ROLES)
        title = f"{topic.capitalize()} Report {i}"
        
        content = generate_long_text(topic)
        
        # Create Frontmatter
        metadata = {
            "id": f"doc_{i}",
            "title": title,
            "topic": topic,
            "access_required": [role] if role != "guest" else []
        }
        
        file_content = f"""---
{json.dumps(metadata, indent=2)}
---

# {title}

**Topic:** {topic.capitalize()}
**Confidentiality:** {role.upper()} only.

{content}
"""
        file_path = os.path.join(DATA_DIR, f"doc_{i}.md")
        with open(file_path, "w") as f:
            f.write(file_content)
            
    print(f"Generated {NUM_DOCS} files.")

if __name__ == "__main__":
    generate_data()
