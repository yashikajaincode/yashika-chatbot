import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import re

class ResumeCustomizer:
    def __init__(self):
        # Load lightweight model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Predefined skills and experiences
        self.skills_db = [
            "Python", "TensorFlow", "PyTorch", "NLP", "Machine Learning",
            "Data Analysis", "Model Deployment", "Data Visualization",
            "Cloud Computing", "SQL", "Deep Learning", "AI Research"
        ]
        
        self.experiences_db = [
            "Developed machine learning models for predictive analytics.",
            "Preprocessed and analyzed large datasets for AI-driven insights.",
            "Implemented NLP techniques for text classification.",
            "Built and deployed AI models using Flask and FastAPI.",
            "Worked on cloud platforms like AWS and GCP for scalable AI solutions.",
            "Performed data visualization using Tableau and Power BI."
        ]

    def extract_job_details(self, job_description):
        """Extract job title, skills, and relevant experiences from JD."""
        # Extract job title (first line before a dash or colon)
        title_match = re.search(r"^(.+?)(?:\s[-:]\s|\n)", job_description)
        job_title = title_match.group(1).strip() if title_match else "the role"

        # Extract matching skills
        skills = [skill for skill in self.skills_db if skill.lower() in job_description.lower()]
        
        # Match experiences using semantic similarity
        job_desc_embedding = self.model.encode(job_description)
        experiences_embeddings = self.model.encode(self.experiences_db)
        
        similarities = cosine_similarity(job_desc_embedding.reshape(1, -1), experiences_embeddings)[0]
        top_experiences = [self.experiences_db[idx] for idx in similarities.argsort()[-2:][::-1]]

        return {
            "job_title": job_title,
            "skills": skills or ["General ML Skills"],
            "experiences": top_experiences
        }

    def generate_fit_summary(self, job_details):
        """Generate a dynamic fit summary based on the extracted job title."""
        skills_str = ", ".join(job_details['skills'])
        experiences_str = " and ".join(job_details['experiences'])
        
        return (
            f"Yashika demonstrates strong alignment with the {job_details['job_title']} role, "
            f"possessing key skills in {skills_str}. Her previous experience includes {experiences_str}, "
            f"making her a strong candidate for this opportunity."
        )

def resume_ai_agent(job_description):
    """Gradio interface function."""
    customizer = ResumeCustomizer()
    
    try:
        job_details = customizer.extract_job_details(job_description)
        fit_summary = customizer.generate_fit_summary(job_details)
        return fit_summary
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=resume_ai_agent,
    inputs=gr.Textbox(
        label="Paste Job Description", 
        lines=6, 
        placeholder="Enter complete job description here..."
    ),
    outputs=gr.Textbox(label="Why You're a Perfect Fit"),
    title="🚀 AI Job Fit Analyzer",
    description="Paste a job description, and the AI will generate a tailored response explaining why you're a great fit!"
)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860)
