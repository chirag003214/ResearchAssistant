from pydantic import BaseModel, Field
from typing import List, Optional
import instructor
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# --- Schema Definition ---
class ResearchMetric(BaseModel):
    paper_title: str = Field(..., description="The title or main subject of the text")
    metric_name: str = Field(..., description="The name of the metric (e.g., Accuracy, F1-Score, ROI)")
    metric_value: float = Field(..., description="The numerical value of the metric")
    unit: Optional[str] = Field(None, description="The unit (e.g., %, ms, $)")
    year: Optional[int] = Field(None, description="The year this data point is from")

class ExtractionResponse(BaseModel):
    metrics: List[ResearchMetric]

# --- Extraction Logic ---
def extract_data_from_text(text_chunk: str):
    api_key = os.getenv("GROQ_API_KEY")
    
    # 1. Initialize Groq Client patched with Instructor
    client = Groq(api_key=api_key)
    client = instructor.from_groq(client, mode=instructor.Mode.JSON)

    print("📊 Extracting structured data (via Groq)...")

    try:
        # 2. Ask Llama 3.3 to extract data (UPDATED MODEL NAME)
        extraction = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=ExtractionResponse,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Data Science Assistant. Extract statistical findings from the text."
                },
                {"role": "user", "content": f"Extract metrics:\n\n{text_chunk}"}
            ]
        )
        return extraction.metrics
    except Exception as e:
        print(f"Extraction Error: {e}")
        return []

if __name__ == "__main__":
    dummy_text = """
    In our 2023 study, we achieved an accuracy of 94.5% on the test set. 
    Previous work in 2022 only reached 88.0% accuracy.
    """
    data = extract_data_from_text(dummy_text)
    
    if data:
        for item in data:
            print(f"- [{item.year}] {item.metric_name}: {item.metric_value} {item.unit}")