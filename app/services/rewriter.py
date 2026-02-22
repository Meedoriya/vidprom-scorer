import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert in text-to-video prompt engineering.
Your task is to rewrite a weak video generation prompt to make it better.
You will be given the original prompt and its weak areas.
Return ONLY the improved prompt, nothing else."""


def rewrite_prompt(prompt: str, weak_areas: list[str]) -> str:
    weak_str = ", ".join(weak_areas) if weak_areas else "overall quality"

    user_msg = f"""Original Prompt: "{prompt}"
    
Weak areas: {weak_str}

Rewrite this prompt to improve the weak areas.
Add specific details for each weak area:
- specificity: add scene details, subjects, actions
- clarity: make the description unambiguous
- visual_richness: add style, camera, lighting, color descriptions
Keep the original idea intact."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()