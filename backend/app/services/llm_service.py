from typing import Tuple, List, Dict
from openai import AsyncOpenAI
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
client = None

try:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
except Exception as e:
    logger.warning(f"OpenAI client initialization failed: {e}. Running in test mode.")

async def generate_report(data: List[Dict]) -> Tuple[str, str, str]:
    """
    Generate a report from collected data using OpenAI's GPT model.
    Returns: (title, content, summary)
    """
    if not client or settings.OPENAI_API_KEY == "test-mode":
        # Return test data when in test mode
        return (
            "Test Report Title",
            "This is a test report generated in development mode. The actual report will be generated using OpenAI's GPT model in production.",
            "Test report summary for development purposes."
        )

    try:
        # Prepare prompt with collected data
        prompt = "Based on the following information, generate a comprehensive industry report:\n\n"
        for item in data:
            prompt += f"Source: {item['source']}\n"
            prompt += f"Title: {item['title']}\n"
            prompt += f"Content: {item['content']}\n\n"

        # Generate report using GPT
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert industry analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract report content
        report_text = response.choices[0].message.content

        # Generate title and summary using GPT-3.5-turbo
        title_response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a concise title for this report:"},
                {"role": "user", "content": report_text}
            ]
        )

        summary_response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a brief summary of this report:"},
                {"role": "user", "content": report_text}
            ]
        )

        return (
            title_response.choices[0].message.content,
            report_text,
            summary_response.choices[0].message.content
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return (
            "Error Report",
            f"Failed to generate report due to an error: {str(e)}",
            "Error occurred during report generation."
        )
