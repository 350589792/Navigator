from typing import Tuple, List, Dict
from openai import AsyncOpenAI
from app.core.config import settings
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)
client = None

try:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
except Exception as e:
    logger.warning(f"OpenAI client initialization failed: {e}. Using NLTK fallback mode.")

def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tagged = pos_tag(tokens)

    important_words = [word for word, tag in tagged
                      if tag.startswith(('NN', 'JJ'))
                      and word not in stop_words
                      and len(word) > 2]

    return [word for word, _ in Counter(important_words).most_common(num_keywords)]

def analyze_trends(data: List[Dict]) -> List[str]:
    all_text = ' '.join(item.get('content', '') for item in data)
    sentences = sent_tokenize(all_text)

    trend_indicators = ['increase', 'decrease', 'grow', 'decline', 'rise', 'fall', 'trend']
    trends = []

    for sentence in sentences:
        if any(indicator in sentence.lower() for indicator in trend_indicators):
            trends.append(sentence)

    return trends[:5]

def extract_key_points(text: str, num_points: int = 5) -> List[str]:
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))

    word_freq = Counter(w for w in words if w not in stop_words and w.isalnum())

    sent_scores = {}
    for sentence in sentences:
        score = sum(word_freq[word.lower()]
                   for word in word_tokenize(sentence)
                   if word.lower() in word_freq)
        sent_scores[sentence] = score

    return [sent for sent, _ in sorted(sent_scores.items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:num_points]]

async def generate_report(
    data: List[Dict],
    format: str = 'text',
    include_keywords: bool = True,
    include_trends: bool = True,
    include_key_points: bool = True
) -> Tuple[str, str, str]:
    if not client or not settings.OPENAI_API_KEY:
        logger.info("Using NLTK-based report generation")

        all_content = '\n'.join(f"{item.get('title', '')}\n{item.get('content', '')}"
                               for item in data)

        sections = []
        if include_keywords:
            keywords = extract_keywords(all_content)
            sections.append("Key Industry Keywords:\n" +
                          "\n".join(f"- {keyword}" for keyword in keywords))

        if include_trends:
            trends = analyze_trends(data)
            sections.append("Current Trends:\n" +
                          "\n".join(f"- {trend}" for trend in trends))

        if include_key_points:
            key_points = extract_key_points(all_content)
            sections.append("Key Points:\n" +
                          "\n".join(f"- {point}" for point in key_points))

        content = "\n\n".join(sections)
        if format == 'html':
            content = f"<div class='report'>\n" + \
                     "\n".join(f"<section class='section'>{section}</section>"
                             for section in sections) + \
                     "\n</div>"

        title = f"Industry Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
        summary = "Report generated using local natural language processing."

        return title, content, summary

    try:
        prompt = "Based on the following information, generate a comprehensive industry report"
        if format == 'html':
            prompt += " in HTML format with appropriate tags and styling"
        prompt += ".\n\nInclude the following sections:\n"

        if include_keywords:
            prompt += "- Key industry keywords and their significance\n"
        if include_trends:
            prompt += "- Current trends and future predictions\n"
        if include_key_points:
            prompt += "- Key points and insights\n"

        prompt += "\nSource data:\n\n"

        for item in data:
            prompt += f"Source: {item['source']}\n"
            prompt += f"Title: {item['title']}\n"
            prompt += f"Content: {item['content']}\n\n"

        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert industry analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        report_text = response.choices[0].message.content

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
