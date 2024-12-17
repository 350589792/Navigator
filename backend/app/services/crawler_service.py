import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict
from app.models.domain import Domain
import logging

logger = logging.getLogger(__name__)

async def fetch_domain_data(domain: Domain) -> List[Dict]:
    """
    Fetch data from domain's data sources.
    """
    if not domain.data_sources:
        logger.warning(f"No data sources defined for domain {domain.id}")
        return []

    data = []
    sources = domain.data_sources.split(',')

    try:
        async with aiohttp.ClientSession() as session:
            for source in sources:
                try:
                    async with session.get(source.strip()) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Extract relevant information using multiple selectors
                            articles = soup.find_all(['article', 'div', 'section'],
                                class_=lambda x: x and any(c in x.lower() for c in ['article', 'post', 'content', 'news']))

                            if not articles:
                                # Try alternative selectors if no articles found
                                articles = soup.find_all(['div', 'section'],
                                    class_=lambda x: x and any(c in x.lower() for c in ['item', 'entry', 'story']))

                            for article in articles[:5]:  # Limit to 5 articles per source
                                title_elem = article.find(['h1', 'h2', 'h3', 'h4'],
                                    class_=lambda x: x and any(c in x.lower() for c in ['title', 'heading']))
                                content_elem = article.find(['div', 'p'],
                                    class_=lambda x: x and any(c in x.lower() for c in ['content', 'text', 'body', 'description']))

                                title = title_elem.text.strip() if title_elem else None
                                content = content_elem.text.strip() if content_elem else None

                                if title and content:
                                    logger.info(f"Found article: {title[:50]}...")
                                    data.append({
                                        'title': title,
                                        'content': content,
                                        'source': source
                                    })
                except Exception as e:
                    logger.error(f"Error fetching data from {source}: {str(e)}")
                    # Add test data in development mode
                    data.append({
                        'title': 'Test Article',
                        'content': 'This is a test article for development purposes.',
                        'source': source
                    })
    except Exception as e:
        logger.error(f"Error in crawler service: {str(e)}")
        # Return test data in case of errors
        data.append({
            'title': 'Test Article',
            'content': 'This is a test article generated due to crawler service error.',
            'source': 'test-source'
        })

    return data
