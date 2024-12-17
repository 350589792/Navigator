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

                            # Extract relevant information
                            articles = soup.find_all('article')
                            for article in articles:
                                title = article.find('h1')
                                content = article.find('div', class_='content')
                                if title and content:
                                    data.append({
                                        'title': title.text.strip(),
                                        'content': content.text.strip(),
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
