import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any
import time
import random
from datetime import datetime, timedelta
import json
import re
from .data_manager import CaseDataManager
from urllib.parse import quote, urljoin
import logging
from aiohttp_retry import RetryClient, ExponentialRetry
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp_proxy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementation for controlling request rates"""
    def __init__(self, requests_per_minute: int, burst_requests: int, cooldown_time: int, backoff_factor: float):
        self.requests_per_minute = requests_per_minute
        self.burst_requests = burst_requests
        self.cooldown_time = cooldown_time
        self.backoff_factor = backoff_factor
        self.requests = []
        self.cooldown_until = None
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request with rate limiting"""
        async with self._lock:
            now = time.time()
            
            # Clear old requests
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # Check cooldown
            if self.cooldown_until and now < self.cooldown_until:
                sleep_time = self.cooldown_until - now
                logger.info(f"Rate limit cooldown: sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                self.cooldown_until = None
                self.requests = []
                
            # Check rate limit
            if len(self.requests) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                logger.info(f"Rate limit reached: sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                self.requests = self.requests[1:]
                
            # Add request
            self.requests.append(now)
            
            # Check burst limit
            if len(self.requests) > self.burst_requests:
                sleep_time = random.uniform(1, 3)
                logger.info(f"Burst limit reached: adding delay of {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
    
    def trigger_cooldown(self):
        """Trigger a cooldown period after detecting rate limiting"""
        now = time.time()
        if not self.cooldown_until:
            self.cooldown_until = now + self.cooldown_time
        else:
            # Extend cooldown with backoff
            self.cooldown_until = now + (self.cooldown_time * self.backoff_factor)
        logger.warning(f"Triggered rate limit cooldown until {datetime.fromtimestamp(self.cooldown_until)}")

# Constants
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 10

# Working proxy list from trusted proxy providers
PROXY_LIST = [
    'http://51.158.68.133:8811',  # France
    'http://51.158.68.68:8811',   # France
    'http://51.158.119.88:8811',  # France
    'http://176.31.68.252:20000', # France
    'http://51.158.172.165:8811', # France
    'http://159.203.61.169:8080', # USA
    'http://167.172.158.85:81',   # USA
    'http://165.227.71.60:80',    # USA
    'http://178.62.92.133:80',    # UK
    'http://139.59.1.14:8080'     # India
]

# Rate limiting settings
RATE_LIMIT_CONFIG = {
    'default': {
        'requests_per_minute': 30,
        'burst_requests': 5,
        'cooldown_time': 60,
        'backoff_factor': 2
    },
    'court_gov': {
        'requests_per_minute': 20,
        'burst_requests': 3,
        'cooldown_time': 120,
        'backoff_factor': 3
    },
    'pkulaw': {
        'requests_per_minute': 25,
        'burst_requests': 4,
        'cooldown_time': 90,
        'backoff_factor': 2.5
    },
    'lawlib': {
        'requests_per_minute': 15,
        'burst_requests': 3,
        'cooldown_time': 180,
        'backoff_factor': 4
    }
}

class TrafficCaseScraper:
    def __init__(self):
        self.data_manager = None
        self.base_url = "https://wenshu.court.gov.cn"
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.rate_limiters = {
            'default': RateLimiter(**RATE_LIMIT_CONFIG['default']),
            'court_gov': RateLimiter(**RATE_LIMIT_CONFIG['court_gov']),
            'pkulaw': RateLimiter(**RATE_LIMIT_CONFIG['pkulaw']),
            'lawlib': RateLimiter(**RATE_LIMIT_CONFIG['lawlib'])
        }
        self.search_params = {
            '案件类型': '刑事案件',
            '文书类型': '判决书',
            '案由': '交通肇事罪',
            '法院层级': '基层法院',
            '裁判年份': datetime.now().year
        }

    @classmethod
    async def create(cls, test_mode=False):
        """Factory method to create and initialize a TrafficCaseScraper instance
        
        Args:
            test_mode (bool): 如果为True，则仅使用示例数据进行测试
        """
        instance = cls()
        instance.test_mode = test_mode
        instance.data_manager = await CaseDataManager.create()
        return instance
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
            self.session = None

    def _extract_laws(self, content: str) -> List[str]:
        """从判决书中提取相关法条"""
        law_patterns = [
            r'《中华人民共和国刑法》第[一二三四五六七八九十百千]+条',
            r'《中华人民共和国道路交通安全法》第[一二三四五六七八九十百千]+条',
            r'《中华人民共和国刑事诉讼法》第[一二三四五六七八九十百千]+条'
        ]
        laws = []
        for pattern in law_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if match.group() not in laws:
                    laws.append(match.group())
        return laws

    def _extract_case_info(self, doc_content: str) -> Dict:
        """从文书中提取案件信息"""
        # 提取案号
        case_number_match = re.search(r'（\d{4}）[^号]+号', doc_content)
        case_number = case_number_match.group() if case_number_match else ''
        
        # 提取法院名称
        court_match = re.search(r'(.+)人民法院', doc_content)
        court = court_match.group() if court_match else ''
        
        # 提取判决结果
        judgment_match = re.search(r'判决如下[：:]([\s\S]+?)。', doc_content)
        judgment = judgment_match.group(1).strip() if judgment_match else ''
        
        # 提取相关法条
        laws = self._extract_laws(doc_content)
        
        return {
            'case_number': case_number,
            'court': court,
            'content': doc_content,
            'judgment': judgment,
            'laws_applied': laws,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
    def _get_example_cases(self) -> List[Dict]:
        """获取示例案例数据作为备份"""
        # 定义基础数据
        # 定义法院和对应的省份代码
        court_info = {
            "北京市朝阳区人民法院": ("北京市", "京"),
            "上海市浦东新区人民法院": ("上海市", "沪"),
            "广东省广州市天河区人民法院": ("广东省", "粤"),
            "广东省深圳市南山区人民法院": ("广东省", "粤"),
            "浙江省杭州市西湖区人民法院": ("浙江省", "浙"),
            "四川省成都市武侯区人民法院": ("四川省", "川"),
            "重庆市渝中区人民法院": ("重庆市", "渝"),
            "湖北省武汉市江汉区人民法院": ("湖北省", "鄂"),
            "江苏省南京市鼓楼区人民法院": ("江苏省", "苏"),
            "陕西省西安市雁塔区人民法院": ("陕西省", "陕"),
            "山东省青岛市市南区人民法院": ("山东省", "鲁"),
            "天津市和平区人民法院": ("天津市", "津")
        }
        
        courts = list(court_info.keys())
        
        surnames = ["王", "李", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周"]
        
        accident_types = [
            ("超速行驶", "《中华人民共和国道路交通安全法》第四十二条"),
            ("疲劳驾驶", "《中华人民共和国道路交通安全法》第二十二条"),
            ("违规超车", "《中华人民共和国道路交通安全法》第四十三条"),
            ("闯红灯", "《中华人民共和国道路交通安全法》第三十八条"),
            ("酒后驾驶", "《中华人民共和国道路交通安全法》第九十一条"),
            ("未保持安全距离", "《中华人民共和国道路交通安全法》第四十二条")
        ]
        
        consequences = [
            ("造成一人死亡", "《中华人民共和国刑法》第一百三十三条", "判处有期徒刑三年，缓刑三年"),
            ("造成两人重伤", "《中华人民共和国刑法》第一百三十三条", "判处有期徒刑二年，缓刑二年"),
            ("造成三人轻伤", "《中华人民共和国刑法》第一百三十三条", "判处有期徒刑一年，缓刑一年六个月"),
            ("造成他人重伤", "《中华人民共和国刑法》第一百三十三条", "判处有期徒刑一年三个月"),
            ("造成多人受伤", "《中华人民共和国刑法》第一百三十三条", "判处拘役六个月，并处罚金人民币五千元"),
            ("醉酒驾驶机动车", "《中华人民共和国刑法》第一百三十三条之一第一款", "判处拘役三个月，并处罚金人民币三千元")
        ]
        
        # 生成500个案例
        example_cases = []
        used_case_numbers = set()
        
        for _ in range(500):
            # 生成基本信息
            year = random.randint(2020, 2023)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            court = random.choice(courts)
            province, province_code = court_info[court]
            
            # 确保案号唯一
            while True:
                case_number = f"（{year}）{province_code}{random.randint(0, 9999):04d}刑初{random.randint(1, 999):03d}号"
                if case_number not in used_case_numbers:
                    used_case_numbers.add(case_number)
                    break
            
            # 生成案例内容
            accident_type, traffic_law = random.choice(accident_types)
            consequence, criminal_law, judgment = random.choice(consequences)
            surname = random.choice(surnames)
            
            content = f"""
            {court}
            刑 事 判 决 书
            {case_number}
            
            经审理查明：{year}年{month}月{day}日，被告人{surname}某某驾驶{province_code}A××××号机动车，
            在{court[:2]}市某路段行驶时，因{accident_type}，{consequence}。案发后，被告人主动报警并积极救助伤者。
            根据《中华人民共和国刑法》和《中华人民共和国道路交通安全法》之规定，
            判决如下：
            {judgment}
            """.strip()
            
            case = {
                "title": f"被告人{surname}某某交通肇事案",
                "case_number": case_number,
                "court": court,
                "date": f"{year}-{month:02d}-{day:02d}",
                "content": content,
                "judgment": judgment,
                "laws_applied": [criminal_law, traffic_law]
            }
            example_cases.append(case)
        
        return example_cases

    async def _get_proxy(self) -> Optional[str]:
        """Get a random proxy from the pool"""
        return random.choice(PROXY_LIST) if PROXY_LIST else None

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=INITIAL_DELAY, max=MAX_DELAY),
        reraise=True
    )
    async def _fetch_case_page(self, page: int) -> Tuple[List[Dict], bool]:
        """获取指定页码的案例数据，带重试和代理支持"""
        try:
            if not self.session:
                retry_options = ExponentialRetry(attempts=MAX_RETRIES)
                self.session = RetryClient(
                    client_session=aiohttp.ClientSession(headers=self._get_headers()),
                    retry_options=retry_options
                )
            
            # Apply rate limiting
            await self.rate_limiters['default'].acquire()

            # 获取代理
            proxy = await self._get_proxy()
            if proxy:
                logger.info(f"Using proxy: {proxy}")
                
            # 构建搜索请求
            search_url = f"{self.base_url}/api/case/search"
            params = {
                'page': page,
                'pageSize': 20,
                **self.search_params
            }
            
            async with self.session.post(
                search_url,
                json=params,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    logger.error(f"请求失败: HTTP {response.status}")
                    if response.status == 429:  # Rate limit
                        await asyncio.sleep(random.uniform(30, 60))
                    elif response.status == 403:  # Forbidden
                        logger.warning("可能被封禁，切换代理...")
                        return [], False
                    return [], False
                
                data = await response.json()
                if not data.get('data', {}).get('cases', []):
                    return [], False
                
                cases = []
                for case_data in data['data']['cases']:
                    doc_id = case_data.get('docId')
                    if not doc_id:
                        continue
                    
                    # 获取完整判决书内容
                    doc_url = f"{self.base_url}/api/case/document/{doc_id}"
                    try:
                        async with self.session.get(
                            doc_url,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as doc_response:
                            if doc_response.status != 200:
                                logger.warning(f"获取文档失败: {doc_response.status}")
                                continue
                            
                            doc_data = await doc_response.json()
                            doc_content = doc_data.get('data', {}).get('content', '')
                            if not doc_content:
                                continue
                            
                            case_info = self._extract_case_info(doc_content)
                            cases.append(case_info)
                            
                            # 智能延迟策略
                            delay = random.uniform(2, 5)
                            if len(cases) % 10 == 0:  # 每10个请求增加额外延迟
                                delay += random.uniform(5, 10)
                            await asyncio.sleep(delay)
                    
                    except asyncio.TimeoutError:
                        logger.warning(f"获取文档超时: {doc_id}")
                        continue
                    except Exception as e:
                        logger.error(f"处理文档时出错: {str(e)}")
                        continue
                
                return cases, True
                
        except asyncio.TimeoutError:
            logger.error("请求超时")
            return [], False
        except aiohttp.ClientError as e:
            logger.error(f"网络错误: {str(e)}")
            return [], False
        except Exception as e:
            logger.error(f"获取案例数据失败: {str(e)}")
            return [], False

    async def _get_real_case_data(self) -> List[Dict]:
        """
        获取真实案例数据
        从裁判文书网抓取交通事故相关判例
        """
        all_cases = []
        page = 1
        
        while len(all_cases) < 500:
            cases, has_more = await self._fetch_case_page(page)
            if not has_more:
                break
                
            all_cases.extend(cases)
            print(f"已获取 {len(all_cases)} 个案例")
            page += 1
            
            # 每50个请求后暂停一下
            if page % 50 == 0:
                await asyncio.sleep(random.uniform(5, 10))
        
        return all_cases[:500]  # 确保只返回500个案例

    async def update_case_database(self):
        """更新案例数据库"""
        print("开始更新案例数据库...")
        all_cases = []
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(headers=self.headers)
            
            # 获取示例数据
            example_cases = self._get_example_cases()
            all_cases.extend(example_cases)
            print(f"已加载 {len(example_cases)} 个示例案例作为基础数据")
            
            if not self.test_mode:
                try:
                    # 尝试从裁判文书网获取数据
                    real_cases = await self._get_real_case_data()
                    if real_cases:
                        all_cases.extend(real_cases)
                        print(f"从裁判文书网成功获取 {len(real_cases)} 个案例")
                    else:
                        print("无法从裁判文书网获取数据，尝试其他来源...")
                        alternative_cases = await self._try_alternative_sources()
                        if alternative_cases:
                            all_cases.extend(alternative_cases)
                            print(f"从其他来源获取 {len(alternative_cases)} 个案例")
                except Exception as e:
                    print(f"获取在线数据时出错: {str(e)}")
                    print("继续使用已有的示例数据")
            
            # 确保案例不重复
            unique_cases = {case['case_number']: case for case in all_cases if case.get('case_number')}.values()
            all_cases = list(unique_cases)
            
            # 清空并更新数据库
            await self.data_manager.clear_database()
            for case in all_cases:
                await self.data_manager.add_case(case)
            
            print(f"成功添加 {len(all_cases)} 个独特案例到数据库")
            return all_cases
            
        except Exception as e:
            print(f"更新数据库失败: {str(e)}")
            print("使用纯示例数据作为备份")
            example_cases = self._get_example_cases()
            await self.data_manager.clear_database()
            for case in example_cases:
                await self.data_manager.add_case(case)
            return example_cases
            
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def _try_alternative_sources(self) -> List[Dict]:
        """从备选数据源获取案例数据"""
        alternative_sources = {
            "http://www.court.gov.cn/": self._scrape_court_gov,
            "http://www.pkulaw.cn/": self._scrape_pkulaw,
            "http://www.law-lib.com/": self._scrape_lawlib
        }
        
        all_cases = []
        for source_url, scraper_func in alternative_sources.items():
            try:
                logger.info(f"尝试从 {source_url} 获取数据...")
                proxy = await self._get_proxy()
                
                async with self.session.get(
                    source_url,
                    proxy=proxy,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"成功连接到 {source_url}")
                        cases = await scraper_func(response)
                        if cases:
                            all_cases.extend(cases)
                            if len(all_cases) >= 500:
                                return all_cases[:500]
                    else:
                        logger.warning(f"访问 {source_url} 失败: HTTP {response.status}")
                        
            except asyncio.TimeoutError:
                logger.error(f"连接 {source_url} 超时")
            except Exception as e:
                logger.error(f"处理 {source_url} 时出错: {str(e)}")
            
            # 在源之间添加延迟
            await asyncio.sleep(random.uniform(5, 10))
            
        return all_cases

    async def _scrape_court_gov(self, response: aiohttp.ClientResponse) -> List[Dict]:
        """从最高法院网站抓取数据"""
        cases = []
        try:
            # Apply rate limiting
            await self.rate_limiters['court_gov'].acquire()
            
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            
            # 查找交通事故相关判决
            for article in soup.find_all('article', class_='judgment-item'):
                if len(cases) >= 200:  # 每个源限制200个案例
                    break
                    
                title = article.find('h3')
                content = article.find('div', class_='content')
                if title and content and '交通' in content.text:
                    case_info = self._extract_case_info(content.text)
                    if case_info:
                        cases.append(case_info)
                        
                await asyncio.sleep(random.uniform(1, 3))
                
        except Exception as e:
            logger.error(f"解析最高法院网站数据失败: {str(e)}")
            
        return cases

    async def _scrape_pkulaw(self, response: aiohttp.ClientResponse) -> List[Dict]:
        """从北大法宝抓取数据"""
        cases = []
        try:
            # Apply rate limiting
            await self.rate_limiters['pkulaw'].acquire()
            
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            
            # 查找交通事故相关判决
            for case_div in soup.find_all('div', class_='case-item'):
                if len(cases) >= 200:
                    break
                    
                content = case_div.find('div', class_='case-content')
                if content and '交通肇事' in content.text:
                    case_info = self._extract_case_info(content.text)
                    if case_info:
                        cases.append(case_info)
                        
                await asyncio.sleep(random.uniform(1, 3))
                
        except Exception as e:
            logger.error(f"解析北大法宝数据失败: {str(e)}")
            
        return cases

    async def _scrape_lawlib(self, response: aiohttp.ClientResponse) -> List[Dict]:
        """从法律图书馆抓取数据"""
        cases = []
        try:
            # Apply rate limiting
            await self.rate_limiters['lawlib'].acquire()
            
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            
            # 查找交通事故相关判决
            for case_elem in soup.find_all('div', class_='judgment'):
                if len(cases) >= 200:
                    break
                    
                content = case_elem.find('div', class_='judgment-content')
                if content and '交通事故' in content.text:
                    case_info = self._extract_case_info(content.text)
                    if case_info:
                        cases.append(case_info)
                        
                await asyncio.sleep(random.uniform(1, 3))
                
        except Exception as e:
            logger.error(f"解析法律图书馆数据失败: {str(e)}")
            
        return cases
