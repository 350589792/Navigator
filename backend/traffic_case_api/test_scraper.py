import asyncio
import re
from app.scraper import TrafficCaseScraper

async def test_scraper():
    print("开始测试案例数据采集...")
    scraper = await TrafficCaseScraper.create(test_mode=True)
    async with scraper:
        cases = await scraper.update_case_database()
        
        # 验证案例数量
        if len(cases) < 500:
            raise ValueError(f"案例数量不足: 期望500个，实际{len(cases)}个")
        print(f'\n总共收集案例数: {len(cases)}')
        
        # 验证案例质量
        required_fields = ['case_number', 'court', 'content', 'judgment', 'laws_applied']
        invalid_cases = []
        
        for i, case in enumerate(cases):
            errors = []
            
            # 检查必填字段
            for field in required_fields:
                if not case.get(field):
                    errors.append(f"缺少{field}字段")
                    
            # 验证案号格式
            if case.get('case_number') and not re.match(r'（\d{4}）[^号]+号', case['case_number']):
                errors.append("案号格式不正确")
                
            # 验证法院名称
            if case.get('court') and not case['court'].endswith('人民法院'):
                errors.append("法院名称格式不正确")
                
            # 验证内容长度
            if case.get('content') and len(case['content']) < 100:
                errors.append("案例内容过短")
                
            # 验证法条引用
            if not case.get('laws_applied') or len(case['laws_applied']) == 0:
                errors.append("缺少法条引用")
                
            if errors:
                invalid_cases.append((i, errors))
        
        # 输出验证结果
        if invalid_cases:
            print("\n发现以下问题案例:")
            for case_index, errors in invalid_cases:
                print(f"案例 {case_index + 1}:")
                for error in errors:
                    print(f"  - {error}")
            raise ValueError(f"发现 {len(invalid_cases)} 个问题案例")
        
        print("\n所有案例验证通过！")
        if cases:
            print('\n示例案例:')
            print(f'法院: {cases[0].get("court", "N/A")}')
            print(f'案号: {cases[0].get("case_number", "N/A")}')
            print(f'判决: {cases[0].get("judgment", "N/A")}')
            print(f'法条: {", ".join(cases[0].get("laws_applied", []))}')
            
            # 验证数据质量
            print("\n开始验证案例数据质量...")
            errors_found = 0
            for i, case in enumerate(cases):
                case_errors = []
                
                # 验证必需字段存在且非空
                required_fields = ["court", "case_number", "judgment", "laws_applied", "content", "date", "title"]
                for field in required_fields:
                    if not case.get(field):
                        case_errors.append(f"缺少必需字段或字段为空: {field}")
                
                if not case_errors:
                    # 验证案号格式
                    if not (case['case_number'].startswith('（') and 
                          case['case_number'].endswith('号') and 
                          '刑初' in case['case_number']):
                        case_errors.append(f"案号格式错误: {case['case_number']}")
                    
                    # 验证法院名称
                    if not case['court'].endswith('人民法院'):
                        case_errors.append(f"法院名称格式错误: {case['court']}")
                    
                    # 验证判决内容
                    if len(case['content']) < 50:
                        case_errors.append("判决书内容过短")
                    
                    # 验证法条引用
                    if not isinstance(case['laws_applied'], list) or len(case['laws_applied']) < 1:
                        case_errors.append("缺少法条引用")
                    else:
                        for law in case['laws_applied']:
                            if not ('中华人民共和国' in law and '法' in law):
                                case_errors.append(f"法条格式错误: {law}")
                
                if case_errors:
                    errors_found += 1
                    if errors_found <= 5:  # 只显示前5个错误案例的详细信息
                        print(f"\n案例 {i+1} 存在以下问题：")
                        for error in case_errors:
                            print(f"- {error}")
            
            valid_cases = len(cases) - errors_found
            print(f'\n有效案例数: {valid_cases} (通过所有验证检查的案例)')
            print(f'案例合格率: {(valid_cases/len(cases)*100):.2f}%')
        
if __name__ == "__main__":
    asyncio.run(test_scraper())
