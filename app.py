from flask import Flask, request, jsonify, send_from_directory
from app.similarity import SimilarityAnalyzer
from app.data_manager import CaseDataManager
from app.models import Case
from app.utils import custom_tokenizer
import os

app = Flask(__name__, static_folder='app/static')

# Initialize components
data_manager = CaseDataManager(data_dir="data")
similarity_analyzer = SimilarityAnalyzer(min_similarity=0.15)

@app.route("/analyze_case", methods=["POST"])
def analyze_case():
    try:
        body = request.json
        if not body or not body.get("content"):
            return jsonify({"error": "案例内容不能为空"}), 422

        # Create case object
        user_case = Case(content=body["content"])
        
        # Get all cases
        all_cases = data_manager.get_all_cases_sync()
        if not all_cases:
            return jsonify({"error": "案例数据库为空"}), 500
            
        # Prepare texts for similarity analysis
        all_texts = [case["content"] for case in all_cases]
        all_texts.append(user_case.content)
        
        # Transform all texts to vectors
        case_vectors = similarity_analyzer.fit_transform(all_texts[:-1])
        query_vector = similarity_analyzer.transform([user_case.content])
        
        # Get similar cases
        similar_cases_with_scores = similarity_analyzer.get_similar_cases(
            query_vector, 
            case_vectors, 
            all_cases,
            top_k=5
        )
        
        # Extract relevant laws based on keywords in the case
        relevant_laws = []
        keywords = similarity_analyzer.legal_keywords.keys()
        case_text = user_case.content.lower()
        
        # Basic traffic laws
        basic_laws = [
            "《中华人民共和国道路交通安全法》第一百一十九条规定，驾驶人有下列情形之一的，处2000元以上5000元以下罚款：（一）违反道路交通安全法律、法规关于机动车停放、临时停车规定的。",
            "《中华人民共和国道路交通安全法》第九十一条，饮酒后驾驶机动车的，处暂扣6个月机动车驾驶证，并处1000元以上2000元以下罚款。",
            "《中华人民共和国道路交通安全法》第四十二条，机动车上道路行驶，不得超过限速标志标明的最高时速。"
        ]
        
        # Add laws based on keywords found in case
        if "酒" in case_text or "醉" in case_text:
            relevant_laws.append(
                "《中华人民共和国刑法》第一百三十三条规定：违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役。"
            )
            relevant_laws.append(
                "《中华人民共和国道路交通安全法》第九十一条规定：饮酒后驾驶机动车的，处暂扣6个月机动车驾驶证，并处1000元以上2000元以下罚款。醉酒后驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任。"
            )
            
        if "超速" in case_text or "速度" in case_text:
            relevant_laws.append(
                "《中华人民共和国道路交通安全法》第四十二条规定：机动车上道路行驶，不得超过限速标志标明的最高时速。在没有限速标志的路段，应当保持安全车速。"
            )
            
        if "逃逸" in case_text:
            relevant_laws.append(
                "《中华人民共和国刑法》第一百三十三条之一规定：在道路上驾驶机动车，发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失，逃逸的，处三年以上七年以下有期徒刑。"
            )

        # Add basic laws if no specific laws were found
        if not relevant_laws:
            relevant_laws.extend(basic_laws)

        # Format response
        response_data = {
            "similar_cases": [
                {
                    "content": case["content"],
                    "similarity_score": score
                }
                for case, score in similar_cases_with_scores
            ],
            "relevant_laws": relevant_laws
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing case: {str(e)}")
        return jsonify({"error": "处理案例时发生错误"}), 500

@app.route("/")
def home():
    return app.send_static_file('index.html')

if __name__ == "__main__":
    # Use localhost for better security
    # debug=True is for development only
    app.run(host='127.0.0.1', port=5000, debug=True)
