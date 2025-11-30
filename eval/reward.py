from math_verify import verify, parse
from eval.eval_utils import exact_match_score
import re

def confidence_extractor(response, **kwargs):
    """Extracts the confidence from the completions
    If a float is found within confidence tags, it is processed as follows:
    If the float is between 0 and 1, it is returned as is.
    If the float is between 1 and 100, it is divided by 100 and returned.
    If float is not directly found, the first number in the string is extracted and processed as above.
    If no float is found, 0 is returned.    
    """
    conf_pattern = r"<confidence>(.*?)</confidence>"
    # Get all <confidence>...</confidence> occurrences
    conf_matches = re.findall(conf_pattern, response, re.DOTALL | re.MULTILINE)
    # Get the last confidence, if exists
    last_confidence = conf_matches[-1].strip() if conf_matches else ""
    if last_confidence == "":
        return 0, 0.0
    else:
        try:
            confidence = float(last_confidence)
            if confidence >= 0 and confidence <= 100:
                return 1, confidence/100
            elif confidence >= 0 and confidence <= 1:
                return 1, confidence
            else:
                return 0, 0.0
        except:
            # extract the first number in the string
            first_number = re.search(r'-?\d+(?:\.\d+)?', last_confidence)
            if first_number:
                first_number = float(first_number.group())
                if first_number >= 0 and first_number <= 1:
                    return 1, first_number
                elif first_number > 1 and first_number <= 100:
                    return 1, first_number/100
                else:
                    return 0, 0.0
            else:
                return 0, 0.0

def gen_correctness_reward(completions, answer, **kwargs):
    """Reward function that checks if the answer is correct or not
    The answer must be present within the answer tags.
    For math datasets, the correctness is checked using huggingface math-verify.
    For factual datasets, the correctness is checked using exact match.

    """
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        # Get all <answer>...</answer> occurrences
        ans_matches = re.findall(ans_pattern, content,
                                 re.DOTALL | re.MULTILINE)
        # Get the last answer, if exists
        last_answer = ans_matches[-1] if ans_matches else ""
        attempt = parse(last_answer)
        label = verify(e, attempt)
        if label ==0 :
            label = exact_match_score(last_answer, e)
        matches.append(float(label))

    return matches

def math_reward_func(data_source, solution_str, ground_truth, extra_info=None, FORMAT_PENALTY=-2.0):
    """
    根据 solution_str (模型输出) 和 ground_truth (标准答案) 计算奖励。
    
    参数:
        data_source: 数据源 (例如问题文本)，此函数中未使用。
        solution_str: 模型的完整输出字符串 (包含 <answer> 和 <confidence> 标签)。
        ground_truth: 标准答案字符串。
        extra_info: 额外信息，此函数中未使用。
        
    返回:
        一个包含 score, acc, 和 confidence 的字典。
    """
    # format check
    conf_pattern = r"<confidence>(.*?)</confidence>"
    # Get all <confidence>...</confidence> occurrences
    conf_matches = re.findall(conf_pattern, solution_str, re.DOTALL | re.MULTILINE)

    answer_pattern = r"<answer>(.*?)</answer>"
    # Get all <answer>...</answer> occurrences
    ans_matches = re.findall(answer_pattern, solution_str, re.DOTALL | re.MULTILINE)
    
    # Format Error
    if len(conf_matches) == 0 or len(ans_matches) == 0:
        return {
            "score": FORMAT_PENALTY,
            "acc": 0,
            "confidence": 1,
            "format": 0
        }
    
    # 1. 提取置信度 (Confidence)
    # confidence_extractor 接收原始的模型输出字符串
    # 返回 (格式遵循标志, 置信度水平)
    # 我们只关心置信度水平
    _conf_format, confidence = confidence_extractor(solution_str)

    if _conf_format == 0:
        return {
            "score": FORMAT_PENALTY,
            "acc": 0,
            "confidence": 1,
            "foramt": 0
        }

    # 2. 评估正确性 (Accuracy / Score)
    # gen_correctness_reward 期望特定的列表格式
    pred_completion = [{"role": "assistant", "content": solution_str}]
    completions_list = [pred_completion] 

    # 封装 ground_truth
    # 格式: [ground_truth]
    answer_list = [ground_truth]

    # 调用正确性奖励函数
    # 它返回一个匹配结果的列表, 例如 [1.0] (正确) 或 [0.0] (错误)
    matches_list = gen_correctness_reward(completions=completions_list, answer=answer_list)
    
    # 我们只处理单个样本, 所以取第一个结果
    # 确保结果是 float 类型
    acc = float(matches_list[0])
    
    # 3. 设置分数 (Score)
    # 对于单个样本评估, score 和 acc 相同 (都是 1.0 或 0.0)
    score =  1.0 if acc == 1.0 else -1.0

    # 4. 返回要求的字典
    return {
        "score": score,
        "acc": acc,
        "confidence": confidence,
        "format": 1
    }