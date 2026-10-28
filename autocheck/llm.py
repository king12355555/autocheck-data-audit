# -*- coding: utf-8 -*-
"""LLM 报告生成(OpenAI 兼容接口, 支持 DeepSeek / 本地 Ollama / Claude 网关)。

降级策略: 未配置 API Key 或调用失败时返回 None, 由报告层自动回退到模板结论,
保证“断网也能出报告”。
"""
import os


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def llm_settings():
    load_env()
    return {
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    }


SYSTEM_PROMPT = """你是一名资深的风控/成本方向数据分析师, 擅长赔付(退款补偿)业务的波动归因、异常识别与规则迭代。
你只会收到一份“机器计算好的事实清单”(facts), 里面每个数字都经过了确定性代码核算, 可信但只代表工具已完成的检查。
要求:
1. 只能引用事实清单中出现的数字、日期和结论, 严禁自行计算或编造任何新数字;
2. 输出一份面向业务方的《结论与建议》markdown 文本, 结构为:
   - 总体判断(2-3句)
   - 重点问题(编号列表, 每条必须带事实清单中的证据数字)
   - 可能原因与下一步建议(按优先级)
   - 口径与局限(一句话)
3. 语言: 中文, 简洁、可执行, 不要使用 emoji。
4. 硬性规则: 事实清单开头的“检查结果总览”会说明有几个检查项告警/异常。
   只要存在任何 [error] 或 [warn] 条目, 你的总体判断就必须明确点出问题;
   严禁出现“一切正常”“数据无明显异常”“整体健康”等表述。
   只有所有检查项都通过(总览为 0 个告警/异常)时, 才可如实表述为无明显异常。"""


def generate_conclusion(facts_text, settings=None):
    settings = settings or llm_settings()
    if not settings.get("api_key"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"事实清单如下:\n\n{facts_text}"},
            ],
            temperature=0.3,
            max_tokens=6000,  # 推理型模型会先消耗推理 token, 预算给足避免返回空
        )
        content = resp.choices[0].message.content
        return content.strip() if content and content.strip() else None
    except Exception:
        return None  # 调用失败 → 模板降级
