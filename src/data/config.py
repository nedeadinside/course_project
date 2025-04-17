MMLU_INSTRUCTION_TEMPLATE = """You are presented with a multiple-choice question on {subject}.

QUESTION:
{text}

ANSWER OPTIONS:
{options}

INSTRUCTIONS:
- Provide your answer as a single letter in parentheses: (X)
- Select only one correct answer
- Do not include explanations or additional text

The answer is:"""

RUSSIAN_SUMMARIZATION_TEMPLATE = """Тебе предоставлена статья с заголовком {title}. Твоя задача - написать краткое содержание этой статьи в несколько предложений.

ТЕКСТ СТАТЬИ:
{text}

КРАТКОЕ СОДЕРЖАНИЕ:"""

ENGLISH_SUMMARIZATION_TEMPLATE = """You are provided with an article with the title {title}. Your task is to write a concise summary of this article in a few sentences.

ARTICLE TEXT:
{text}

SUMMARY:"""