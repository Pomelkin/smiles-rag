system_prompt = "Ты - оценщик ответов."

user_prompt = """Тебе даны вопрос, правильный ответ и ответ пользователя.
Твоя задача - сравнить ответ пользователя с правильным ответом и написать, правильно ли пользователь ответил на вопрос.

ВОПРОС:
{0}

ПРАВИЛЬНЫЙ ОТВЕТ:
{1}

ОТВЕТ ПОЛЬЗОВАТЕЛЯ:
{2}

Напиши в качестве оценки "+", если ответ правильный, или "-", если ответ неправильный. Никогда не пиши ничего кроме "+" или "-".

ОЦЕНКА:
"""