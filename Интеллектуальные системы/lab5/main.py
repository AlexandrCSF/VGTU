import random

# Правила для приветствия
HELLO_RULE = {
    'inputs': ['привет', 'здравствуй', 'добрый день', 'добрый вечер'],
    'outputs': ['привет!', 'здравствуйте!', 'добрый день!', 'добрый вечер!'],
}

# Правила для прощания
GOODBYE_RULE = {
    'inputs': ['пока', 'до свидания', 'всего хорошего', 'до встречи'],
    'outputs': ['пока!', 'до свидания!', 'всего хорошего!', 'до встречи!'],
}

# Правила для негативных фраз
NEGATIVE_RULE = {
    'inputs': ['редиска', 'дурак', 'глупый'],
    'outputs': ['Диалог закончен.'],
}

# Основные правила для диалога
RULES = [
    {
        'inputs': ['как дела', 'как ты', 'как жизнь'],
        'outputs': ['отлично!', 'прекрасно!', 'хорошо!', 'лучше всех!'],
    },
    {
        'inputs': ['чем занимаешься', 'что делаешь', 'чем занят'],
        'outputs': ['читаю книгу.', 'слушаю музыку.', 'обедаю.', 'прогуливаюсь.'],
    },
    {
        'inputs': ['как прошел твой день', 'как день', 'как прошел день'],
        'outputs': ['интересно!', 'насыщенно!', 'хорошо!', 'спокойно.'],
    },
    {
        'inputs': ['какие планы на вечер', 'что планируешь вечером', 'чем займешься вечером'],
        'outputs': ['посмотрю фильм.', 'почитаю книгу.', 'встречусь с друзьями.', 'отдохну.'],
    },
]

# Общие фразы для неизвестных вопросов
GENERAL_PHRASES = [
    'извини, я не хотел бы говорить об этом.',
    'если ты не возражаешь, давай сменим тему.',
    'это секретная информация.',
    'интересный вопрос, но я не знаю, что ответить.',
]

def main():
    print("Чат-бот: Привет! Давай пообщаемся.")
    rules_hits = []  # Список для отслеживания использованных правил

    while True:
        phrase = input("Вы: ").strip().lower()

        # Убираем знаки препинания
        for char in ['.', ',', '-', '?', '!']:
            phrase = phrase.replace(char, '')

        # Проверяем на прощание
        if phrase in GOODBYE_RULE['inputs']:
            print("Чат-бот:", random.choice(GOODBYE_RULE['outputs']))
            break

        # Проверяем на негативные фразы
        if phrase in NEGATIVE_RULE['inputs']:
            print("Чат-бот:", random.choice(NEGATIVE_RULE['outputs']))
            break

        # Проверяем на приветствие
        if phrase in HELLO_RULE['inputs']:
            print("Чат-бот:", random.choice(HELLO_RULE['outputs']))
            continue

        # Проверяем основные правила
        found = False
        for index, rule in enumerate(RULES):
            if phrase in rule['inputs']:
                if index in rules_hits:
                    print("Чат-бот: Мы это уже обсуждали.")
                    found = True
                    break
                rules_hits.append(index)
                print("Чат-бот:", random.choice(rule['outputs']))
                found = True
                break

        # Если фраза не найдена в правилах
        if not found:
            print("Чат-бот:", random.choice(GENERAL_PHRASES))

if __name__ == "__main__":
    main()
