from __future__ import annotations

import random
import string


def random_string(length: int) -> str:
    characters = ' ' + string.ascii_letters + string.digits
    weights = [20] + [1 for _ in characters[1:]]
    return ''.join(random.choices(characters, weights, k=length))


def random_document(sentences: int) -> str:
    lines: list[str] = []
    for _ in range(sentences):
        # 5-20 words per sentence, 5 characters per word
        length = random.randint(5 * 5, 20 * 5)
        line = random_string(length).strip()
        lines.append(f'{line}.')

    return '\n'.join(lines)
