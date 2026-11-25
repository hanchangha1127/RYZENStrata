"""Per-request language selection for the Ryzen DFlash2 head."""
import pytest

from strata_control.draft_language import detect_language


@pytest.mark.parametrize(("prompt", "expected"), [
    ("Explain GPU memory bandwidth in simple terms.", "en"),
    ("Compute x = y * m + c.", "en"),
    ("Can you translate \"안녕하세요\" into English?", "en"),
    ("Python asyncio gather 사용법을 알려줘", "ko"),
    ("\u1100\u1161\u11ab\u1103\u1161\u11ab\u1112\u1175 설명해 줘", "ko"),
    ("この Python 関数を説明してください。", "ja"),
    ("请告诉我如何优化这段 Python 函数。", "zh"),
    ("請用三個步驟解釋這個演算法。", "zh"),
    ("點解我嘅程式咁慢？", "zh"),
    ("ㄋㄧˇ ㄏㄠˇ", "zh"),
    ("Comment utiliser un dictionnaire Python pour compter les mots ?", "fr"),
    ("Où est l’erreur dans cette fonction ?", "fr"),
    ("Voici le code :\n```python\nprint('한국어')\n```\nPeux-tu le corriger ?", "fr"),
])
def test_prompt_language(prompt, expected):
    assert detect_language([{"role": "user", "content": prompt}]) == expected


def test_latest_user_prose_controls_each_turn():
    messages = [
        {"role": "system", "content": "You are a helpful English assistant."},
        {"role": "user", "content": "함수를 고쳐줘"},
        {"role": "assistant", "content": "Exemple en français"},
        {"role": "tool", "content": "日本語の結果です"},
        {"role": "user", "content": [
            {"type": "tool_result", "content": "中文執行結果"},
            {"type": "input_image", "url": "https://example.test/如何.png"},
            {"type": "input_text", "text": "OK"},
        ]},
    ]
    assert detect_language(messages) == "ko"
    messages.append({"role": "user", "content": [{"type": "text", "text": "What comes next?"}]})
    assert detect_language(messages) == "en"


def test_code_only_followup_keeps_previous_language():
    messages = [
        {"role": "user", "content": "Expliquez cette erreur."},
        {"role": "user", "content": "<tool_response>한국어 결과</tool_response>\nprint('你好')\n```js\n'日本語'\n```"},
    ]
    assert detect_language(messages) == "fr"
    assert detect_language([messages[-1]]) == "en"
    assert detect_language([]) == "en"


def test_long_attached_code_and_tool_output_do_not_hide_the_question():
    source = "const title = 'こんにちは';\n" * 10000
    for start, end in [("```js\n", "\n```"), ("<tool_response>", "</tool_response>")]:
        prompt = start + source + end + "\n이 문제의 원인을 알려줘."
        assert detect_language([{"role": "user", "content": prompt}]) == "ko"
