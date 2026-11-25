"""Choose a DFlash2 vocabulary from the user's prose in a single conversation."""
from __future__ import annotations

import re
import unicodedata


# Tool results can also arrive as text in an agent's user turn. Ignore them and
# quoted source material before counting scripts or looking for French words.
_NON_PROSE = re.compile(
    r"<(tool_response|tool_result|system-reminder|environment_context)\b[^<>]*>.*?(?:</\1>|\Z)"
    r"|```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`"
    r"|^[ \t]*(?:def\s+\w+\(|from\s+[\w.]+\s+import\s|(?:let|const|var)\s+\w+\s*=|#include\s|(?:print|console\.log)\()[^\n]*"
    r"|^[ \t]{0,3}>[^\n]*|https?://\S+",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
_HANGUL = re.compile(r"[\u1100-\u11ff\u3131-\u318e\ua960-\ua97f\uac00-\ud7ff]")
_KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u31f0-\u31ff\uff66-\uff9d]")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ee5f\U00030000-\U000323af]")
_BOPOMOFO = re.compile(r"[\u3105-\u312f\u31a0-\u31bf]")
_WORD = re.compile(r"[a-zà-öø-ÿœ]+")
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」|『[^』\n]*』')
_FRENCH = frozenset("""
    alors au aux avec bonjour pourquoi comment combien cette cet ces chaque dans des deux doit du elle
    elles encore entre est et explique expliquez expliquer fais faire faut fonction grâce il ils je
    la le laquelle lequel les leur leurs ma mais me merci mes moi mon ne nous obtenir on ou où par
    pas peut peux pouvez pour pourrais pourrait présenter propose proposez qu quand que quel quelle
    quelles quels qui résume résumer rédige réponds réponse sans se ses si soit sont son sous suis
    salut sur suivant suivante ton traduire traduis tu un une utiliser vous votre vos vrai écris écrire
""".split())
_ENGLISH = frozenset("""
    a an and are as at be can could did do does explain for from give hello how i in into is it me
    my of on or please should summarize that the their them then there these this to what when where
    which why will with without would write you your
""".split())
_FOLLOWUP = frozenset(("ok", "okay", "continue", "go on"))


def _prose_language(text: str) -> str | None:
    # Bound all normalization and markup scanning on files pasted into a user turn.
    if len(text) > 8192:
        text = text[:4096] + "\n" + text[-4096:]
    text = _NON_PROSE.sub(" ", unicodedata.normalize("NFKC", text))
    unquoted = _QUOTED.sub(" ", text)
    if any(c.isalpha() for c in unquoted):
        text = unquoted
    text = text.lower()
    korean = len(_HANGUL.findall(text))
    kana = len(_KANA.findall(text))
    han = len(_HAN.findall(text))
    bopomofo = len(_BOPOMOFO.findall(text))
    words = _WORD.findall(text)
    french = sum(w in _FRENCH for w in words)
    english = sum(w in _ENGLISH for w in words)
    if re.search(r"[àâçéèêëîïôùûüÿœ]", text):
        french += 1
    # Japanese normally includes Han characters together with kana. Korean can
    # also include Han alongside Hangul. English technical identifiers provide
    # no vote, so e.g. "Python asyncio 사용법" still selects Korean.
    if kana or korean or han or bopomofo:
        if korean >= kana and korean:
            script, weight = "ko", korean
        elif kana:
            script, weight = "ja", kana + han
        else:
            script, weight = "zh", han + bopomofo
        if weight >= max(english, french):
            return script
    if french > english:
        return "fr"
    if words and " ".join(words) not in _FOLLOWUP:
        return "en"
    return None


def detect_language(messages: list[dict]) -> str:
    """Return en, ko, ja, zh or fr. A code-only or neutral turn inherits the last user language."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if isinstance(part, dict)
                                and part.get("type") in ("text", "input_text", None)
                                and isinstance(part.get("text"), str))
        if isinstance(content, str):
            language = _prose_language(content)
            if language:
                return language
    return "en"
