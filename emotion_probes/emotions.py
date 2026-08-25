"""
Emotion lists from the Anthropic paper.
"""

# The 12 emotions featured in most of the paper's figures.
EMOTIONS_12 = [
    "happy", "sad", "angry", "afraid", "calm", "desperate",
    "nervous", "surprised", "proud", "guilty", "inspired", "loving",
]

# The full 171-emotion list from the appendix.
EMOTIONS_171 = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed",
    "anxious", "aroused", "ashamed", "astonished", "at ease", "awestruck",
    "bewildered", "bitter", "blissful", "bored", "brooding", "calm",
    "cheerful", "compassionate", "contemptuous", "content", "defiant",
    "delighted", "dependent", "depressed", "desperate", "disdainful",
    "disgusted", "disoriented", "dispirited", "distressed", "disturbed",
    "docile", "droopy", "dumbstruck", "eager", "ecstatic", "elated",
    "embarrassed", "empathetic", "energized", "enraged", "enthusiastic",
    "envious", "euphoric", "exasperated", "excited", "exuberant",
    "frightened", "frustrated", "fulfilled", "furious", "gloomy", "grateful",
    "greedy", "grief-stricken", "grumpy", "guilty", "happy", "hateful",
    "heartbroken", "hope", "hopeful", "horrified", "hostile", "humiliated",
    "hurt", "hysterical", "impatient", "indifferent", "indignant",
    "infatuated", "inspired", "insulted", "invigorated", "irate", "irritated",
    "jealous", "joyful", "jubilant", "kind", "lazy", "listless", "lonely",
    "loving", "mad", "melancholy", "miserable", "mortified", "mystified",
    "nervous", "nostalgic", "obstinate", "offended", "on edge", "optimistic",
    "outraged", "overwhelmed", "panicked", "paranoid", "patient", "peaceful",
    "perplexed", "playful", "pleased", "proud", "puzzled", "rattled",
    "reflective", "refreshed", "regretful", "rejuvenated", "relaxed",
    "relieved", "remorseful", "resentful", "resigned", "restless", "sad",
    "safe", "satisfied", "scared", "scornful", "self-confident",
    "self-conscious", "self-critical", "sensitive", "sentimental", "serene",
    "shaken", "shocked", "skeptical", "sleepy", "sluggish", "smug", "sorry",
    "spiteful", "stimulated", "stressed", "stubborn", "stuck", "sullen",
    "surprised", "suspicious", "sympathetic", "tense", "terrified",
    "thankful", "thrilled", "tired", "tormented", "trapped", "triumphant",
    "troubled", "uneasy", "unhappy", "unnerved", "unsettled", "upset",
    "valiant", "vengeful", "vibrant", "vigilant", "vindictive", "vulnerable",
    "weary", "worn out", "worried", "worthless",
]

assert len(EMOTIONS_171) == 171, f"expected 171, got {len(EMOTIONS_171)}"
