"""
P3 属性包 — 每个词的完整属性描述
多标签、分层、可组合、带置信度
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class AttributePacket:
    """一个词的完整属性包 — P3栈的最终输出"""
    word: str

    # P3-A: 基础词类 (带置信度)
    basic_type: str = ""            # noun/verb/pronoun/question/place/time/manner/function
    basic_confidence: float = 0.0

    # P3-S: 语义族 (可多标签)
    semantic_types: List[Tuple[str, float]] = field(default_factory=list)
    # 如 [("食物", 0.9), ("物体", 0.3)]

    # P3-P: 句法位置候选 (可多标签)
    syntax_candidates: List[Tuple[str, float]] = field(default_factory=list)
    # 如 [("主语", 0.9), ("宾语", 0.3)]

    # P3-Q: 疑问槽属性
    question_slot: str = ""         # subject/object/place/manner/reason/none
    question_confidence: float = 0.0

    # P3-Role: 人称属性
    person: str = ""                # first/second/third/none
    person_confidence: float = 0.0
    number: str = ""                # singular/plural/none

    # P3-E: 情感属性 (新增)
    polarity: str = ""              # positive/negative/neutral
    emotion_intensity: float = 0.0  # 0~1
    emotion_category: str = ""      # affection/joy/anger/sadness/fear/dislike/praise/hope/help/none

    # P3-D: 动作方向
    direction: str = ""             # outward/inward/mutual/cognitive/unknown
    direction_target: str = ""      # object/self/both/internal/unknown

    # P3-T: 时态体态 (新增v2)
    tense_type: str = ""            # past/present/future/timeless
    aspect_type: str = ""           # completive/progressive/inchoative/...
    time_reference: str = ""        # specific_time/relative_time/...
    tense_confidence: float = 0.0

    # P3-N: 数量程度 (新增v2)
    is_number: bool = False
    is_measure: bool = False
    is_degree: bool = False
    degree_level: str = ""          # high/medium/low/excess
    is_quantifier: bool = False

    # P3-C: 连接逻辑 (新增v2)
    is_connective: bool = False
    conn_type: str = ""             # cause/adversative/coordinate/conditional/...
    conn_confidence: float = 0.85

    # P3-V: 语态语气 (新增v2)
    voice_type: str = ""            # passive/disposal
    mood_type: str = ""             # suggestion/inquiry/exclamation/...
    modal_type: str = ""            # ability/permission/obligation/...

    # P3-M: 修饰限定 (新增v2)
    is_modifier: bool = False
    mod_type: str = ""              # adjective/adverb_manner/scope
    mod_confidence: float = 0.75
    is_de: bool = False             # 的/地/得标记
    de_type: str = ""               # attributive/adverbial/complement
    is_comparative: bool = False
    is_comparative: bool = False

    # P3-X: 代码技术 (新增v2)
    is_code: bool = False
    code_type: str = ""             # keyword/identifier/operator/literal/...
    code_language: str = ""         # python/javascript/sql/...
    code_role: str = ""             # declaration/control_flow/function_call/...
    cn_tech: bool = False           # 中文技术词汇

    # P3-Punct: 标点符号 (新增v2)
    is_punct: bool = False
    punct_type: str = ""            # period/comma/semicolon/colon/quote/...
    punct_role: str = ""            # statement/clause_boundary/explanation/...
    is_sent_boundary: bool = False

    # P3-Phrase: 词组级属性 (同一词组内所有词共享)
    phrase_role: str = ""           # NP/VP/PP/subject_phrase/predicate_phrase/object_phrase
    phrase_type: str = ""           # action_phrase/entity_phrase/query_phrase/answer_phrase
    phrase_sem: str = ""            # 词组语义: concrete/abstract/relational/modal

    # P3-Sent: 句子级属性 (新增 — 整句所有词共享)
    sent_type: str = ""             # question/statement/command/exclamation
    sent_domain: str = ""           # person/location/manner/time/reason/general
    sent_mood: str = ""             # formal/casual/informal/neutral
    sent_tense: str = ""            # past/present/future/none

    # 兼容旧P3的单一属性
    primary_attr: str = ""          # 主语/谓语/宾语/定语/状语/补语/虚词/?

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "basic_type": (self.basic_type, self.basic_confidence),
            "semantic_types": self.semantic_types,
            "syntax_candidates": self.syntax_candidates,
            "question_slot": (self.question_slot, self.question_confidence),
            "person": (self.person, self.person_confidence),
            "number": self.number,
            "polarity": (self.polarity, self.emotion_intensity, self.emotion_category),
            "direction": (self.direction, self.direction_target),
            "primary_attr": self.primary_attr,
            "conn_type": self.conn_type,
            "conn_confidence": self.conn_confidence if hasattr(self,'conn_confidence') else 0.85,
            "is_connective": self.is_connective,
            "mod_type": self.mod_type,
            "mod_confidence": self.mod_confidence if hasattr(self,'mod_confidence') else 0.75,
            "is_modifier": self.is_modifier,
            "is_comparative": self.is_comparative if hasattr(self,'is_comparative') else False,
        }

    def best_syntax(self) -> str:
        """返回最可能的句法角色"""
        if not self.syntax_candidates:
            return self.primary_attr
        return max(self.syntax_candidates, key=lambda x: x[1])[0]

    def best_semantic(self) -> str:
        """返回最可能的语义类型"""
        if not self.semantic_types:
            return ""
        return max(self.semantic_types, key=lambda x: x[1])[0]

    def is_question_word(self) -> bool:
        return self.question_slot not in ("", "none")

    def is_pronoun(self) -> bool:
        return self.basic_type == "pronoun"

    def __repr__(self):
        parts = [f"P3[{self.word}]"]
        if self.basic_type:
            parts.append(f"type={self.basic_type}")
        if self.person and self.person != "none":
            parts.append(f"person={self.person}")
        if self.polarity and self.polarity != "neutral":
            parts.append(f"{self.polarity}({self.emotion_intensity:.1f})")
        if self.direction and self.direction != "unknown":
            parts.append(f"dir={self.direction}")
        if self.question_slot and self.question_slot != "none":
            parts.append(f"Q={self.question_slot}")
        if self.syntax_candidates:
            top = self.syntax_candidates[:2]
            parts.append(f"syn={top}")
        return " ".join(parts)
