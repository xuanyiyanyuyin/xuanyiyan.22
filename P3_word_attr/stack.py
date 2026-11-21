"""
P3 属性栈 v2 — 14子模块, 覆盖中文全局
========================================
原始7: A词类 S语义 P句法 Q疑问 R人称 E情感 D方向
新增7: T时态 N数量 C连接 V语态 M修饰 X代码 Punct标点
"""
from typing import Dict, List, Optional
from .attr_packet import AttributePacket
from .p3a_basic import P3A_BasicType
from .p3s_semantic import P3S_SemanticType
from .p3p_syntax import P3P_SyntaxPosition
from .p3q_question import P3Q_QuestionSlot
from .p3r_person import P3R_PersonRole
from .p3e_emotion import P3E_Emotion
from .p3d_direction import P3D_Direction
from .p3t_tense import P3T_TenseAspect
from .p3n_number import P3N_NumberDegree
from .p3c_connect import P3C_ConnectLogic
from .p3v_voice import P3V_VoiceMood
from .p3m_modifier import P3M_Modifier
from .p3c_code import P3C_CodeDetect
from .p3p_punct import P3P_Punctuation


class P3AttributeStack:
    """P3属性栈v2 — 14个子模块全开"""

    def __init__(self):
        self.p3a = P3A_BasicType()
        self.p3s = P3S_SemanticType()
        self.p3p = P3P_SyntaxPosition()
        self.p3q = P3Q_QuestionSlot()
        self.p3r = P3R_PersonRole()
        self.p3e = P3E_Emotion()
        self.p3d = P3D_Direction()
        self.p3t = P3T_TenseAspect()
        self.p3n = P3N_NumberDegree()
        self.p3c = P3C_ConnectLogic()
        self.p3v = P3V_VoiceMood()
        self.p3m = P3M_Modifier()
        self.p3x = P3C_CodeDetect()
        self.p3punct = P3P_Punctuation()

    def process_word(self, word: str,
                     position: int = -1,
                     sentence_length: int = 0) -> AttributePacket:
        """处理单个词, 返回完整属性包"""
        packet = AttributePacket(word=word)

        context = {
            "position": position,
            "sentence_length": sentence_length,
        }

        # P3-A: 基础词类
        a_result = self.p3a.process(word, context)
        packet.basic_type = a_result.get("basic_type", "")
        packet.basic_confidence = a_result.get("confidence", 0.0)

        # P3-S: 语义族
        s_result = self.p3s.process(word, context)
        packet.semantic_types = s_result.get("semantic_types", [])

        # P3-P: 句法位置候选
        p_result = self.p3p.process(word, context)
        packet.syntax_candidates = p_result.get("syntax_candidates", [])

        # P3-Q: 疑问槽
        q_result = self.p3q.process(word, context)
        packet.question_slot = q_result.get("question_slot", "none")
        packet.question_confidence = q_result.get("confidence", 0.0)

        # P3-Role: 人称
        r_result = self.p3r.process(word, context)
        packet.person = r_result.get("person", "none")
        packet.person_confidence = r_result.get("confidence", 0.0)
        packet.number = r_result.get("number", "none")

        # P3-E: 情感极性+强度 (新增)
        e_result = self.p3e.process(word, context)
        packet.polarity = e_result.get("polarity", "neutral")
        packet.emotion_intensity = e_result.get("intensity", 0.0)
        packet.emotion_category = e_result.get("category", "none")

        # P3-D: 动作方向
        d_result = self.p3d.process(word, context)
        packet.direction = d_result.get("direction", "unknown")
        packet.direction_target = d_result.get("target_type", "unknown")

        # P3-T: 时态体态 (新增)
        t_result = self.p3t.process(word, context)
        packet.tense_type = t_result.get("tense_type", "")
        packet.aspect_type = t_result.get("aspect_type", "")
        packet.time_reference = t_result.get("time_reference", "")
        packet.tense_confidence = t_result.get("tense_confidence", 0.0)

        # P3-N: 数量程度 (新增)
        n_result = self.p3n.process(word, context)
        packet.is_number = n_result.get("is_number", False)
        packet.is_measure = n_result.get("is_measure", False)
        packet.is_degree = n_result.get("is_degree", False)
        packet.degree_level = n_result.get("degree_level", "")
        packet.is_quantifier = n_result.get("is_quantifier", False)

        # P3-C: 连接逻辑 (新增)
        c_result = self.p3c.process(word, context)
        packet.is_connective = c_result.get("is_connective", False)
        packet.conn_type = c_result.get("conn_type", "")
        packet.conn_confidence = c_result.get("conn_confidence", 0.85)

        # P3-V: 语态语气 (新增)
        v_result = self.p3v.process(word, context)
        packet.voice_type = v_result.get("voice_type", "")
        packet.mood_type = v_result.get("mood_type", "")
        packet.modal_type = v_result.get("modal_type", "")

        # P3-M: 修饰限定 (新增)
        m_result = self.p3m.process(word, context)
        packet.is_modifier = m_result.get("is_modifier", False)
        packet.mod_type = m_result.get("mod_type", "")
        packet.mod_confidence = m_result.get("mod_confidence", 0.75)
        packet.is_de = m_result.get("is_de", False)
        packet.de_type = m_result.get("de_type", "")
        packet.is_comparative = m_result.get("is_comparative", False)

        # P3-X: 代码技术 (新增)
        x_result = self.p3x.process(word, context)
        packet.is_code = x_result.get("is_code", False)
        packet.code_type = x_result.get("code_type", "")
        packet.code_language = x_result.get("language", "")
        packet.code_role = x_result.get("code_role", "")
        packet.cn_tech = x_result.get("cn_tech", False)

        # P3-Punct: 标点符号 (新增)
        punct_result = self.p3punct.process(word, context)
        packet.is_punct = punct_result.get("is_punct", False)
        packet.punct_type = punct_result.get("punct_type", "")
        packet.punct_role = punct_result.get("punct_role", "")
        packet.is_sent_boundary = punct_result.get("is_sent_boundary", False)

        # 推导 primary_attr (兼容旧P3接口)
        packet.primary_attr = self._derive_primary_attr(packet)

        return packet

    def process_sentence(self, words: List[str]) -> List[AttributePacket]:
        """处理整句, 先拼单字成词, 再属性分析"""
        # 单字合并: 收集所有P3子模块已知的词
        all_known = set()
        for mod in [self.p3a, self.p3s]:
            for attr in dir(mod):
                val = getattr(mod, attr)
                if isinstance(val, set):
                    all_known.update(val)
        # 扫描合并: 最长匹配优先
        merged = []
        i = 0
        while i < len(words):
            best_len = 1
            for l in range(min(4, len(words)-i), 1, -1):  # 最长4字
                candidate = ''.join(words[i:i+l])
                if candidate in all_known or any(candidate in s for s in all_known if len(s)==l and candidate==s):
                    best_len = l; break
            merged.append(''.join(words[i:i+best_len]))
            i += best_len
        words = merged
        n = len(words)
        packets = []
        for i, word in enumerate(words):
            packet = self.process_word(word, position=i, sentence_length=n)
            packets.append(packet)
        # 自动推断词组级属性: 相邻同角色词归组
        self._infer_phrase_attrs(packets, words)
        # 自动推断句级属性
        self._infer_sentence_attrs(packets, words)
        return packets

    def _infer_phrase_attrs(self, packets, words):
        """自动推断词组级属性: 按句法角色分组"""
        n = len(packets)
        if n < 2:
            return
        i = 0
        while i < n:
            role = packets[i].best_syntax()
            j = i + 1
            while j < n and packets[j].best_syntax() == role:
                j += 1
            if j - i >= 2:  # 连续2+词同角色 → 词组
                phrase_type = self._infer_phrase_type(packets[i:j])
                for k in range(i, j):
                    packets[k].phrase_role = role
                    packets[k].phrase_type = phrase_type
            i = j

    def _infer_phrase_type(self, group_packets):
        """从词组中推断短语类型"""
        types = [p.basic_type for p in group_packets if p.basic_type]
        sem_roles = [p.best_semantic() for p in group_packets if p.best_semantic()]
        if any("question" in t for t in types):
            return "query_phrase"
        if any(t in ("noun_place", "noun_time") for t in types):
            return "entity_phrase"
        if any(s in ("行为", "状态") for s in sem_roles):
            return "action_phrase"
        if any(t in ("verb", "function") for t in types):
            return "relational_phrase"
        return "entity_phrase"

    def _infer_sentence_attrs(self, packets, words):
        """自动推断句子级属性"""
        has_question = False
        has_neg = False
        polarities = []
        question_domains = set()
        for p in packets:
            if p.is_question_word():
                has_question = True
                qs = p.question_slot
                if qs == "subject" or "person" in (p.basic_type or ""):
                    question_domains.add("person")
                elif qs == "place" or "place" in (p.basic_type or ""):
                    question_domains.add("location")
                elif qs == "manner":
                    question_domains.add("manner")
                elif qs == "object" or "thing" in (p.basic_type or ""):
                    question_domains.add("thing")
            if p.polarity:
                polarities.append(p.polarity)
            if p.basic_type and "negative" in p.basic_type:
                has_neg = True

        # 句类型
        if has_question:
            sent_type = "question"
        elif has_neg:
            sent_type = "negation"
        else:
            sent_type = "statement"

        # 句领域
        if question_domains:
            sent_domain = list(question_domains)[0]  # 取第一个匹配
        else:
            sent_domain = "general"

        # 句情感基调
        if polarities:
            pos_count = sum(1 for p in polarities if p == "positive")
            neg_count = sum(1 for p in polarities if p == "negative")
            if pos_count > neg_count:
                sent_mood = "positive"
            elif neg_count > pos_count:
                sent_mood = "negative"
            else:
                sent_mood = "neutral"
        else:
            sent_mood = "neutral"

        # 写入所有词的句级属性
        for p in packets:
            p.sent_type = sent_type
            p.sent_domain = sent_domain
            p.sent_mood = sent_mood

    def _derive_primary_attr(self, packet: AttributePacket) -> str:
        """从属性包推导主要的句法属性 (兼容旧接口)"""
        # 疑问词→根据槽位映射
        if packet.question_slot == "subject":
            return "主语"
        if packet.question_slot in ("object", "place"):
            return "宾语"
        if packet.question_slot in ("manner", "reason"):
            return "状语"

        # 从syntax_candidates取最高置信度
        if packet.syntax_candidates:
            return packet.syntax_candidates[0][0]

        # 从basic_type推断
        bt = packet.basic_type
        if "question" in bt:
            return "宾语"
        if bt == "pronoun":
            return "主语"
        if bt in ("noun_place", "noun_time"):
            return "宾语"
        if bt == "function":
            return "虚词"

        return "?"
