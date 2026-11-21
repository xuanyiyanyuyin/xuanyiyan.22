"""
P3-C: ContextCache 视角绑定
============================
P3-Role只判断语法人称(first/second/third)
ContextCache根据对话上下文绑定对话角色:
  我(用户侧) → USER, 我(系统侧) → SELF
  你(用户对系统说) → SELF, 你(系统回用户) → USER
不修改P3属性包, 作为附加层叠加角色信息
"""
from typing import Dict, List, Optional


class ContextCache:
    """视角绑定缓存: 语法人称→对话角色

    规则:
      用户输入时: 用户说的"我"=USER, 用户说的"你"=SELF
      系统输出时: 系统说的"我"=SELF, 系统说的"你"=USER

    第三人称(他/她/老师/学生等)不受影响 → OTHER
    """

    def __init__(self):
        # 当前对话角色
        self.speaker = "user"  # "user" or "system"
        self.user_name = "用户"
        self.system_name = "系统"

    def set_speaker(self, speaker: str):
        """切换当前说话者: 'user' 或 'system'"""
        self.speaker = speaker

    def bind(self, person: str, word: str = "") -> Dict:
        """语法人称→对话角色绑定

        输入: person='first'/'second'/'third'/'none'
        输出: {dialogue_role, confidence, note}

        dialogue_role:
          USER  — 指代用户
          SELF  — 指代当前说话者自己
          OTHER — 第三方, 不受视角影响
        """
        if person == "none":
            return {"dialogue_role": "none", "confidence": 0.0, "note": ""}

        if person == "first":
            # "我" = 说话者自己
            role = "USER" if self.speaker == "user" else "SELF"
            return {
                "dialogue_role": role,
                "confidence": 0.99,
                "note": f"第一人称, 当前说话者={self.speaker}, 绑定={role}"
            }

        if person == "second":
            # "你" = 听者
            role = "SELF" if self.speaker == "user" else "USER"
            return {
                "dialogue_role": role,
                "confidence": 0.99,
                "note": f"第二人称, 当前说话者={self.speaker}, 绑定={role}"
            }

        if person == "third":
            return {
                "dialogue_role": "OTHER",
                "confidence": 0.95,
                "note": "第三人称, 不受视角影响"
            }

        if person == "interrogative":
            return {
                "dialogue_role": "OTHER",
                "confidence": 0.90,
                "note": "疑问代词, 需要填充"
            }

        return {"dialogue_role": "OTHER", "confidence": 0.50, "note": f"未识别: {person}"}

    def process_packet(self, packet, speaker: str = None) -> Dict:
        """处理P3属性包, 附加角色绑定

        packet: AttributePacket (来自P3栈)
        返回: {dialogue_role, swap_target, confidence}
        """
        if speaker:
            self.set_speaker(speaker)

        result = self.bind(packet.person, packet.word)

        # 人称切换目标: 我↔你 的转换
        swap_map = {
            ("first", "user"): ("你", "second"),
            ("second", "user"): ("我", "first"),
            ("first", "system"): ("你", "second"),
            ("second", "system"): ("我", "first"),
        }
        swap_key = (packet.person, self.speaker)
        if swap_key in swap_map:
            result["swap_word"], result["swap_person"] = swap_map[swap_key]
        else:
            result["swap_word"] = packet.word
            result["swap_person"] = packet.person

        return result

    def process_sentence(self, packets: List, speaker: str = None) -> List[Dict]:
        """处理整句属性包列表"""
        return [self.process_packet(p, speaker) for p in packets]
