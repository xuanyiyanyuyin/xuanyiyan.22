"""自动注册 lingbotvla 到 transformers AutoConfig/AutoModel（2026-10-14 加，仅新文件，不改训练代码）"""
try:
    from transformers import AutoConfig, AutoModel
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    try:
        AutoConfig.register("lingbotvla", LingbotVLAV2Config)
    except Exception:
        pass
    try:
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
        AutoModel.register(LingbotVLAV2Config, LingbotVlaV2Policy)
    except Exception:
        pass
    try:
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla import LingbotVlaPolicy
        AutoModel.register(LingbotVLAV2Config, LingbotVlaPolicy)
    except Exception:
        pass
except Exception:
    pass