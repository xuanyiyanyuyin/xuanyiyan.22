"""LTX-Video download - ModelScope"""
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Lightricks/LTX-Video', cache_dir=r'D:\ai\models', local_dir=r'D:\ai\models\LTX-Video')
print('Done.')
