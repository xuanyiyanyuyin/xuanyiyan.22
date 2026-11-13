import sys; sys.path.insert(0,r'D:/ai')
import gen_image as gi
paths = gi.generate_batch('a red apple on a table', steps=20, count=1, size=512, height=512)
print('RESULT:'+repr(paths))
gi.free_pipe()
