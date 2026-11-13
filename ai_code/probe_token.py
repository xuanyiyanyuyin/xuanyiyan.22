from transformers import AutoTokenizer
clip="D:/ai/models/FLUX.1-dev/tokenizer"
t5="D:/ai/models/FLUX.1-dev/tokenizer_2"
tok_c=AutoTokenizer.from_pretrained(clip,local_files_only=True)
tok_t=AutoTokenizer.from_pretrained(t5,local_files_only=True)
print('CLIP tokenizer max_len:', getattr(tok_c,"model_max_length","?"))
print('T5 tokenizer max_len:', getattr(tok_t,"model_max_length","?"))
tests=["a single beautiful girl, pink hair, elegant kimono",
       "一个好看的女生, 粉发, 优雅和服",
       "solo, single girl, vertical portrait, detailed eyes",
       "intense desire, sultry, chaotic tangled sheets, two bodies"]
for s in tests:
    c=len(tok_c.tokenize(s)); t=len(tok_t.tokenize(s))
    print(f'  [{s[:34]}] CLIP={c}tok T5={t}tok')
