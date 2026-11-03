"""Small, reproducible LongBench pilot. Run with the Ollama service available."""
import argparse, collections, json, math, random, re, statistics, string, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import make_chunks, embed, cosine, evaluate_boundary, ollama_post


def tokens(text): return re.findall(r'\w+', text.lower())
def bm25(question, chunks):
    docs = [tokens(c['text']) for c in chunks]; q = tokens(question)
    avg = sum(map(len,docs))/len(docs); counts = collections.Counter(t for doc in docs for t in set(doc))
    scores=[]
    for doc in docs:
        freq=collections.Counter(doc); score=0
        for t in q:
            n=counts[t]; idf=math.log(1+(len(docs)-n+0.5)/(n+0.5))
            score += idf*freq[t]*2.5/(freq[t]+1.5*(1-0.75+0.75*len(doc)/avg)) if freq[t] else 0
        scores.append(score)
    return scores

def answer(question, context):
    response=ollama_post('/api/chat', {'model':'llama3.2', 'stream':False, 'options':{'temperature':0, 'num_ctx':8192, 'num_predict':100}, 'messages':[
        {'role':'system','content':'Answer the question concisely using only the provided text. Return only the answer. If the answer is absent, return NOT_FOUND.'},
        {'role':'user','content':f'Text:\n{context}\n\nQuestion: {question}'}]})
    return response['message']['content'].strip()

def paper_rber(question, text):
    # Same partitioning, 400-word boundary views, boolean confidence and early stopping as res.py.
    words=text.split(); calls=0; viewed=0
    for depth in range(9):
        if len(words)<=500 or depth>=8:
            calls+=1; viewed+=len(words)
            return evaluate_boundary(question,words) or 'NOT_FOUND', calls, viewed
        third=len(words)//3
        segments=[words[:third], words[2*third:]]
        for segment in segments:
            view=segment if len(segment)<=800 else segment[:400]+segment[-400:]
            calls+=1; viewed+=len(view)
            result=evaluate_boundary(question,view)
            if result:
                return result,calls,viewed
        words=words[third:2*third]
    raise AssertionError()

def normalize(s):
    s=s.lower(); s=''.join(ch for ch in s if ch not in string.punctuation); s=re.sub(r'\b(a|an|the)\b',' ',s)
    return ' '.join(s.split())
def f1(pred, gold):
    a=normalize(pred).split(); b=normalize(gold).split(); common=sum((collections.Counter(a)&collections.Counter(b)).values())
    return 2*common/(len(a)+len(b)) if a and b else float(a==b)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('data'); parser.add_argument('--n',type=int,default=10); parser.add_argument('--seed',type=int,default=42); parser.add_argument('--out',default='benchmark/results.json'); args=parser.parse_args()
    all_data=[json.loads(line) for line in open(args.data)]
    eligible=[d for d in all_data if len(d['context'].split())>=2000]
    sample=random.Random(args.seed).sample(eligible,min(args.n,len(eligible)))
    out={'dataset':'LongBench MultiFieldQA-en mirror','seed':args.seed,'eligible':len(eligible),'sample_size':len(sample),'results':[]}
    for number,d in enumerate(sample,1):
        q=d['input']; raw,_=make_chunks([(None,d['context'])]); chunks=[]
        vectors=[]
        for i in range(0,len(raw),16): vectors.extend(embed([c['text'] for c in raw[i:i+16]]))
        for i,(c,v) in enumerate(zip(raw,vectors)): chunks.append({**c,'idx':i,'document_id':'doc','embedding':json.dumps(v)})
        qv=embed([q],query=True)[0]
        for chunk in chunks: chunk['semantic']=cosine(qv,json.loads(chunk['embedding']))
        lexical=bm25(q,chunks)
        methods={
            'bm25':sorted(range(len(chunks)), key=lambda i:lexical[i],reverse=True)[:8],
            'dense':sorted(range(len(chunks)), key=lambda i:chunks[i]['semantic'],reverse=True)[:8],
            'hybrid':sorted(range(len(chunks)), key=lambda i: 0.8 * chunks[i]['semantic'] + 0.2 * (sum(1 for t in set(tokens(q)) if t in set(tokens(chunks[i]['text']))) / max(1,len(set(tokens(q))))),reverse=True)[:8],
        }
        row={'id':d['_id'],'words':len(d['context'].split()),'question':q,'gold':d['answers'],'methods':{}}
        for method,indices in methods.items():
            selected=[chunks[i] for i in sorted(indices)]; context='\n\n'.join(c['text'] for c in selected)
            started=time.monotonic(); pred=answer(q,context); elapsed=time.monotonic()-started
            row['methods'][method]={'answer':pred,'f1':max(f1(pred,g) for g in d['answers']), 'answer_string_in_context':any(normalize(g) in normalize(context) for g in d['answers'] if normalize(g)), 'generator_seconds':elapsed,'context_words':sum(len(c['text'].split()) for c in selected),'calls':1}
        started=time.monotonic(); pred,calls,words=paper_rber(q,d['context']); elapsed=time.monotonic()-started
        row['methods']['original_rber']={'answer':pred,'f1':max(f1(pred,g) for g in d['answers']), 'generator_seconds':elapsed,'context_words':words,'calls':calls}
        out['results'].append(row); Path(args.out).write_text(json.dumps(out,indent=2)); print(f'{number}/{len(sample)} {d["_id"]} ' + ' '.join(f"{m}:{v['f1']:.2f}" for m,v in row['methods'].items()),flush=True)
if __name__=='__main__': main()
