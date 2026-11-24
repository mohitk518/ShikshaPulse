"""ShikshaPulse risk model: trains logistic regression on synthetic school data, exports weights to model.json"""
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(42); N=6000
att=np.clip(rng.normal(.82,.12,N),.2,1); adrop=np.clip(rng.exponential(.05,N),0,.6)
marks=np.clip(rng.normal(62,17,N),10,98); mdrop=rng.normal(2,7,N)
hw=np.clip(rng.normal(.7,.22,N),0,1); fee=np.where(rng.random(N)<.15,rng.integers(15,90,N),0)
z=-3.6+ -5*(att-.8)+9*adrop-.035*(marks-60)+.07*mdrop-2.2*(hw-.7)+.012*fee+rng.normal(0,.9,N)
y=(rng.random(N)<1/(1+np.exp(-z))).astype(int)
X=np.c_[att,adrop,marks,mdrop,hw,fee]; names=["att","adrop","marks","mdrop","hw","fee"]
mu,sd=X.mean(0),X.std(0)
Xtr,Xte,ytr,yte=train_test_split((X-mu)/sd,y,test_size=.25,random_state=1)
m=LogisticRegression(max_iter=1000).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
auc=roc_auc_score(yte,p); top=np.argsort(-p)[:len(p)//10]; catch=yte[top].sum()/yte.sum()
res=dict(names=names,mu=mu.tolist(),sd=sd.tolist(),coef=m.coef_[0].tolist(),b=float(m.intercept_[0]),
 auc=round(auc,3),top10_catch=round(float(catch),3),base_rate=round(float(y.mean()),3),n=N)
json.dump(res,open("model.json","w"),indent=1); print(res)
# sample csv
import random; random.seed(7)
F="Aarav Ananya Rohan Priya Kunal Sneha Vikas Pooja Arjun Neha Rahul Kavya Aditya Riya Sahil Isha Manish Tanvi Yash Simran Deepak Muskan Harsh Nikita Gaurav Khushi Abhay Divya Ritesh Sakshi".split()
L="Sharma Verma Patel Yadav Joshi Malviya Rathore Chouhan Jain Mishra".split()
T=["Maths: Mensuration","Maths: Fractions","Maths: Linear equations","Science: Force & pressure","Science: Sound","English: Tenses","Hindi: Sandhi","Maths: Exponents"]
rows=["student_id,name,class,week,attendance_pct,test_marks,homework_pct,fee_pending_days,weak_topic"]
for i,f in enumerate(F):
    bad=i%7==3 or i%11==5 or f=="Rahul"; base=random.uniform(.65,.97); m0=random.uniform(45,90)
    for w in range(11,15):
        k=w-11; a=base-(.3 if bad and k>=2 else 0); mk=m0-(7*k if bad else random.uniform(-3,3))
        h=random.uniform(.6,1)-(.4 if bad else 0)
        if f=="Rahul": a=[.97,.97,.49,.49][k]; mk=[79,78,67,56][k]; h=.35
        rows.append(f"S{i+1:03d},{f} {L[i%10]},8-A,{w},{round(a*100)},{round(mk)},{round(max(h,.1)*100)},{45 if i%9==4 else 0},{'Maths: Mensuration' if f=='Rahul' else random.choice(T)}")
open("sample_students.csv","w").write("\n".join(rows))
