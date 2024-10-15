from sklearn.metrics import roc_auc_score
gold = [1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,0,1,1,0]
pred = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0]
pred1 = [0.8, 0.8, 0.9, 0.8, 0.65, 0.8, 0.7, 0.9, 0.85, 0.8, 0.65, 0.9, 0.8, 0.85, 0.8, 0.65, 0.68, 0.86, 0.94, 0.72]
pred2 = [p/2 for p in pred1]
print(roc_auc_score(gold, pred))
print(roc_auc_score(gold, pred1))
print(roc_auc_score(gold, pred2))