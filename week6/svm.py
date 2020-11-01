from sklearn.svm import LinearSVC
from sklearn import datasets
from mlxtend.plotting import category_scatter
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X,y = datasets.make_blobs(random_state=42)

df = pd.DataFrame(X, columns = ['feature0','feature1'])
df['class'] = y


flg = category_scatter(x='feature0',y='feature1', label_col='class',data=df,legend_loc='lower right')
# flg.show()

# 5. 모델링
linear_svm=LinearSVC().fit(X,y) # C=1.0

# 6.1 클래스별 계수 확인
linear_svm.coef_

# 6.2 클래스별 절편
linear_svm.intercept_


line = np.linspace(-15, 15) # 지정된 간격 동안 균일 한 간격의 숫자를 반환합니다.

fig, ax = plt.subplots()
scatter = ax.scatter(X[:,0], X[:,1],c = y, s = 50)

for coef, intercept, color in zip(linear_svm.coef_,
                                 linear_svm.intercept_,
                                 ["black","blue","red"]):
    ax.plot(line, -(line*coef[0]+intercept)/coef[1], c = color)
ax.set_ylim(-10, 15)
ax.set_xlim(-10, 8)
ax.legend(*scatter.legend_elements(), loc = "best", title = "class")
plt.show()

plot_decision_regions(X=X, y=y, clf=linear_svm, legend=4)
plt.title('SVC Decision Region Boundary', size=15)

plt.show()